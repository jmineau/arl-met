"""
NOAA ARL meteorology archives: download ARL files by product and date.

Each archive class encodes the filename convention, path layout, and
approximate spatial extent for one ARL-formatted met product hosted on
the NOAA ARL public archives.

Mirrors
-------
"s3"   : AWS S3 (noaa-oar-arl-hysplit-pds, anonymous) — recommended
"ftp"  : NOAA ARL FTP (ftp.arl.noaa.gov, anonymous, 2-connection limit)
"http" : NOAA READY web (www.ready.noaa.gov/data/archives)

Every archive is registered by its short name in :data:`ARCHIVES`, and
:func:`get_archive` builds one by name, which suits config files and CLIs::

    >>> from arlmet.archives import ARCHIVES, get_archive
    >>> sorted(ARCHIVES)[:3]
    ['gdas0p5', 'gdas1', 'gfs0p25']
    >>> get_archive("nams", domain="ak")
    NAMSArchive(domain='ak')

Example
-------
>>> from arlmet.archives import HRRRArchive
>>> archive = HRRRArchive()
>>> files = archive.fetch(
...     "2024-07-18", "2024-07-19", dest_dir="./met/"
... )  # doctest: +SKIP

>>> # Crop to domain on download (recommended due to large file sizes)
>>> files = archive.fetch(  # doctest: +SKIP
...     "2024-07-18",
...     "2024-07-19",
...     dest_dir="./met/",
...     bbox=(-114.0, 39.0, -110.0, 42.0),
... )

Requires ``fsspec`` (and ``s3fs`` for the S3 mirror).
Install with: ``pip install arlmet[archives]``
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import shutil
import socket
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO, ClassVar, Literal, NamedTuple, cast

import pandas as pd
from typing_extensions import override

from arlmet._time import ensure_timestamp

logger = logging.getLogger(__name__)

_MONTH_CODES: tuple[str, ...] = (
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
)


def _bbox_value_tag(value: float) -> str:
    """
    Return one bbox coordinate as it appears in a cached file's name.

    Values with at most two decimals keep the original ``.2f`` form
    (``-112.0`` → ``"-112.00"``), so caches named before finer bboxes were
    supported still match. Anything finer uses the shortest exact repr
    (``40.125`` → ``"40.125"``), so bboxes that differ past the second
    decimal never share a cache file. Differences under 1e-9 degrees are
    treated as float noise, not a different bbox.
    """
    value = float(value)
    # Tolerate float noise from arithmetic like 40.1 - 0.05 (40.050000000000004)
    # so those bboxes keep their old two-decimal name.
    if math.isclose(value, round(value, 2), rel_tol=0.0, abs_tol=1e-9):
        return f"{value:.2f}"
    return repr(value)


def _partial_path(dest: Path) -> Path:
    """
    Return a unique hidden temp path next to *dest*, ``.<name>.<random>.partial``.

    Files are written here first and moved onto *dest* with :func:`os.replace`
    once complete, so an interrupted fetch never leaves a truncated file under
    the cached name. The random part keeps concurrent fetches of the same file
    from writing to the same temp path.
    """
    return dest.with_name(f".{dest.name}.{uuid.uuid4().hex[:12]}.partial")


def _level_ranges(levels: list[int]) -> str:
    """Return sorted level indices as runs, e.g. ``[0, 1, 2, 5]`` → ``"0-2_5"``."""
    runs: list[list[int]] = []
    for level in sorted(set(levels)):
        if runs and level == runs[-1][-1] + 1:
            runs[-1].append(level)
        else:
            runs.append([level])
    return "_".join(
        str(run[0]) if len(run) == 1 else f"{run[0]}-{run[-1]}" for run in runs
    )


#: NOAA's list of the files in a product, with their MD5 sums, at its top folder.
_LISTING_NAME = "listing.md5.txt"
#: What ``sync`` knows about each local file, in the mirror's top folder.
_MANIFEST_NAME = "arlmet-manifest.tsv"
#: The folder ``sync`` makes (``mkdir`` is atomic on NFS) while it runs.
_LOCK_NAME = ".arlmet-sync.lock"

_MD5_LINE = re.compile(r"^([0-9a-fA-F]{32}) [ *](.+)$")


def _parse_md5_lines(text: str) -> dict[str, str]:
    """
    Return ``{name: md5}`` from ``md5sum`` output (``md5  name`` per line).

    Lines in any other form (a blank line, a half-written last line) are
    skipped. A name that appears twice keeps its last line.
    """
    entries: dict[str, str] = {}
    for line in text.splitlines():
        match = _MD5_LINE.match(line)
        if match:
            entries[match.group(2)] = match.group(1).lower()
    return entries


def _md5_of_file(path: Path) -> str:
    """Return the MD5 hex digest of the file at *path* (read in blocks)."""
    with open(path, "rb") as f:
        digest = hashlib.file_digest(f, lambda: hashlib.md5(usedforsecurity=False))
    return digest.hexdigest()


class _HashingWriter:
    """Wrap a binary file so the MD5 of everything written to it is kept."""

    def __init__(self, file: BinaryIO) -> None:
        self._file = file
        self.md5 = hashlib.md5(usedforsecurity=False)

    def write(self, data: bytes) -> int:
        """Hash *data*, then write it."""
        self.md5.update(data)
        return self._file.write(data)


class _ChecksumError(OSError):
    """A download whose MD5 is not the one NOAA lists."""


def _raise(error: OSError) -> None:
    """``os.walk`` ignores errors unless told otherwise; a folder it cannot read must stop ``sync``."""
    raise error


def _local_files(root: Path) -> dict[str, list[str]]:
    """
    Return every file under *root* as ``{file name: [relative paths]}``.

    Hidden files and folders (``.partial`` files, the lock) and the manifest
    are left out; links to folders are followed. A folder that cannot be read
    raises, since a file missed here would be downloaded again. The paths of
    a name are sorted and use ``/``.
    """
    found: dict[str, list[str]] = {}
    if not root.is_dir():
        return found
    for folder, dirs, files in os.walk(root, followlinks=True, onerror=_raise):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in files:
            if name.startswith(".") or name == _MANIFEST_NAME:
                continue
            rel = (Path(folder) / name).relative_to(root).as_posix()
            found.setdefault(name, []).append(rel)
    for rels in found.values():
        rels.sort()
    return found


class _Checked(NamedTuple):
    """What ``sync`` recorded about one local file when it last looked at it."""

    md5: str  #: the file's own MD5
    size: int  #: its size in bytes, as it was then
    mtime_ns: int  #: its modification time in nanoseconds, as it was then
    noaa_md5: str  #: the MD5 NOAA listed for it then

    def line(self, rel: str) -> str:
        """Return the manifest line: ``md5 TAB noaa_md5 TAB size TAB mtime_ns TAB path``."""
        return f"{self.md5}\t{self.noaa_md5}\t{self.size}\t{self.mtime_ns}\t{rel}\n"


def _read_manifest(root: Path) -> dict[str, _Checked]:
    """
    Return ``{relative path: what was recorded}`` from the manifest of *root*.

    Lines that do not have the five tab-separated fields (such as a half-written
    last line) are skipped. A path that appears twice keeps its last line.
    """
    path = root / _MANIFEST_NAME
    entries: dict[str, _Checked] = {}
    if not path.is_file():
        return entries
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t", 4)
        if len(fields) != 5 or not fields[2].isdigit() or not fields[3].isdigit():
            continue
        md5, noaa_md5, size, mtime_ns, rel = fields
        entries[rel] = _Checked(md5, int(size), int(mtime_ns), noaa_md5)
    return entries


def _append_manifest(root: Path, rel: str, checked: _Checked) -> None:
    """Add one line to the manifest, so a killed run keeps its progress."""
    with open(root / _MANIFEST_NAME, "a", encoding="utf-8") as f:
        f.write(checked.line(rel))


def _write_manifest(
    root: Path, manifest: Mapping[str, _Checked], keep: set[str]
) -> None:
    """Rewrite the manifest sorted by path, with only the files in *keep*."""
    path = root / _MANIFEST_NAME
    tmp = _partial_path(path)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            for rel in sorted(manifest):
                if rel in keep:
                    f.write(manifest[rel].line(rel))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


@contextmanager
def _sync_lock(root: Path) -> Iterator[None]:
    """
    Hold the lock folder of *root* while a sync runs.

    ``mkdir`` is atomic, also on NFS, where a ``flock`` may not be seen by
    another host. The folder holds a file naming the host, process and start
    time, and is removed on the way out, also after an error. A run killed
    without a chance to clean up (``kill -9``, a lost node) leaves it behind.
    """
    lock = root / _LOCK_NAME
    try:
        lock.mkdir()
    except FileExistsError:
        try:
            owner = (lock / "owner").read_text(encoding="utf-8").strip()
        except OSError:
            owner = "owner unknown"
        raise RuntimeError(
            f"Another sync holds the lock {lock} ({owner}). If none is running, "
            "delete that folder."
        ) from None
    try:
        started = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        (lock / "owner").write_text(
            f"host {socket.gethostname()}, process {os.getpid()}, started {started}\n",
            encoding="utf-8",
        )
        yield
    finally:
        shutil.rmtree(lock, ignore_errors=True)


def _utc_naive(value: pd.Timestamp | str) -> pd.Timestamp:
    """Return *value* as a timestamp without a time zone, in UTC (as archive dates are)."""
    ts = ensure_timestamp(value)
    return ts if ts.tzinfo is None else ts.tz_convert("UTC").tz_localize(None)


@dataclass
class SyncResult:
    """
    What :meth:`Archive.sync` did, as lists of local file paths.

    Attributes
    ----------
    downloaded : list[pathlib.Path]
        Files fetched this run and checked against NOAA's MD5: the ones
        missing locally and the ones NOAA has corrected since they were
        checked (replaced in place).
    adopted : list[pathlib.Path]
        Files already in the folder that were read this run (new to the
        manifest, or changed since it recorded them) and whose MD5 is NOAA's.
    unchanged : list[pathlib.Path]
        Files the manifest records as NOAA's, with the size and modification
        time it recorded: not read.
    mismatched : list[pathlib.Path]
        Files already in the folder whose MD5 is not NOAA's (or that cannot
        be read). Reported on every run, from the manifest without reading
        them, and left alone until NOAA's file changes.
    failed : list[pathlib.Path]
        Files that could not be fetched or whose download did not match
        NOAA's MD5. Nothing is left under these paths but what was there
        before (the reason is logged).
    dry_run : bool
        True when nothing was changed. ``downloaded`` then lists what a real
        run would fetch, and ``adopted`` the files it would read and check;
        ``mismatched`` and ``failed`` are empty.

    Examples
    --------
    >>> from arlmet.archives import SyncResult
    >>> SyncResult(dry_run=True)
    SyncResult(downloaded=0, adopted=0, unchanged=0, mismatched=0, failed=0, dry_run=True)
    """

    downloaded: list[Path] = field(default_factory=list)
    adopted: list[Path] = field(default_factory=list)
    unchanged: list[Path] = field(default_factory=list)
    mismatched: list[Path] = field(default_factory=list)
    failed: list[Path] = field(default_factory=list)
    dry_run: bool = False

    @override
    def __repr__(self) -> str:
        counts = ", ".join(
            f"{name}={len(getattr(self, name))}"
            for name in ("downloaded", "adopted", "unchanged", "mismatched", "failed")
        )
        return f"SyncResult({counts}, dry_run={self.dry_run})"


_REGISTRY: dict[str, type[Archive]] = {}

#: Every registered archive class by its :attr:`Archive.name`
#: (read-only). Subclasses of :class:`Archive` that set ``name``
#: are added when they are defined, including ones defined outside arlmet.
ARCHIVES: Mapping[str, type[Archive]] = MappingProxyType(_REGISTRY)


def get_archive(name: str, **options: Any) -> Archive:
    """
    Build a meteorology archive from its short name.

    Parameters
    ----------
    name : str
        A key of :data:`ARCHIVES`, e.g. ``"hrrr"``, ``"nam12"``, ``"gdas1"``.
    **options
        Passed to the archive's constructor, e.g. ``domain="ak"`` for
        ``"nams"``.

    Returns
    -------
    Archive
        A new instance of the named archive.

    Raises
    ------
    ValueError
        If ``name`` is not a registered archive.
    TypeError
        If the archive does not accept ``options``.

    Examples
    --------
    >>> from arlmet.archives import get_archive
    >>> files = get_archive("hrrr").fetch(  # doctest: +SKIP
    ...     "2024-07-18", "2024-07-19", dest_dir="./met/"
    ... )
    """
    try:
        cls = ARCHIVES[name]
    except KeyError:
        raise ValueError(
            f"Unknown meteorology archive {name!r}. Available: {sorted(ARCHIVES)}."
        ) from None
    return cls(**options)


class Archive(ABC):
    """
    Abstract base class for NOAA ARL meteorology archives.

    An archive is one ARL-formatted product in NOAA's archive (e.g. HRRR 3 km):
    its filename convention, path layout, and start date. Download files with
    :meth:`fetch`, or keep a whole local folder in step with NOAA's with
    :meth:`sync`, choosing the ``mirror`` (server) to download from.

    Subclasses set class-level metadata and implement ``_archive_path()`` to
    encode the filename convention for their product. A subclass that sets
    ``name`` is registered in :data:`ARCHIVES` under that name.

    Attributes
    ----------
    name : str
        Short archive identifier used by callers and config files (stable).
    description : str
        Human-readable product description.
    start_date : pandas.Timestamp
        Earliest date in the archive.
    source : str
        The source id its files' headers carry (:attr:`arlmet.File.source`),
        such as ``"HRRR"``, so a caller can tell which product an archive
        holds without downloading a file. Every file of an archive carries
        the same id, except where a subclass says otherwise.

    Methods
    -------
    paths_for_range(start, end)
        Return archive paths covering the requested inclusive time range.
    fetch(start, end, ...)
        Download or crop local ARL files for the requested time range.
    sync(dest_dir, ...)
        Keep a local folder complete and checked against NOAA's MD5 list.
    """

    # Subclasses must define these
    name: ClassVar[str]
    description: ClassVar[str]
    #: Earliest date available in the NOAA archive.
    start_date: ClassVar[pd.Timestamp]
    #: The source id in the headers of the archive's files.
    source: ClassVar[str]

    S3_BUCKET: ClassVar[str] = "noaa-oar-arl-hysplit-pds"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register each subclass that defines its own ``name`` in :data:`ARCHIVES`."""
        super().__init_subclass__(**kwargs)
        name = cls.__dict__.get("name")
        if name is None:
            return  # an intermediate base class, not an archive
        existing = _REGISTRY.get(name)
        if existing is not None and existing.__qualname__ != cls.__qualname__:
            raise ValueError(
                f"A meteorology archive named {name!r} is already registered "
                f"({existing.__module__}.{existing.__qualname__})."
            )
        _REGISTRY[name] = cls

    FTP_HOST: ClassVar[str] = "ftp.arl.noaa.gov"
    HTTP_BASE: ClassVar[str] = "https://www.ready.noaa.gov/data/archives"

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _archive_path(self, time: pd.Timestamp) -> str:
        """Path within the archive (no leading slash) of the ARL file containing *time*."""

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def paths_for_range(
        self,
        start: pd.Timestamp | str,
        end: pd.Timestamp | str,
    ) -> list[str]:
        """
        Return deduplicated, sorted archive paths covering ``[start, end]``.

        Each path is relative to the archive root and is the same on every
        mirror, e.g. ``"hrrr/2024/07/20240718_00-05_hrrr"``.

        Handles backward trajectories (``start > end``) by normalizing to
        chronological order before scanning.

        Parameters
        ----------
        start, end : pandas.Timestamp or str
            Inclusive time range to cover.

        Returns
        -------
        list[str]
            Unique archive paths in chronological order.

        Raises
        ------
        ValueError
            If the range begins before the archive's ``start_date``.
        """
        t0 = ensure_timestamp(start, floor="h")
        t1 = ensure_timestamp(end, floor="h")
        if t0 > t1:
            t0, t1 = t1, t0

        start_date = self.start_date
        if t0.tzinfo is not None:
            # start_date is naive UTC; match it to a tz-aware request.
            start_date = start_date.tz_localize("UTC")
        if t0 < start_date:
            raise ValueError(
                f"{type(self).__name__} archive begins {start_date:%Y-%m-%d}, "
                f"but the requested range starts at {t0}."
            )

        seen: set[str] = set()
        paths: list[str] = []
        t = t0
        while t <= t1:
            path = self._archive_path(t)
            if path not in seen:
                seen.add(path)
                paths.append(path)
            t = ensure_timestamp(t + pd.Timedelta(hours=1))
        return paths

    def fetch(
        self,
        start: pd.Timestamp | str,
        end: pd.Timestamp | str,
        *,
        dest_dir: Path | str,
        mirror: Literal["s3", "ftp", "http"] = "s3",
        bbox: tuple[float, float, float, float] | None = None,
        levels: Iterable[int] | None = None,
        overwrite: bool = False,
    ) -> list[Path]:
        """
        Download ARL files covering ``[start, end]`` to *dest_dir*.

        Parameters
        ----------
        start, end :
            Time range (inclusive). Backward trajectories (start > end)
            are handled automatically.
        dest_dir :
            Directory to save downloaded files. Created if absent.
        mirror :
            Which copy of the archive to download from — ``"s3"`` (default,
            AWS), ``"ftp"`` (NOAA ARL FTP), or ``"http"`` (NOAA READY web).
        bbox :
            ``(west, south, east, north)`` in degrees. When provided, each
            file is cropped with :func:`arlmet.extract_subset` before
            caching (on a projected grid, to the smallest rectangle of cells
            that holds the box's part on the grid). Strongly recommended for global
            products (GFS, GDAS).
        levels :
            ARL level indices to keep, counted from 0 at the surface. When
            provided, each file keeps only these levels, cropped with
            :func:`arlmet.extract_subset` like *bbox*. All levels are kept
            by default.
        overwrite :
            Re-download even if a matching local file already exists.

        Returns
        -------
        list[Path]
            Local paths to the downloaded (and optionally cropped) files,
            in chronological order.

        Raises
        ------
        ImportError
            If ``fsspec`` is not installed.
        ValueError
            If the range begins before the archive's ``start_date``.

        Notes
        -----
        Files are written to a hidden ``.<name>.<random>.partial`` file in
        *dest_dir* and renamed into place only once complete, so an
        interrupted fetch never leaves a truncated file that a later call
        would reuse. When cropping, the full download is also staged in
        *dest_dir* (not the system temp directory) and removed afterwards.

        Examples
        --------
        >>> from arlmet.archives import HRRRArchive
        >>> archive = HRRRArchive()
        >>> archive.fetch(
        ...     "2024-07-18", "2024-07-19", dest_dir="./met"
        ... )  # doctest: +SKIP
        """
        try:
            import fsspec  # noqa: F401
        except ImportError:
            raise ImportError(
                "fsspec is required for Archive.fetch(). "
                "Install with: pip install arlmet[archives]"
            ) from None

        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        keep = None if levels is None else sorted({int(level) for level in levels})

        results: list[Path] = []
        for archive_path in self.paths_for_range(start, end):
            filename = Path(archive_path).name
            dest = self._dest_path(dest_dir, filename, bbox, keep)

            if not overwrite and dest.exists():
                logger.debug("Using cached %s", dest.name)
                results.append(dest)
                continue

            url = self._url(archive_path, mirror)
            opts = self._storage_options(mirror)
            logger.info("Fetching %s → %s", url, dest.name)

            if bbox is not None or keep is not None:
                self._fetch_and_crop(url, dest, opts, bbox=bbox, levels=keep)
            else:
                self._download(url, dest, opts)

            results.append(dest)

        return results

    def sync(
        self,
        dest_dir: Path | str,
        *,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        mirror: Literal["s3", "ftp", "http"] = "s3",
        dry_run: bool = False,
    ) -> SyncResult:
        """
        Keep the local folder *dest_dir* complete and checked against NOAA's MD5 list.

        Reads the product's ``listing.md5.txt`` from the *mirror*, finds the
        local copy of each listed file anywhere under *dest_dir*, and downloads
        what is missing. Nothing is ever deleted, and files NOAA does not list
        are not touched. *dest_dir* is the product's folder (for HRRR, the one
        that holds ``2024/07/...``), and new files go where the archive's own
        path puts them.

        Parameters
        ----------
        dest_dir :
            The local mirror of the product. Created if absent.
        start, end :
            Sync only the files covering this time range (inclusive). By
            default, the whole archive: from its ``start_date`` to now.
        mirror :
            Which copy of the archive to read the list and the files from,
            as for :meth:`fetch`.
        dry_run :
            Say what would be done without downloading, reading a local file,
            or writing anything (not even the lock or the manifest).

        Returns
        -------
        SyncResult
            The files downloaded, adopted, unchanged, mismatched and failed.

        Raises
        ------
        ImportError
            If ``fsspec`` is not installed.
        ValueError
            If the range begins before the archive's ``start_date``, or the
            mirror's list has no entries.
        RuntimeError
            If another sync holds the lock (see Notes).

        Notes
        -----
        Each listed file is dealt with by what the folder holds:

        - **No local copy**: downloaded to a hidden ``.partial`` file, hashed
          as it streams, and renamed into place only if its MD5 is NOAA's;
          otherwise it is deleted and the file is in ``failed``.
        - **A copy the manifest records as NOAA's** (its size and
          modification time are as recorded): nothing is done and the file is
          not read.
        - **A copy the manifest records as differing from NOAA's**, while
          NOAA's MD5 is still the one recorded: it is in ``mismatched`` and
          left alone, without being read.
        - **A copy NOAA has corrected** (NOAA's MD5 is not the one recorded
          when the copy was checked, and the copy is not NOAA's new file):
          downloaded again the same way, replacing it where it is.
        - **A copy the manifest does not list** (a mirror made another way)
          **or whose size or modification time changed**: its MD5 is
          computed and recorded. If it is NOAA's it is ``adopted``; if not it
          is ``mismatched`` and left alone.

        A copy is found by its name wherever it sits under *dest_dir*, so a
        folder in another layout (all files in one folder, say) is not
        downloaded again; if a name has several copies, the one at the
        archive's own path is used, else the first by path. Folders whose
        names start with a dot are not searched.

        The manifest, ``arlmet-manifest.tsv`` in *dest_dir*, keeps apart what
        the local file is and what NOAA's copy was when it was checked. It has
        one line per file with five tab-separated fields: the file's MD5,
        NOAA's MD5 at the check, the file's size in bytes, its modification
        time in nanoseconds, and its path relative to *dest_dir* (last, so it
        may hold spaces). A file is read only when it is not in the manifest
        or its size or modification time differ from the line; to have a file
        checked again, delete its line. The manifest is written as the run
        goes, so a killed run keeps what it checked.

        A folder ``.arlmet-sync.lock`` made with ``mkdir`` (atomic on NFS)
        stops a second sync and names the host and start time of the first.
        It is removed at the end, also after an error; a run killed without a
        chance to clean up leaves it, and the next run then asks you to
        delete it.

        A file that cannot be fetched or fails its check is logged and the
        run goes on to the next. A file that NOAA lists but this archive has
        no path for (another NAMS domain, say) is skipped.

        Examples
        --------
        >>> from arlmet.archives import HRRRArchive
        >>> result = HRRRArchive().sync("./hrrr", dry_run=True)  # doctest: +SKIP
        >>> result.downloaded  # doctest: +SKIP
        [PosixPath('hrrr/2019/06/20190612_00-05_hrrr'), ...]
        """
        try:
            import fsspec
        except ImportError:
            raise ImportError(
                "fsspec is required for Archive.sync(). "
                "Install with: pip install arlmet[archives]"
            ) from None

        dest_dir = Path(dest_dir)
        first = self.start_date if start is None else _utc_naive(start)
        last = (
            pd.Timestamp.now("UTC").tz_localize(None)
            if end is None
            else _utc_naive(end)
        )
        # The archive's path of each file in the range, by file name: NOAA's list
        # holds bare names.
        paths = {Path(p).name: p for p in self.paths_for_range(first, last)}
        product = self._archive_path(self.start_date).split("/", 1)[0]

        opts = self._storage_options(mirror)
        listing_url = self._url(f"{product}/{_LISTING_NAME}", mirror)
        logger.info("Reading %s", listing_url)
        with fsspec.open(listing_url, "rb", **opts) as f:
            listing = _parse_md5_lines(cast(BinaryIO, f).read().decode("utf-8"))
        if not listing:
            raise ValueError(f"{listing_url} has no 'md5  name' lines.")

        if dry_run:
            return self._sync_files(
                dest_dir, paths, listing, product, mirror, dry_run=True
            )
        dest_dir.mkdir(parents=True, exist_ok=True)
        with _sync_lock(dest_dir):
            return self._sync_files(
                dest_dir, paths, listing, product, mirror, dry_run=False
            )

    def _sync_files(
        self,
        dest_dir: Path,
        paths: Mapping[str, str],
        listing: Mapping[str, str],
        product: str,
        mirror: str,
        *,
        dry_run: bool,
    ) -> SyncResult:
        """
        Do the work of :meth:`sync` once the list is read and the lock held.

        *paths* maps a file name to its archive path (in time order) and
        *listing* maps a file name to NOAA's MD5.
        """
        local = _local_files(dest_dir)
        manifest = _read_manifest(dest_dir)
        present = {rel for rels in local.values() for rel in rels}
        result = SyncResult(dry_run=dry_run)
        opts = self._storage_options(mirror)

        for name, archive_path in paths.items():
            expected = listing.get(name)
            if expected is None:
                continue  # not in NOAA's list (yet): not ours to touch
            own = Path(archive_path).relative_to(product).as_posix()
            copies = local.get(name, [])
            rel = own if own in copies else (copies[0] if copies else None)

            if rel is not None:
                path = dest_dir / rel
                try:
                    stat = path.stat()
                except OSError as err:  # a broken link
                    logger.warning("Cannot read %s: %s", path, err)
                    result.mismatched.append(path)
                    continue
                recorded = manifest.get(rel)
                # The file is read only if it is new to the manifest or has changed
                # (size or modification time) since the manifest recorded it.
                if recorded is not None and (recorded.size, recorded.mtime_ns) == (
                    stat.st_size,
                    stat.st_mtime_ns,
                ):
                    must_read = False
                    md5, noaa_before = recorded.md5, recorded.noaa_md5
                elif dry_run:
                    result.adopted.append(path)  # a real run would read and check it
                    continue
                else:
                    must_read = True
                    try:
                        md5 = _md5_of_file(path)
                    except OSError as err:
                        logger.warning("Cannot read %s: %s", path, err)
                        result.mismatched.append(path)
                        continue
                    noaa_before = expected if recorded is None else recorded.noaa_md5

                if md5 == expected or noaa_before == expected:
                    # NOAA's file is the one checked (or a better copy is in place);
                    # the local file either is it or differs and stays as it is.
                    checked = _Checked(md5, stat.st_size, stat.st_mtime_ns, expected)
                    if checked != recorded and not dry_run:
                        manifest[rel] = checked
                        _append_manifest(dest_dir, rel, checked)
                    if md5 != expected:
                        logger.warning(
                            "%s has MD5 %s, NOAA lists %s; left alone",
                            path,
                            md5,
                            expected,
                        )
                        result.mismatched.append(path)
                    elif must_read:
                        result.adopted.append(path)
                    else:
                        result.unchanged.append(path)
                    continue
                # else: NOAA's MD5 is not the one recorded: NOAA corrected the file.
                if must_read:  # keep what we learned about the local file
                    manifest[rel] = _Checked(
                        md5, stat.st_size, stat.st_mtime_ns, noaa_before
                    )
                    _append_manifest(dest_dir, rel, manifest[rel])

            # Missing, or NOAA has corrected it since the manifest recorded it.
            rel = own if rel is None else rel
            target = dest_dir / rel
            if dry_run:
                result.downloaded.append(target)
                continue
            url = self._url(archive_path, mirror)
            logger.info("Fetching %s → %s", url, target)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                self._download(url, target, opts, expected_md5=expected)
            except OSError as err:
                logger.warning("Could not fetch %s: %s", url, err)
                result.failed.append(target)
                continue
            stat = target.stat()
            manifest[rel] = _Checked(expected, stat.st_size, stat.st_mtime_ns, expected)
            present.add(rel)
            _append_manifest(dest_dir, rel, manifest[rel])
            result.downloaded.append(target)

        if not dry_run:
            _write_manifest(dest_dir, manifest, keep=present)
        logger.info("Synced %s: %r", dest_dir, result)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dest_path(
        self,
        dest_dir: Path,
        filename: str,
        bbox: tuple[float, float, float, float] | None,
        levels: list[int] | None = None,
    ) -> Path:
        """
        Return the local cache path for a downloaded file and optional crop.

        A bbox adds ``.crop_<west>_<south>_<east>_<north>`` to the name, and
        levels add ``.levels_<ranges>`` (``[0, 1, 2, 5]`` is ``.levels_0-2_5``),
        so files cropped differently never share a path. Bbox values are
        written with two decimals (``-112.00``) unless they have more, which
        are kept in full (``-111.925``).
        """
        tag = ""
        if bbox is not None:
            tag += ".crop_" + "_".join(_bbox_value_tag(v) for v in bbox)
        if levels is not None:
            tag += f".levels_{_level_ranges(levels)}"
        return dest_dir / f"{filename}{tag}"

    def _url(self, path: str, mirror: str) -> str:
        """Return the fully qualified remote URL for an archive path."""
        if mirror == "s3":
            return f"s3://{self.S3_BUCKET}/{path}"
        if mirror == "ftp":
            # The FTP mirror has the same layout under /archives/
            return f"ftp://anonymous@{self.FTP_HOST}/archives/{path}"
        if mirror == "http":
            return f"{self.HTTP_BASE}/{path}"
        raise ValueError(f"Unknown mirror {mirror!r}. Choose 's3', 'ftp', or 'http'.")

    def _storage_options(self, mirror: str) -> dict[str, Any]:
        """Return fsspec storage options for the selected mirror."""
        if mirror == "s3":
            return {"anon": True}
        return {}

    def _download(
        self,
        url: str,
        dest: Path,
        opts: dict[str, Any],
        *,
        expected_md5: str | None = None,
    ) -> None:
        """
        Download one ARL file to a temporary path and atomically move it into place.

        With *expected_md5*, the MD5 of the bytes is computed as they stream
        and the file is moved into place only if it matches; otherwise
        :class:`_ChecksumError` is raised and nothing is left behind.
        """
        import fsspec

        tmp = _partial_path(dest)
        try:
            with fsspec.open(url, "rb", **opts) as src, open(tmp, "wb") as dst:
                # fsspec.open() stubs return IO[Any]; "rb"/"wb" mode guarantees BinaryIO.
                out = (
                    dst if expected_md5 is None else _HashingWriter(cast(BinaryIO, dst))
                )
                shutil.copyfileobj(
                    cast(BinaryIO, src),
                    cast(BinaryIO, out),
                    length=8 * 1024 * 1024,
                )
            if isinstance(out, _HashingWriter) and out.md5.hexdigest() != expected_md5:
                raise _ChecksumError(
                    f"{url} has MD5 {out.md5.hexdigest()}, expected {expected_md5}"
                )
            os.replace(tmp, dest)
        finally:
            # Only left behind if the download failed before the replace.
            tmp.unlink(missing_ok=True)

    def _fetch_and_crop(
        self,
        url: str,
        dest: Path,
        opts: dict[str, Any],
        *,
        bbox: tuple[float, float, float, float] | None,
        levels: list[int] | None,
    ) -> None:
        """
        Download one ARL file, crop it to *bbox* and *levels*, and write the cropped copy.

        The full download is staged next to *dest* rather than in the system
        temp directory, which may be too small for multi-GB files (HRRR is
        ~3 GB each). The crop is written to its own temp file and moved onto
        *dest* only once complete (by :func:`arlmet.extract_subset`). The download
        is removed either way.
        """
        from arlmet.ops.subset import extract_subset

        raw = _partial_path(dest)
        try:
            self._download(url, raw, opts)
            # extract_subset writes the crop to a temp file and renames it onto
            # dest only once complete.
            extract_subset(raw, dest, bbox=bbox, levels=levels)
        finally:
            raw.unlink(missing_ok=True)

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


# ---------------------------------------------------------------------------
# Concrete archives
# ---------------------------------------------------------------------------


class HRRRArchive(Archive):
    """
    HRRR 3 km analysis (CONUS, June 2019–present).

    Files cover 6-hour UTC blocks (00–05, 06–11, 12–17, 18–23),
    approximately 3.2 GB each.

    S3: ``s3://noaa-oar-arl-hysplit-pds/hrrr/{year}/{month:02d}/{YYYYMMDD}_{HH}-{HH}_hrrr``

    The archive begins at the 2019-06-12 00Z block; every file, including the
    earliest, lives under the ``{year}/{month:02d}/`` layout above.
    """

    name = "hrrr"
    description = "HRRR 3 km analysis"
    start_date = ensure_timestamp("2019-06-12")
    source = "HRRR"

    _HOURS_PER_FILE: ClassVar[int] = 6

    def _filename(self, time: pd.Timestamp) -> str:
        """Return the HRRR archive filename covering *time*."""
        start_h = (time.hour // self._HOURS_PER_FILE) * self._HOURS_PER_FILE
        end_h = start_h + self._HOURS_PER_FILE - 1
        return f"{time.strftime('%Y%m%d')}_{start_h:02d}-{end_h:02d}_hrrr"

    @override
    def _archive_path(self, time: pd.Timestamp) -> str:
        """Return the archive path of the HRRR file covering *time*."""
        return f"hrrr/{time.year}/{time.month:02d}/{self._filename(time)}"


class NAMArchive(Archive):
    """
    NAM 12 km analysis (North America, May 26, 2007–present).

    One file per calendar day.

    S3: ``s3://noaa-oar-arl-hysplit-pds/nam12/{year}/{month:02d}/{YYYYMMDD}_nam12``
    """

    name = "nam12"
    description = "NAM 12 km analysis"
    start_date = ensure_timestamp("2007-05-26")
    source = "NAM"

    def _filename(self, time: pd.Timestamp) -> str:
        """Return the daily NAM archive filename for *time*."""
        return f"{time.strftime('%Y%m%d')}_nam12"

    @override
    def _archive_path(self, time: pd.Timestamp) -> str:
        """Return the archive path of the NAM file covering *time*."""
        return f"nam12/{time.year}/{time.month:02d}/{self._filename(time)}"


class GDASArchive(Archive):
    """
    GDAS 1-degree global analysis (December 2004–present).

    Weekly files (~571 MB each). Week boundaries are fixed per month:
    w1 = days 1–7, w2 = days 8–14, w3 = days 15–21,
    w4 = days 22–28, w5 = days 29–end.

    S3: ``s3://noaa-oar-arl-hysplit-pds/gdas1/{year}/gdas1.{mon}{YY}.w{N}``
    """

    name = "gdas1"
    description = "GDAS 1-degree global analysis"
    start_date = ensure_timestamp("2004-12-01")
    source = "GDAS"

    def _week(self, time: pd.Timestamp) -> int:
        """Return the 1-based archive week within the month for *time*."""
        return (time.day - 1) // 7 + 1

    def _filename(self, time: pd.Timestamp) -> str:
        """Return the weekly GDAS archive filename for *time*."""
        month = _MONTH_CODES[time.month - 1]
        year_2d = time.strftime("%y")
        return f"gdas1.{month}{year_2d}.w{self._week(time)}"

    @override
    def _archive_path(self, time: pd.Timestamp) -> str:
        """Return the archive path of the GDAS file covering *time*."""
        return f"gdas1/{time.year}/{self._filename(time)}"


class GFSArchive(Archive):
    """
    GFS 0.25-degree global analysis (June 13, 2019–present).

    One file per calendar day, approximately 2.7 GB each.
    Cropping with ``bbox=`` on fetch is strongly recommended.

    S3: ``s3://noaa-oar-arl-hysplit-pds/gfs0p25/{year}/{month:02d}/{YYYYMMDD}_gfs0p25``
    """

    name = "gfs0p25"
    description = "GFS 0.25-degree global analysis"
    start_date = ensure_timestamp("2019-06-13")
    source = "GFSQ"

    def _filename(self, time: pd.Timestamp) -> str:
        """Return the daily GFS archive filename for *time*."""
        return f"{time.strftime('%Y%m%d')}_gfs0p25"

    @override
    def _archive_path(self, time: pd.Timestamp) -> str:
        """Return the archive path of the GFS file covering *time*."""
        return f"gfs0p25/{time.year}/{time.month:02d}/{self._filename(time)}"


class NAMSArchive(Archive):
    """
    NAMS hybrid sigma-pressure analysis (CONUS/Alaska/Hawaii, March 2009–present).

    One file per calendar day. Uses hybrid sigma-pressure vertical coordinates
    (flag=4), making it suitable for high-accuracy boundary-layer transport.

    Parameters
    ----------
    domain : {"conus", "ak", "hi"}
        Regional domain — CONUS (default), Alaska, or Hawaii.

    S3: ``s3://noaa-oar-arl-hysplit-pds/nams/{year}/{month:02d}/{YYYYMMDD}_hysplit.t00z.namsa[.AK|.HI]``
    """

    name = "nams"
    description = "NAMS hybrid sigma-pressure analysis"
    start_date = ensure_timestamp("2009-03-22")
    source = "NAMS"

    _DOMAIN_SUFFIXES: ClassVar[dict[str, str]] = {
        "conus": "",
        "ak": ".AK",
        "hi": ".HI",
    }

    def __init__(self, domain: Literal["conus", "ak", "hi"] = "conus") -> None:
        if domain not in self._DOMAIN_SUFFIXES:
            raise ValueError(
                f"domain must be one of {list(self._DOMAIN_SUFFIXES)!r}, got {domain!r}"
            )
        self.domain = domain

    def _filename(self, time: pd.Timestamp) -> str:
        """Return the daily NAMS archive filename for *time* and the selected domain."""
        suffix = self._DOMAIN_SUFFIXES[self.domain]
        return f"{time.strftime('%Y%m%d')}_hysplit.t00z.namsa{suffix}"

    @override
    def _archive_path(self, time: pd.Timestamp) -> str:
        """Return the archive path of the NAMS file covering *time*."""
        return f"nams/{time.year}/{time.month:02d}/{self._filename(time)}"

    @override
    def __repr__(self) -> str:
        return f"NAMSArchive(domain={self.domain!r})"


class ReanalysisArchive(Archive):
    """
    NCEP/NCAR Reanalysis 2.5-degree global (1948–present).

    Monthly files (~500 MB each). Covers the full globe at 2.5-degree
    resolution. Useful for long climatological back-trajectory studies.
    Cropping with ``bbox=`` on fetch is strongly recommended.

    S3: ``s3://noaa-oar-arl-hysplit-pds/reanalysis/{year}/RP{YYYYMM}.gbl``
    """

    name = "reanalysis"
    description = "NCEP/NCAR Reanalysis 2.5-degree global"
    start_date = ensure_timestamp("1948-01-01")
    source = "CDC1"

    def _filename(self, time: pd.Timestamp) -> str:
        """Return the monthly reanalysis archive filename for *time*."""
        return f"RP{time.strftime('%Y%m')}.gbl"

    @override
    def _archive_path(self, time: pd.Timestamp) -> str:
        """Return the archive path of the reanalysis file covering *time*."""
        return f"reanalysis/{time.year}/{self._filename(time)}"


class HRRRv1Archive(Archive):
    """
    HRRR 3 km analysis, version 1 (CONUS, June 15, 2015–2019).

    Files cover 6-hour UTC blocks (00z, 06z, 12z, 18z).
    Superseded by :class:`HRRRArchive` from June 2019 onward.

    S3: ``s3://noaa-oar-arl-hysplit-pds/hrrr.v1/{year}/{month:02d}/hysplit.{YYYYMMDD}.{HH}z.hrrra``
    """

    name = "hrrr.v1"
    description = "HRRR 3 km analysis v1"
    start_date = ensure_timestamp("2015-06-15")
    source = "HRRR"

    _HOURS_PER_FILE: ClassVar[int] = 6

    def _filename(self, time: pd.Timestamp) -> str:
        """Return the legacy HRRR v1 archive filename covering *time*."""
        start_h = (time.hour // self._HOURS_PER_FILE) * self._HOURS_PER_FILE
        return f"hysplit.{time.strftime('%Y%m%d')}.{start_h:02d}z.hrrra"

    @override
    def _archive_path(self, time: pd.Timestamp) -> str:
        """Return the archive path of the HRRR v1 file covering *time*."""
        return f"hrrr.v1/{time.year}/{time.month:02d}/{self._filename(time)}"


class GDAS0p5Archive(Archive):
    """
    GDAS 0.5-degree global analysis (September 2007–mid 2019).

    One file per calendar day. Higher resolution than :class:`GDASArchive`
    (1-degree). Cropping with ``bbox=`` on fetch is strongly recommended.

    S3: ``s3://noaa-oar-arl-hysplit-pds/gdas0p5/{year}/{month:02d}/{YYYYMMDD}_gdas0p5``
    """

    name = "gdas0p5"
    description = "GDAS 0.5-degree global analysis"
    start_date = ensure_timestamp("2007-09-01")
    #: Files before 2013-07-29 carry "GHDA", the id of the same product.
    source = "GFSG"

    def _filename(self, time: pd.Timestamp) -> str:
        """Return the daily GDAS 0.5-degree archive filename for *time*."""
        return f"{time.strftime('%Y%m%d')}_gdas0p5"

    @override
    def _archive_path(self, time: pd.Timestamp) -> str:
        """Return the archive path of the GDAS 0.5-degree file covering *time*."""
        return f"gdas0p5/{time.year}/{time.month:02d}/{self._filename(time)}"


class NARRArchive(Archive):
    """
    NCEP North American Regional Reanalysis (January 1979–2019).

    Monthly files at 32 km resolution over North America. Useful for
    long climatological back-trajectory studies over the continent.
    No file extension.

    S3: ``s3://noaa-oar-arl-hysplit-pds/narr/{year}/NARR{YYYYMM}``
    """

    name = "narr"
    description = "NCEP North American Regional Reanalysis 32 km"
    start_date = ensure_timestamp("1979-01-01")
    source = "NARR"

    def _filename(self, time: pd.Timestamp) -> str:
        """Return the monthly NARR archive filename for *time*."""
        return f"NARR{time.strftime('%Y%m')}"

    @override
    def _archive_path(self, time: pd.Timestamp) -> str:
        """Return the archive path of the NARR file covering *time*."""
        return f"narr/{time.year}/{self._filename(time)}"


__all__ = [
    "ARCHIVES",
    "get_archive",
    "Archive",
    "SyncResult",
    "HRRRArchive",
    "HRRRv1Archive",
    "NAMArchive",
    "NAMSArchive",
    "GDASArchive",
    "GDAS0p5Archive",
    "GFSArchive",
    "NARRArchive",
    "ReanalysisArchive",
]
