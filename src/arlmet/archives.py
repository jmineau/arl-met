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
>>> files = archive.fetch("2024-07-18", "2024-07-19", dest_dir="./met/")

>>> # Crop to domain on download (recommended due to large file sizes)
>>> files = archive.fetch(
...     "2024-07-18",
...     "2024-07-19",
...     dest_dir="./met/",
...     bbox=(-114.0, 39.0, -110.0, 42.0),
... )

Requires ``fsspec`` (and ``s3fs`` for the S3 mirror).
Install with: ``pip install arlmet[archives]``
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, BinaryIO, ClassVar, Literal, cast

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
    >>> files = get_archive("hrrr").fetch("2024-07-18", "2024-07-19", dest_dir="./met/")
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
    :meth:`fetch`, choosing the ``mirror`` (server) to download from.

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
            caching. Strongly recommended for global products (GFS, GDAS).
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
        >>> archive.fetch("2024-07-18", "2024-07-19", dest_dir="./met")
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

    def _download(self, url: str, dest: Path, opts: dict[str, Any]) -> None:
        """Download one ARL file to a temporary path and atomically move it into place."""
        import fsspec

        tmp = _partial_path(dest)
        try:
            with fsspec.open(url, "rb", **opts) as src, open(tmp, "wb") as dst:
                # fsspec.open() stubs return IO[Any]; "rb"/"wb" mode guarantees BinaryIO.
                shutil.copyfileobj(
                    cast(BinaryIO, src),
                    cast(BinaryIO, dst),
                    length=8 * 1024 * 1024,
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
