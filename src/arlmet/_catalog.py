"""
Catalog a directory of ARL files by their headers.

:func:`catalog` opens each file in a directory once, keeps what its
index records say (source, grid, first and last valid time, time step) in a
small table, and caches the table in a CSV file. :meth:`Catalog.covering`
then finds the files that cover a time window without knowing how the files
are named.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from pathlib import Path
from stat import S_ISREG
from time import monotonic, sleep
from typing import Any

import pandas as pd

from arlmet._exceptions import ARLFormatError
from arlmet._file import File, _at, _span
from arlmet._io import atomic_output, make_lock
from arlmet._time import ensure_timestamp

__all__ = ["Catalog", "catalog"]

logger = logging.getLogger(__name__)

#: Name of the cache file written beside the files it lists.
CACHE_NAME = ".arlmet-catalog.csv"

#: Environment variable naming the folder for caches that cannot sit beside
#: their directory.
CACHE_ENV = "ARLMET_CACHE_DIR"

# Files read at once, and how many are read between saves of the cache.
_READERS = 16
_SAVE_EVERY = 500

# A process waiting on another's catalog looks every _POLL seconds, and takes
# the lock over when neither it nor the cache has changed for _STALL seconds (up
# to a fifth longer, at random, so waiters do not all act at once). The builder
# saves the cache every 500 files, which took about 80 seconds on a busy shared
# filesystem; another NFS client can see the new file up to a minute late.
_POLL = 2.0
_STALL = 300.0

# Columns of the public table, and of the cache file, which adds what tells
# whether a file changed since it was read.
_TABLE_COLUMNS = ["path", "source", "grid", "start", "end", "step", "error"]
_CACHE_COLUMNS = [
    "path",
    "size",
    "mtime_ns",
    "source",
    "grid",
    "start",
    "end",
    "step",
    "error",
]


# A row that says nothing about a file yet; the size and time of a link that
# cannot be followed stay 0, so it is looked at again on every call.
_EMPTY: dict[str, Any] = {
    "size": 0,
    "mtime_ns": 0,
    "source": "",
    "grid": "",
    "start": pd.NaT,
    "end": pd.NaT,
    "step": pd.NaT,
    "error": "",
}


def _user_cache_dir() -> Path:
    """
    Return the folder for catalog caches that cannot sit beside their directory.

    ``$ARLMET_CACHE_DIR`` when it is set, else ``arlmet`` in the platform's
    user cache folder: ``$XDG_CACHE_HOME`` or ``~/.cache`` on Linux,
    ``~/Library/Caches`` on macOS, ``%LOCALAPPDATA%`` on Windows.
    """
    env = os.environ.get(CACHE_ENV)
    if env:
        return Path(env).expanduser()
    try:
        if sys.platform == "win32":
            base = os.environ.get("LOCALAPPDATA")
            if not base:
                raise RuntimeError("LOCALAPPDATA is not set")
            root = Path(base)
        elif sys.platform == "darwin":
            root = Path.home() / "Library" / "Caches"
        else:
            xdg = os.environ.get("XDG_CACHE_HOME")
            root = Path(xdg) if xdg else Path.home() / ".cache"
    except RuntimeError as exc:
        raise ValueError(
            f"No user cache folder for the catalog ({exc}). Pass cache_dir= or "
            f"set {CACHE_ENV}."
        ) from exc
    return root / "arlmet"


def _cache_path(
    directory: Path, cache_dir: str | os.PathLike[str] | None, recursive: bool
) -> Path:
    """
    Return the cache file for *directory*.

    It is ``.arlmet-catalog.csv`` in *directory* when that is writable;
    otherwise, or when *cache_dir* is given, a file named after the
    directory in *cache_dir* or the user cache folder. A recursive catalog has
    its own file, ending in ``-recursive``, so the two never replace each
    other's rows.
    """
    suffix = "-recursive" if recursive else ""
    if cache_dir is None and os.access(directory, os.W_OK):
        return _published_path(directory, recursive)
    folder = (
        Path(cache_dir).expanduser()
        if cache_dir is not None
        else _user_cache_dir() / "catalog"
    )
    digest = hashlib.sha256(str(directory.resolve()).encode()).hexdigest()[:12]
    return folder / f"{directory.name or 'root'}-{digest}{suffix}.csv"


def _published_path(directory: Path, recursive: bool) -> Path:
    """Return the table the folder's maintainer may keep beside its files."""
    suffix = "-recursive" if recursive else ""
    return directory / f"{CACHE_NAME.removesuffix('.csv')}{suffix}.csv"


def _read_file(path: Path) -> dict[str, Any]:
    """
    Return what *path*'s index records say, as one row of the cache.

    A file that cannot be opened (an empty one included) or whose time steps
    are not evenly spaced gets the reason in ``error``.
    """
    row: dict[str, Any] = dict(_EMPTY)
    try:
        with File(path) as met:
            times = met.times
            row["source"] = met.source
            row["grid"] = repr(met.grid)
    except (ARLFormatError, OSError) as exc:
        row["error"] = str(exc)
        return row

    row["start"], row["end"] = times[0], times[-1]
    if len(times) > 1:
        step = times[1] - times[0]
        for before, after in pairwise(times):
            if after - before != step:
                row["error"] = (
                    f"Its time steps are not evenly spaced: {_at(before)} to "
                    f"{_at(after)} is {_span(after - before)}; the first two "
                    f"are {_span(step)} apart."
                )
                return row
        row["step"] = step
    return row


def _read_cache(cache: Path) -> dict[str, dict[str, Any]]:
    """Return the cached rows by real path, or none when the cache is missing or of another layout."""
    if not cache.is_file():
        return {}
    try:
        table = pd.read_csv(cache, dtype=str, keep_default_na=False)
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring the unreadable catalog cache %s: %s", cache, exc)
        return {}
    if list(table.columns) != _CACHE_COLUMNS:
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for row in table.to_dict("records"):
        rows[str(row["path"])] = {
            "size": int(row["size"]),
            "mtime_ns": int(row["mtime_ns"]),
            "source": row["source"],
            "grid": row["grid"],
            "start": pd.Timestamp(row["start"]) if row["start"] else pd.NaT,
            "end": pd.Timestamp(row["end"]) if row["end"] else pd.NaT,
            "step": pd.Timedelta(row["step"]) if row["step"] else pd.NaT,
            "error": row["error"],
        }
    return rows


def _write_cache(cache: Path, rows: dict[str, dict[str, Any]]) -> None:
    """Write *rows* to *cache*, replacing it only once the new file is complete."""
    table = pd.DataFrame(
        [{"path": key, **row} for key, row in rows.items()],
        columns=_CACHE_COLUMNS,
    )
    try:
        with atomic_output(cache) as tmp:
            table.to_csv(tmp, index=False)
    except OSError as exc:
        raise _cache_error(cache, exc) from exc


def _cache_error(cache: Path, exc: OSError) -> OSError:
    """Return the error for a cache that cannot be written, saying how to choose another place."""
    return OSError(
        f"Cannot write the catalog cache {cache}: {exc}. Pass cache_dir= "
        f"or set {CACHE_ENV} to a folder you can write to."
    )


def _take_known(
    to_read: dict[str, os.stat_result],
    cached: dict[str, dict[str, Any]],
    rows: dict[str, dict[str, Any]],
) -> None:
    """Move each file of *to_read* whose size and modification time match its row in *cached* to *rows*."""
    for key, stat in list(to_read.items()):
        known = cached.get(key)
        if (
            known is not None
            and known["size"] == stat.st_size
            and known["mtime_ns"] == stat.st_mtime_ns
        ):
            rows[key] = known
            del to_read[key]


def _wait_for_build(lock: Path, cache: Path) -> None:
    """
    Wait while another process holds *lock* and opens files for *cache*.

    It returns once the lock is gone. A lock with neither it nor *cache* changing
    for the stall limit belongs to a process that died or hung: it is removed,
    so this one can take it. As for a download's lock, time is measured on this
    process's own clock, so no two clocks have to agree.
    """
    limit = _STALL * random.uniform(1, 1.2)
    seen, since = None, monotonic()
    logger.info("Waiting for another process's catalog cache %s", cache)
    while True:
        try:
            held = lock.stat().st_ino
        except OSError:  # the lock went while we looked: try for it
            return
        try:
            saved = cache.stat()
            now = (held, saved.st_ino, saved.st_size, saved.st_mtime_ns)
        except OSError:  # not saved yet
            now = (held,)
        if now != seen:
            seen, since = now, monotonic()
        elif monotonic() - since > limit:
            logger.warning("No progress on the catalog cache %s; taking it over", cache)
            shutil.rmtree(lock, ignore_errors=True)
            return
        sleep(_POLL)


def _read_files(
    directory: Path,
    cache: Path,
    to_read: dict[str, os.stat_result],
    rows: dict[str, dict[str, Any]],
) -> None:
    """Open the files of *to_read*, add their rows to *rows*, and save *cache*."""
    if not to_read:
        return
    logger.info(
        "Reading %d of %d files in %s",
        len(to_read),
        len(to_read) + len(rows),
        directory,
    )
    # Opening a file waits on a few small reads, which on a network
    # filesystem take far longer than parsing them, so several are read
    # at once. The cache is saved as rows come in, so an interrupted
    # first catalog of a large archive keeps what it read.
    with ThreadPoolExecutor(_READERS) as pool:
        read = pool.map(_read_file, [Path(key) for key in to_read])
        for n, ((key, stat), row) in enumerate(
            zip(to_read.items(), read, strict=True), start=1
        ):
            rows[key] = {**row, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            if n % _SAVE_EVERY == 0:
                logger.info("Read %d of %d files", n, len(to_read))
                _write_cache(cache, rows)
    _write_cache(cache, rows)


def _set_aside(name: str) -> bool:
    """
    Whether a file or folder *name* is skipped, though its header may pass.

    Hidden names (``.``: arl-met's own partial outputs and the cache),
    backup copies (``name~``, ``name.~1~``), lock files (``name.lock``, ``name.lock.1``), a
    download cut short (``.partial``), and a copy set aside as damaged
    (``.bad``), as some archives keep beside the file that replaced it.
    """
    return (
        name.startswith(".")
        or name.endswith(("~", ".partial", ".bad"))
        or name.endswith(".lock")
        or ".lock." in name
    )


def _walk(folder: Path, recursive: bool) -> list[Path]:
    """
    Return the files and links in *folder*, in name order, and in its subfolders with *recursive*.

    Hidden names (``.``), backup copies (``~``) and copies set aside
    (:func:`_set_aside`) are left out, and links to
    folders are not followed, so a link loop cannot make the walk go round.
    """
    found: list[Path] = []
    with os.scandir(folder) as entries:
        for entry in sorted(entries, key=lambda e: e.name):
            if _set_aside(entry.name):
                continue
            if entry.is_dir(follow_symlinks=False):
                if recursive:
                    found += _walk(Path(entry.path), recursive)
            elif entry.is_symlink() or entry.is_file(follow_symlinks=False):
                found.append(Path(entry.path))
    return found


def catalog(
    directory: str | os.PathLike[str],
    *,
    recursive: bool = False,
    cache_dir: str | os.PathLike[str] | None = None,
) -> Catalog:
    """
    Catalog the ARL files in a directory by their headers.

    Each file in *directory* is opened once, and what its index records say
    is kept: its source, grid, first and last valid time, and time step. The
    table is cached, so a later call reads again only the files that are new
    or changed (by size or modification time) and drops the files that are
    gone. Use :meth:`Catalog.covering` to find the files for a time
    window.

    Links to files are followed, and each real file is one row, shown by its
    real path (:meth:`pathlib.Path.resolve`) however many links reach it; of
    hard links to one file, the first by name is the row. A
    link that cannot be followed (its target is gone, or links loop) is a row
    with the reason in ``error``. Links to folders are not followed. Hidden
    files and folders (names starting with ``.``), backup copies (names
    ending in ``~``), lock files (names ending in ``.lock`` or holding ``.lock.``) and copies set
    aside (names ending in ``.partial`` or ``.bad``) are skipped. A file that is not ARL, or is damaged, is
    kept with the reason in ``error`` and is not used to cover a window.

    Files are opened 16 at a time: on a network filesystem each one waits on
    a few small reads. The cache is saved every 500 files, so an interrupted
    first catalog of a large archive keeps what it read. Processes that start
    on the same folder at once, such as the tasks of a Slurm job array, open
    its files one process at a time: the first holds a lock folder beside the
    cache (its name ending in ``.lock``) and the others wait, then read the
    table it saved. A lock with no progress for 5 minutes is taken over.

    When the cache is not beside the files (*directory* is read-only, or
    *cache_dir* is given) and the folder holds the table its maintainer saved
    there by running ``catalog`` on it, that table is the starting point,
    read-only: a file whose size and modification time match its row is not
    opened, and the rest are read. Rows already in your own cache win. A stale
    published table therefore costs a few reads, never a wrong answer.

    Parameters
    ----------
    directory : path-like
        The folder of ARL files.
    recursive : bool, default False
        Also read every folder below *directory*, as a copy of NOAA's
        ``YYYY/MM/`` layout needs. By default only the files directly in
        *directory* are read.
    cache_dir : path-like, optional
        Folder for the cache file. By default the cache is
        ``.arlmet-catalog.csv`` (``.arlmet-catalog-recursive.csv`` with
        *recursive*) in *directory* when *directory* is writable, and
        otherwise a file named after the directory in ``$ARLMET_CACHE_DIR``,
        or in the platform's user cache folder (``arlmet/catalog`` in
        ``$XDG_CACHE_HOME`` on Linux) when that variable is not set.

    Returns
    -------
    Catalog

    Raises
    ------
    FileNotFoundError
        If *directory* does not exist.
    OSError
        If the cache cannot be written, or a folder cannot be listed.

    Warns
    -----
    ARLFormatWarning
        Whatever opening a file warns, such as a time step repeated byte for
        byte.

    See Also
    --------
    Catalog.covering : The files for a time window.
    File.check : Damage that opening a file does not catch.

    Examples
    --------
    >>> cat = arlmet.catalog("hrrr")  # doctest: +SKIP
    >>> cat.covering("2021-01-14 06:00", "2021-01-15 06:00")  # doctest: +SKIP
    [PosixPath('hrrr/20210114_06-11_hrrr'), ..., PosixPath('hrrr/20210115_06-11_hrrr')]
    """
    directory = Path(directory).expanduser().absolute()
    if not directory.is_dir():
        raise FileNotFoundError(f"No directory {directory}.")
    cache = _cache_path(directory, cache_dir, recursive)
    try:  # before reading any file, so a bad cache folder fails at once
        cache.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise _cache_error(cache, exc) from exc
    own = _read_cache(cache)
    # A cache that is not beside the files starts from the table the folder's
    # maintainer published there, read-only; the user's own rows win.
    published = _published_path(directory, recursive)
    cached = {**_read_cache(published), **own} if cache != published else own

    # Rows are keyed by the file's real path, so a file reached by several
    # paths is read and listed once.
    rows: dict[str, dict[str, Any]] = {}
    to_read: dict[str, os.stat_result] = {}
    listed: set[tuple[int, int]] = set()
    for found in _walk(directory, recursive):
        try:
            real = found.resolve(strict=True)
            stat = real.stat()
        except (OSError, RuntimeError) as exc:  # RuntimeError: a loop, before 3.13
            rows[str(found)] = {**_EMPTY, "error": f"Cannot follow the link: {exc}"}
            continue
        key = str(real)
        # A hard link is the same file under a second name: the first name is kept.
        # Some filesystems report no file id (0), and there every name is a file.
        same_file = (stat.st_dev, stat.st_ino)
        if stat.st_ino and same_file in listed:
            continue
        listed.add(same_file)
        if S_ISREG(stat.st_mode) and key not in rows:
            to_read[key] = stat
    _take_known(to_read, cached, rows)

    # One process at a time opens files for a cache: the first to make the lock
    # folder beside it reads them, and the others wait, then take the rows it
    # saved. A process with nothing to read takes no lock.
    lock = cache.with_name(f"{cache.name}.lock")
    while to_read:
        try:
            mine = make_lock(lock)
        except OSError:  # a cache folder we cannot write: no lock, the save says why
            _read_files(directory, cache, to_read, rows)
            break
        if not mine:
            _wait_for_build(lock, cache)
        # The rows another process saved while this one listed the folder or waited.
        own = _read_cache(cache)
        _take_known(to_read, own, rows)
        if mine:
            try:
                _read_files(directory, cache, to_read, rows)
            finally:
                shutil.rmtree(lock, ignore_errors=True)
            own = rows  # what the cache now holds
            break
    if rows.keys() != own.keys():
        _write_cache(cache, rows)

    # Built column by column so each has its type even when it is all NaT,
    # in path order (as strings: Path order differs between platforms), which
    # the stable sort by start keeps for equal starts.
    rows = dict(sorted(rows.items()))
    values = list(rows.values())
    table = pd.DataFrame(
        {
            "path": [Path(key) for key in rows],
            "source": [row["source"] for row in values],
            "grid": [row["grid"] for row in values],
            "start": pd.to_datetime([row["start"] for row in values]),
            "end": pd.to_datetime([row["end"] for row in values]),
            "step": pd.to_timedelta([row["step"] for row in values]),
            "error": [row["error"] for row in values],
        },
        columns=_TABLE_COLUMNS,
    )
    table = table.sort_values("start", kind="stable", na_position="last")
    return Catalog(directory, table.reset_index(drop=True), cache=cache)


class Catalog:
    """
    The ARL files in a directory, by the times their headers say they hold.

    Made by :func:`arlmet.catalog`.

    Parameters
    ----------
    directory : pathlib.Path
        The folder the catalog reads.
    table : pandas.DataFrame
        One row per file (see :attr:`table`).
    cache : pathlib.Path, optional
        The cache file the table was saved to.
    """

    def __init__(
        self,
        directory: Path,
        table: pd.DataFrame,
        *,
        cache: Path | None = None,
    ):
        self.directory = directory
        self.table = table
        self.cache = cache
        # Each usable file's source and grid, and its valid times.
        files: dict[Path, tuple[tuple[str, str], set[pd.Timestamp]]] = {}
        usable = table[table["error"] == ""]
        for path, source, grid, start, end, step in zip(
            usable["path"],
            usable["source"],
            usable["grid"],
            usable["start"],
            usable["end"],
            usable["step"],
            strict=True,
        ):
            first = ensure_timestamp(start)
            times = (
                {first}
                if pd.isna(step)
                else set(pd.date_range(first, ensure_timestamp(end), freq=step))
            )
            files[Path(path)] = ((source, grid), times)
        # Each valid time, and the files that hold it.
        holders: dict[pd.Timestamp, list[Path]] = {}
        for path, (_, times) in files.items():
            for time in sorted(times):
                holders.setdefault(time, []).append(path)
        # A file whose every time a longer file of its source and grid also
        # holds, such as a short crop beside the whole file, is not used.
        inside = {
            path
            for path, (product, times) in files.items()
            if any(
                files[other][0] == product
                and len(files[other][1]) > len(times)
                and times <= files[other][1]
                for other in holders[min(times)]
            )
        }
        self._holders = {
            time: [path for path in paths if path not in inside]
            for time, paths in holders.items()
        }
        self._times = pd.DatetimeIndex(sorted(holders))

    #: The folder the catalog reads.
    directory: Path

    #: One row per file, sorted by ``start``, then name. The columns are ``path``,
    #: ``source`` (the 4-character ARL source ID), ``grid`` (its short
    #: description), ``start`` and ``end`` (the first and last valid time),
    #: ``step`` (the time between valid times, ``NaT`` for a file of one), and
    #: ``error`` (why the file is not used, or ``""``).
    table: pd.DataFrame

    #: The cache file the table was saved to, or ``None``.
    cache: Path | None

    @property
    def step(self) -> pd.Timedelta | None:
        """
        The time between the valid times of the directory, or None for fewer than two.

        The most common spacing between consecutive valid times of the usable
        files. :meth:`missing` expects a valid time at every step.

        See Also
        --------
        Catalog.missing : The valid times no file holds.
        """
        if len(self._times) < 2:
            return None
        spacing = pd.Series(self._times[1:] - self._times[:-1])
        return pd.Timedelta(spacing.mode().iloc[0])

    def _needed(self, start: Any, end: Any) -> list[pd.Timestamp]:
        """Return the valid times a window needs: every step from the one at or before *start* to the one at or after *end*."""
        start, end = ensure_timestamp(start), ensure_timestamp(end)
        if start > end:
            raise ValueError(f"start {start} is after end {end}.")
        if not len(self._times):
            raise ValueError(f"No usable ARL file in {self.directory}.")
        step = self.step
        if step is None:
            return list(dict.fromkeys([start, end]))
        anchor = self._times[0]
        first = anchor + ((start - anchor) // step) * step
        last = anchor - ((anchor - end) // step) * step
        return list(pd.date_range(first, last, freq=step))

    def missing(self, start: Any, end: Any) -> list[pd.Timestamp]:
        """
        Return the valid times a window needs that no usable file holds.

        A window needs a valid time every :attr:`step`, from the one at or
        before *start* to the one at or after *end*, since HYSPLIT
        interpolates between the two around each time.

        Parameters
        ----------
        start, end : datetime-like
            The window, earlier time first. A time without a time zone is UTC;
            a time-zone-aware one is converted to UTC.

        Returns
        -------
        list of pandas.Timestamp
            Empty when every needed time is held.

        Raises
        ------
        ValueError
            If *start* is after *end*, or no file in the directory is usable.

        Examples
        --------
        >>> cat.missing("2021-01-14 06:00", "2021-01-14 08:00")  # doctest: +SKIP
        [Timestamp('2021-01-14 07:00:00')]
        """
        return [t for t in self._needed(start, end) if t not in self._holders]

    def covering(self, start: Any, end: Any, *, missing_ok: bool = False) -> list[Path]:
        """
        Return the files that cover a window, in time order.

        The files hold every valid time the window needs (:meth:`missing`):
        one every :attr:`step` from the one at or before *start* to the one at
        or after *end*. Files whose ``error`` is set are not used, and neither
        is a file whose every valid time a longer file of the same source and
        grid also holds, such as a short crop kept beside the whole file.

        Parameters
        ----------
        start, end : datetime-like
            The window, earlier time first. A time without a time zone is UTC;
            a time-zone-aware one is converted to UTC.
        missing_ok : bool, default False
            Return the files for the times that are held, without raising
            when some are in no file. :meth:`missing` lists the rest, so you
            can find them elsewhere.

        Returns
        -------
        list of pathlib.Path

        Raises
        ------
        ValueError
            If a valid time the window needs is in no file (the message lists
            them) and *missing_ok* is false, if two files hold one of them
            (the message names both), if *start* is after *end*, or if no
            file is usable.

        Examples
        --------
        >>> cat.covering("2021-01-14 06:00", "2021-01-14 08:00")  # doctest: +SKIP
        [PosixPath('hrrr/20210114_06-11_hrrr')]
        """
        needed = self._needed(start, end)
        missing = [t for t in needed if t not in self._holders]
        if missing and not missing_ok:
            listed = ", ".join(_at(t) for t in missing[:10])
            if len(missing) > 10:
                listed += f" and {len(missing) - 10} more"
            raise ValueError(
                f"No file in {self.directory} holds {listed}, which "
                f"{_at(needed[0])} to {_at(needed[-1])} needs."
            )
        files: list[Path] = []
        for time in needed:
            holders = self._holders.get(time, [])
            if not holders:
                continue
            if len(holders) > 1:
                raise ValueError(
                    f"{_at(time)} is in two files, {holders[0]} and "
                    f"{holders[1]}. Move one of them out of {self.directory}."
                )
            if holders[0] not in files:
                files.append(holders[0])
        return files

    def __len__(self) -> int:
        """Return the number of files in the catalog."""
        return len(self.table)

    def __repr__(self) -> str:
        """Return the directory, the number of files and the table, with file names for paths."""
        n_bad = int((self.table["error"] != "").sum())
        names = self.table.assign(path=[path.name for path in self.table["path"]])
        return (
            f"Catalog({str(self.directory)!r}: {len(self)} files, "
            f"{n_bad} not used)\n{names!r}"
        )
