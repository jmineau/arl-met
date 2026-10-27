"""
Index a directory of ARL files by their headers.

:func:`index` opens each file in a directory once, keeps what its
index records say (source, grid, first and last valid time, time step) in a
small table, and caches the table in a CSV file. :meth:`DirectoryIndex.covering`
then finds the files that cover a time window without knowing how the files
are named.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from pathlib import Path
from stat import S_ISREG
from typing import Any

import pandas as pd

from arlmet._io import atomic_output
from arlmet._time import ensure_timestamp
from arlmet.exceptions import ARLFormatError
from arlmet.file import File, _at, _span

__all__ = ["DirectoryIndex", "index"]

logger = logging.getLogger(__name__)

#: Name of the cache file written beside the files it indexes.
CACHE_NAME = ".arlmet-index.csv"

#: Environment variable naming the folder for caches that cannot sit beside
#: their directory.
CACHE_ENV = "ARLMET_CACHE_DIR"

# Files read at once, and how many are read between saves of the cache.
_READERS = 16
_SAVE_EVERY = 500

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
    Return the folder for index caches that cannot sit beside their directory.

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
            f"No user cache folder for the index ({exc}). Pass cache_dir= or "
            f"set {CACHE_ENV}."
        ) from exc
    return root / "arlmet"


def _cache_path(
    directory: Path, cache_dir: str | os.PathLike[str] | None, recursive: bool
) -> Path:
    """
    Return the cache file for *directory*.

    It is ``.arlmet-index.csv`` in *directory* when that is writable;
    otherwise, or when *cache_dir* is given, a file named after the
    directory in *cache_dir* or the user cache folder. A recursive index has
    its own file, ending in ``-recursive``, so the two never replace each
    other's rows.
    """
    suffix = "-recursive" if recursive else ""
    if cache_dir is None and os.access(directory, os.W_OK):
        return directory / f"{CACHE_NAME.removesuffix('.csv')}{suffix}.csv"
    folder = (
        Path(cache_dir).expanduser()
        if cache_dir is not None
        else _user_cache_dir() / "index"
    )
    digest = hashlib.sha256(str(directory.resolve()).encode()).hexdigest()[:12]
    return folder / f"{directory.name or 'root'}-{digest}{suffix}.csv"


def _read_file(path: Path) -> dict[str, Any]:
    """
    Return what *path*'s index records say, as one row of the cache.

    A file that cannot be opened, holds no time step, or whose time steps
    are not evenly spaced gets the reason in ``error``.
    """
    row: dict[str, Any] = dict(_EMPTY)
    try:
        with File(path) as met:
            times = met.times
            if not times:
                row["error"] = "The file holds no time step."
                return row
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
        logger.warning("Ignoring the unreadable index cache %s: %s", cache, exc)
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
        f"Cannot write the index cache {cache}: {exc}. Pass cache_dir= "
        f"or set {CACHE_ENV} to a folder you can write to."
    )


def _walk(folder: Path, recursive: bool) -> list[Path]:
    """
    Return the files and links in *folder*, in name order, and in its subfolders with *recursive*.

    Hidden names (``.``) and backup copies (``~``) are left out, and links to
    folders are not followed, so a link loop cannot make the walk go round.
    """
    found: list[Path] = []
    with os.scandir(folder) as entries:
        for entry in sorted(entries, key=lambda e: e.name):
            if entry.name.startswith(".") or entry.name.endswith("~"):
                continue
            if entry.is_dir(follow_symlinks=False):
                if recursive:
                    found += _walk(Path(entry.path), recursive)
            elif entry.is_symlink() or entry.is_file(follow_symlinks=False):
                found.append(Path(entry.path))
    return found


def index(
    directory: str | os.PathLike[str],
    *,
    recursive: bool = False,
    cache_dir: str | os.PathLike[str] | None = None,
) -> DirectoryIndex:
    """
    Index the ARL files in a directory by their headers.

    Each file in *directory* is opened once, and what its index records say
    is kept: its source, grid, first and last valid time, and time step. The
    table is cached, so a later call reads again only the files that are new
    or changed (by size or modification time) and drops the files that are
    gone. Use :meth:`DirectoryIndex.covering` to find the files for a time
    window.

    Links to files are followed, and each real file is one row, shown by its
    real path (:meth:`pathlib.Path.resolve`) however many links reach it. A
    link that cannot be followed (its target is gone, or links loop) is a row
    with the reason in ``error``. Links to folders are not followed. Hidden
    files and folders (names starting with ``.``) and backup copies (names
    ending in ``~``) are skipped. A file that is not ARL, or is damaged, is
    kept with the reason in ``error`` and is not used to cover a window.

    Files are opened 16 at a time: on a network filesystem each one waits on
    a few small reads. The cache is saved every 500 files, so an interrupted
    first index of a large archive keeps what it read.

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
        ``.arlmet-index.csv`` (``.arlmet-index-recursive.csv`` with
        *recursive*) in *directory* when *directory* is writable, and
        otherwise a file named after the directory in ``$ARLMET_CACHE_DIR``,
        or in the platform's user cache folder (``arlmet/index`` in
        ``$XDG_CACHE_HOME`` on Linux) when that variable is not set.

    Returns
    -------
    DirectoryIndex

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
    DirectoryIndex.covering : The files for a time window.
    File.check : Damage that opening a file does not catch.

    Examples
    --------
    >>> idx = arlmet.index("hrrr")  # doctest: +SKIP
    >>> idx.covering("2021-01-14 06:00", "2021-01-15 06:00")  # doctest: +SKIP
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
    cached = _read_cache(cache)

    # Rows are keyed by the file's real path, so a file reached by several
    # paths is read and listed once.
    rows: dict[str, dict[str, Any]] = {}
    to_read: dict[str, os.stat_result] = {}
    for found in _walk(directory, recursive):
        try:
            real = found.resolve(strict=True)
            stat = real.stat()
        except (OSError, RuntimeError) as exc:  # RuntimeError: a loop, before 3.13
            rows[str(found)] = {**_EMPTY, "error": f"Cannot follow the link: {exc}"}
            continue
        key = str(real)
        if not S_ISREG(stat.st_mode) or key in rows or key in to_read:
            continue
        known = cached.get(key)
        if (
            known is not None
            and known["size"] == stat.st_size
            and known["mtime_ns"] == stat.st_mtime_ns
        ):
            rows[key] = known
        else:
            to_read[key] = stat

    if to_read:
        logger.info(
            "Reading %d of %d files in %s",
            len(to_read),
            len(to_read) + len(rows),
            directory,
        )
        # Opening a file waits on a few small reads, which on a network
        # filesystem take far longer than parsing them, so several are read
        # at once. The cache is saved as rows come in, so an interrupted
        # first index of a large archive keeps what it read.
        with ThreadPoolExecutor(_READERS) as pool:
            read = pool.map(_read_file, [Path(key) for key in to_read])
            for n, ((key, stat), row) in enumerate(
                zip(to_read.items(), read, strict=True), start=1
            ):
                rows[key] = {**row, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
                if n % _SAVE_EVERY == 0:
                    logger.info("Read %d of %d files", n, len(to_read))
                    _write_cache(cache, rows)
    if to_read or rows.keys() != cached.keys():
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
    return DirectoryIndex(directory, table.reset_index(drop=True), cache=cache)


class DirectoryIndex:
    """
    The ARL files in a directory, by the times their headers say they hold.

    Made by :func:`arlmet.index`.

    Parameters
    ----------
    directory : pathlib.Path
        The indexed folder.
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
        # Each valid time a usable file holds, and the files that hold it.
        holders: dict[pd.Timestamp, list[Path]] = {}
        usable = table[table["error"] == ""]
        for path, start, end, step in zip(
            usable["path"], usable["start"], usable["end"], usable["step"], strict=True
        ):
            first = ensure_timestamp(start)
            times = (
                [first]
                if pd.isna(step)
                else list(pd.date_range(first, ensure_timestamp(end), freq=step))
            )
            for time in times:
                holders.setdefault(time, []).append(Path(path))
        self._holders = holders
        self._times = pd.DatetimeIndex(sorted(holders))

    #: The indexed folder.
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
        DirectoryIndex.missing : The valid times no file holds.
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
            The window, earlier time first.

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
        >>> idx.missing("2021-01-14 06:00", "2021-01-14 08:00")  # doctest: +SKIP
        [Timestamp('2021-01-14 07:00:00')]
        """
        return [t for t in self._needed(start, end) if t not in self._holders]

    def covering(self, start: Any, end: Any) -> list[Path]:
        """
        Return the files that cover a window, in time order.

        The files hold every valid time the window needs (:meth:`missing`):
        one every :attr:`step` from the one at or before *start* to the one at
        or after *end*. Files whose ``error`` is set are not used.

        Parameters
        ----------
        start, end : datetime-like
            The window, earlier time first.

        Returns
        -------
        list of pathlib.Path

        Raises
        ------
        ValueError
            If a valid time the window needs is in no file (the message lists
            them), if two files hold one of them (the message names both), if
            *start* is after *end*, or if no file is usable.

        Examples
        --------
        >>> idx.covering("2021-01-14 06:00", "2021-01-14 08:00")  # doctest: +SKIP
        [PosixPath('hrrr/20210114_06-11_hrrr')]
        """
        needed = self._needed(start, end)
        missing = [t for t in needed if t not in self._holders]
        if missing:
            listed = ", ".join(_at(t) for t in missing[:10])
            if len(missing) > 10:
                listed += f" and {len(missing) - 10} more"
            raise ValueError(
                f"No file in {self.directory} holds {listed}, which "
                f"{_at(needed[0])} to {_at(needed[-1])} needs."
            )
        files: list[Path] = []
        for time in needed:
            holders = self._holders[time]
            if len(holders) > 1:
                raise ValueError(
                    f"{_at(time)} is in two files, {holders[0]} and "
                    f"{holders[1]}. Move one of them out of {self.directory}."
                )
            if holders[0] not in files:
                files.append(holders[0])
        return files

    def __len__(self) -> int:
        """Return the number of files in the index."""
        return len(self.table)

    def __repr__(self) -> str:
        """Return the directory, the number of files and the table, with file names for paths."""
        n_bad = int((self.table["error"] != "").sum())
        names = self.table.assign(path=[path.name for path in self.table["path"]])
        return (
            f"DirectoryIndex({str(self.directory)!r}: {len(self)} files, "
            f"{n_bad} not used)\n{names!r}"
        )
