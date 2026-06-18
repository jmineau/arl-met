"""
Concatenate ARL meteorology files into a single ARL file.

ARL files are flat streams of fixed-size records, so joining several files is a
byte-level append — the same operation as ``cat a.arl b.arl > out.arl``. This is
the standard way to combine short (e.g. 6-hourly) met files into longer (e.g.
daily) files: HYSPLIT limits a simulation to at most 12 meteorological input
files when a single grid is specified, so longer per-file coverage is the only
way to span a long run. See the HYSPLIT user guide, "Compilation Limits":
https://www.ready.noaa.gov/hysplitusersguide/S441.htm

The original idea and the HRRR use case come from Derek Mallia's
``concat_hrrr_daily.py`` script.
"""

from __future__ import annotations

import os
import shutil
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from arlmet.file import File
from arlmet.index import IndexRecord

__all__ = ["concat", "concat_by_time"]


def concat(
    sources: Iterable[str | os.PathLike[str]],
    destination: str | os.PathLike[str],
    *,
    sort: bool = True,
) -> File:
    """
    Concatenate multiple ARL files into a single ARL file.

    Each input is appended to the output byte-for-byte, preserving every record
    (including diff records and checksums) exactly. The inputs are first scanned
    to ensure they share one grid and vertical axis and do not repeat valid
    times, since a concatenated ARL file must be a single coherent record stream.

    Parameters
    ----------
    sources : iterable of path-like
        Input ARL files to join. Must contain at least one path. A bare string
        or path is rejected — wrap a single file in a list.
    destination : path-like
        Output ARL file path. Overwrites any existing file. Must not be one of
        ``sources``.
    sort : bool, default True
        Order the inputs by their earliest valid time before joining, so the
        output is chronological regardless of input order. When False, inputs
        are joined in the order given (like ``cat``).

    Returns
    -------
    File
        The newly written file, opened in read mode. Close it when done (or use
        it as a context manager). Callers that only need the file on disk may
        ignore the return value.

    Raises
    ------
    ValueError
        If ``sources`` is empty, if ``destination`` is also a source, if any
        source is empty, if the inputs disagree on grid or vertical axis, or if
        the same valid time appears in more than one input.

    Examples
    --------
    Join three 6-hourly HRRR files into one daily file:

    >>> import arlmet
    >>> arlmet.concat(
    ...     ["20240101_00_hrrr", "20240101_06_hrrr", "20240101_12_hrrr"],
    ...     "20240101_hrrr",
    ... )

    Combine every 6-hourly file for one day discovered by glob (``sort=True``
    orders them by valid time, so glob order does not matter):

    >>> import glob
    >>> arlmet.concat(glob.glob("20240101_*_hrrr"), "20240101_hrrr")
    """
    # A bare str/PathLike is iterable (over characters / not at all), which would
    # silently do the wrong thing — reject it explicitly.
    if isinstance(sources, (str, bytes, os.PathLike)):
        raise TypeError(
            "sources must be an iterable of paths, not a single path. "
            "Wrap a single file in a list: concat([path], destination)."
        )

    source_paths = [Path(p) for p in sources]
    if not source_paths:
        raise ValueError("concat requires at least one source file.")

    destination = Path(destination)
    destination_resolved = destination.resolve()
    if any(p.resolve() == destination_resolved for p in source_paths):
        raise ValueError(
            f"destination {destination} is also one of the sources; "
            "concatenating a file onto itself is not allowed."
        )

    ordered_paths = _scan_sources(source_paths, sort=sort)

    with open(destination, "wb") as out:
        for path in ordered_paths:
            with open(path, "rb") as src:
                shutil.copyfileobj(src, out)

    return File(destination)


def _scan_sources(source_paths: list[Path], *, sort: bool) -> list[Path]:
    """
    Read each source's index records to validate compatibility and order by time.

    Returns the paths in write order. Raises if any source is empty, the grids
    or vertical axes disagree, or a valid time is shared across sources.
    """
    scanned = []
    for path in source_paths:
        with File(path) as src:
            times = src.times
            if not times:
                # An empty file never set a grid/axis, so check before reading them.
                raise ValueError(f"Source file {path} contains no records.")
            grid = src.grid
            axis = src.vertical_axis

        scanned.append((path, times, grid, axis))

    # source_paths is non-empty (checked by concat), so scanned[0] exists.
    reference_path, _, reference_grid, reference_axis = scanned[0]
    for path, _times, grid, axis in scanned[1:]:
        if grid != reference_grid:
            raise ValueError(
                f"Grid mismatch: {path} has grid {grid.nx}x{grid.ny}, "
                f"incompatible with {reference_path} "
                f"({reference_grid.nx}x{reference_grid.ny}). "
                "Concatenated ARL files must share a single grid."
            )
        if axis != reference_axis:
            raise ValueError(
                f"Vertical axis mismatch: {path} (flag {axis.flag}, "
                f"{len(axis.levels)} levels) is incompatible with "
                f"{reference_path} (flag {reference_axis.flag}, "
                f"{len(reference_axis.levels)} levels). Concatenated ARL "
                "files must share a single vertical axis."
            )

    if sort:
        # times is sorted by File.times, so times[0] is each file's earliest.
        scanned.sort(key=lambda item: item[1][0])

    _reject_duplicate_times([(path, times) for path, times, _, _ in scanned])

    return [path for path, _, _, _ in scanned]


def _reject_duplicate_times(scanned: list[tuple[Path, list[pd.Timestamp]]]) -> None:
    """Raise if any valid time appears in more than one source."""
    owner: dict[pd.Timestamp, Path] = {}
    for path, times in scanned:
        for time in times:
            if time in owner:
                raise ValueError(
                    f"Valid time {time} appears in both {owner[time]} and "
                    f"{path}. Concatenated ARL files must not repeat valid times: "
                    "arlmet cannot read a file with duplicate times and HYSPLIT "
                    "behavior on repeated times is undefined."
                )
            owner[time] = path


def concat_by_time(
    directory: str | os.PathLike[str],
    output_directory: str | os.PathLike[str],
    freq: str = "1D",
    *,
    pattern: str = "*",
    time_range: tuple[str | pd.Timestamp, str | pd.Timestamp] | None = None,
    template: str = "{time:%Y%m%d}_arl",
    sort: bool = True,
) -> list[Path]:
    """
    Group every ARL file in a directory by valid time and concatenate each group.

    Each input is assigned to a time bin from its first valid time — read from
    the file's index record, not parsed from its name — floored to ``freq``. All
    files in a bin are concatenated into one output file. This is the batch form
    of :func:`concat`: e.g. turning a directory of 6-hourly HRRR files into one
    file per day.

    Parameters
    ----------
    directory : path-like
        Directory to scan for input ARL files (non-recursive).
    output_directory : path-like
        Directory to write the concatenated files into. Created if missing.
        Should differ from ``directory``.
    freq : str, default "1D"
        Fixed-frequency pandas offset alias giving the size of each output
        chunk: ``"1D"`` = one file per day, ``"6h"`` = one per six hours, etc.
        Each input is binned by its first valid time floored to this frequency,
        so ``freq`` should be at least as long as any single input file's span.
    pattern : str, default "*"
        Glob (relative to ``directory``) selecting input files. Scope it to ARL
        files; every match must be a readable ARL file.
    time_range : tuple of (start, end), optional
        Inclusive ``(start, end)`` filter on each file's first valid time. Files
        whose first time falls outside the range are skipped.
    template : str, default "{time:%Y%m%d}_arl"
        ``str.format`` template for output filenames, given the bin start time
        as ``time`` (a ``pandas.Timestamp``), e.g. ``"{time:%Y%m%d}_hrrr"``. It
        must encode enough resolution to keep bins distinct at ``freq``.
    sort : bool, default True
        Passed through to :func:`concat` for each group.

    Returns
    -------
    list[pathlib.Path]
        The written output paths, one per non-empty time bin, in time order.

    Raises
    ------
    ValueError
        If ``pattern`` matches no files, or a matched file cannot be read as
        ARL. :func:`concat`'s grid/axis and duplicate-time checks also apply
        within each group.

    Examples
    --------
    Turn a directory of 6-hourly HRRR files into one file per day:

    >>> import arlmet
    >>> arlmet.concat_by_time(
    ...     "hrrr/",
    ...     "daily/",
    ...     freq="1D",
    ...     pattern="*_hrrr",
    ...     template="{time:%Y%m%d}_hrrr",
    ... )
    """
    directory = Path(directory)
    output_directory = Path(output_directory)

    candidates = sorted(p for p in directory.glob(pattern) if p.is_file())
    if not candidates:
        raise ValueError(f"No files matched pattern {pattern!r} in {directory}.")

    time_filter: tuple[pd.Timestamp, pd.Timestamp] | None = None
    if time_range is not None:
        time_filter = (pd.Timestamp(time_range[0]), pd.Timestamp(time_range[1]))

    groups: dict[pd.Timestamp, list[Path]] = defaultdict(list)
    for path in candidates:
        first_time = _peek_first_time(path)
        if time_filter is not None and not (
            time_filter[0] <= first_time <= time_filter[1]
        ):
            continue
        groups[first_time.floor(freq)].append(path)

    output_directory.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    for bin_start in sorted(groups):
        out_path = output_directory / template.format(time=bin_start)
        # concat returns an open File; we only need it written, so close it.
        with concat(groups[bin_start], out_path, sort=sort):
            pass
        outputs.append(out_path)
    return outputs


def _peek_first_time(path: Path) -> pd.Timestamp:
    """
    Read only the first index record to get a file's earliest valid time.

    Much cheaper than ``File(path).times`` on large multi-time files: it reads
    one index record instead of seeking to every index record in the file.
    """
    with open(path, "rb") as handle:
        try:
            return IndexRecord.from_position(handle, 0).time
        except Exception as exc:
            raise ValueError(
                f"Could not read an ARL index record from {path}: {exc}. "
                "Scope `pattern` so it only matches ARL files."
            ) from exc
