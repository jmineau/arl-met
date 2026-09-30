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

from arlmet.errors import ARLFormatError
from arlmet.file import File

__all__ = ["concat", "concat_by_time"]


def concat(
    paths: Iterable[str | os.PathLike[str]],
    dest: str | os.PathLike[str],
    *,
    sort: bool = True,
) -> Path:
    """
    Concatenate multiple ARL files into a single ARL file.

    Each input is appended to the output byte-for-byte, preserving every record
    (including diff records and checksums) exactly. The inputs are first scanned
    to ensure they share one grid and vertical axis and do not repeat valid
    times, since a concatenated ARL file must be a single coherent record stream.

    Parameters
    ----------
    paths : iterable of path-like
        Input ARL files to join. Must contain at least one path. A bare string
        or path is rejected — wrap a single file in a list.
    dest : path-like
        Output ARL file. Overwrites any existing file. Must not be one of
        ``paths``.
    sort : bool, default True
        Order the inputs by their earliest valid time before joining, so the
        output is chronological regardless of input order. When False, inputs
        are joined in the order given (like ``cat``).

    Returns
    -------
    pathlib.Path
        The output path, ``Path(dest)``.

    Raises
    ------
    TypeError
        If ``paths`` is a single path rather than an iterable of paths.
    ValueError
        If ``paths`` is empty, if ``dest`` is also an input, if any input is
        empty, if the inputs disagree on grid or vertical axis, or if the same
        valid time appears in more than one input.

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
    if isinstance(paths, (str, bytes, os.PathLike)):
        raise TypeError(
            "paths must be an iterable of paths, not a single path. "
            "Wrap a single file in a list: concat([path], dest)."
        )

    input_paths = [Path(p) for p in paths]
    if not input_paths:
        raise ValueError("concat requires at least one input file.")

    dest = Path(dest)
    dest_resolved = dest.resolve()
    if any(p.resolve() == dest_resolved for p in input_paths):
        raise ValueError(
            f"dest {dest} is also one of the input paths; "
            "concatenating a file onto itself is not allowed."
        )

    ordered_paths = _scan_inputs(input_paths, sort=sort)

    with open(dest, "wb") as out:
        for path in ordered_paths:
            with open(path, "rb") as src:
                shutil.copyfileobj(src, out)

    return dest


def _scan_inputs(paths: list[Path], *, sort: bool) -> list[Path]:
    """
    Read each input's index records to validate compatibility and order by time.

    Returns the paths in write order. Raises if any input is empty, the grids
    or vertical axes disagree, or a valid time is shared across inputs.
    """
    scanned = []
    for path in paths:
        with File(path) as src:
            times = src.times
            if not times:
                # An empty file never set a grid/axis, so check before reading them.
                raise ValueError(f"Input file {path} contains no records.")
            grid = src.grid
            axis = src.vertical_axis

        scanned.append((path, times, grid, axis))

    # paths is non-empty (checked by concat), so scanned[0] exists.
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
    """Raise if any valid time appears in more than one input."""
    owner: dict[pd.Timestamp, Path] = {}
    for path, times in scanned:
        for time in times:
            if time in owner:
                raise ValueError(
                    f"Valid time {time} appears in both {owner[time]} and "
                    f"{path}. Concatenated ARL files must not repeat valid times: "
                    "arlmet rejects a time step repeated with different content "
                    "and HYSPLIT behavior on repeated times is undefined."
                )
            owner[time] = path


def concat_by_time(
    directory: str | os.PathLike[str],
    dest_dir: str | os.PathLike[str],
    *,
    freq: str = "1D",
    pattern: str = "*",
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    template: str = "{time:%Y%m%d}_arl",
    sort: bool = True,
) -> list[Path]:
    """
    Group every ARL file in a directory by valid time and concatenate each group.

    Each input is assigned to a time bin from its valid times — read from the
    file's index records, not parsed from its name — floored to ``freq``. All
    files in a bin are concatenated into one output file. This is the batch form
    of :func:`concat`: e.g. turning a directory of 6-hourly HRRR files into one
    file per day. Files are never split, so every input must fall entirely
    within one bin.

    Parameters
    ----------
    directory : path-like
        Directory to scan for input ARL files (non-recursive).
    dest_dir : path-like
        Directory to write the concatenated files into. Created if missing.
        Should differ from ``directory``.
    freq : str, default "1D"
        Fixed-frequency pandas offset alias giving the size of each output
        chunk: ``"1D"`` = one file per day, ``"6h"`` = one per six hours, etc.
        Each input must fit entirely within one bin: a file whose first and
        last valid times floor to different bins raises ``ValueError``.
    pattern : str, default "*"
        Glob (relative to ``directory``) selecting input files. Scope it to ARL
        files; every match must be a readable ARL file.
    start, end : str or pandas.Timestamp, optional
        Inclusive bounds on each file's first valid time; files outside them
        are skipped. Either may be omitted to leave that side open.
    template : str, default "{time:%Y%m%d}_arl"
        ``str.format`` template for output filenames, given the bin start time
        as ``time`` (a ``pandas.Timestamp``), e.g. ``"{time:%Y%m%d}_hrrr"``. It
        must encode enough resolution to keep bins distinct at ``freq``; two
        bins that format to the same filename raise ``ValueError``.
    sort : bool, default True
        Passed through to :func:`concat` for each group.

    Returns
    -------
    list[pathlib.Path]
        The written output paths, one per non-empty time bin, in time order.

    Raises
    ------
    ValueError
        If ``pattern`` matches no files, a matched file cannot be read as ARL,
        a file's valid times straddle a ``freq`` bin boundary, or ``template``
        formats two bins to the same filename. :func:`concat`'s grid/axis and
        duplicate-time checks also apply within each group. All checks run
        before any output is written.

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
    dest_dir = Path(dest_dir)

    candidates = sorted(p for p in directory.glob(pattern) if p.is_file())
    if not candidates:
        raise ValueError(f"No files matched pattern {pattern!r} in {directory}.")

    start_time = None if start is None else pd.Timestamp(start)
    end_time = None if end is None else pd.Timestamp(end)

    groups: dict[pd.Timestamp, list[Path]] = defaultdict(list)
    for path in candidates:
        first_time, last_time = _read_time_span(path)
        if (start_time is not None and first_time < start_time) or (
            end_time is not None and first_time > end_time
        ):
            continue
        bin_start = first_time.floor(freq)
        if last_time.floor(freq) != bin_start:
            raise ValueError(
                f"{path} spans {first_time} to {last_time}, which straddles a "
                f"{freq!r} bin boundary. concat_by_time does not split files; "
                "use a freq at least as long as each input file's span, aligned "
                "so no file crosses a bin edge."
            )
        groups[bin_start].append(path)

    out_paths = {
        bin_start: dest_dir / template.format(time=bin_start)
        for bin_start in sorted(groups)
    }
    owner: dict[Path, pd.Timestamp] = {}
    for bin_start, out_path in out_paths.items():
        if out_path in owner:
            raise ValueError(
                f"template {template!r} gives the same filename {out_path.name!r} "
                f"for the bins starting {owner[out_path]} and {bin_start}. Add "
                f"enough time resolution to the template to keep {freq!r} bins "
                "distinct, e.g. '{time:%Y%m%d_%H}'."
            )
        owner[out_path] = bin_start

    dest_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    for bin_start, out_path in out_paths.items():
        outputs.append(concat(groups[bin_start], out_path, sort=sort))
    return outputs


def _read_time_span(path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return a file's first and last valid times, read from its index records."""
    try:
        with File(path) as src:
            times = src.times
    except (EOFError, ARLFormatError) as exc:
        raise ARLFormatError(
            f"Could not read an ARL index record from {path}: {exc}. "
            "Scope `pattern` so it only matches ARL files."
        ) from exc
    if not times:
        raise ValueError(f"Input file {path} contains no records.")
    return times[0], times[-1]
