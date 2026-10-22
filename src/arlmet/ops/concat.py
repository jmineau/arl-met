"""
Concatenate ARL meteorology files into a single ARL file.

ARL files are flat streams of fixed-size records, so joining several files is a
byte-level copy with no repacking: each time step (its index record and data
records) is copied unchanged, in valid-time order. HYSPLIT finds a time step by
its position in the file, so the order matters; for inputs already in order
this is the same as ``cat a.arl b.arl > out.arl``. This is the standard way to combine short (e.g. 6-hourly) met files into longer (e.g.
daily) files: HYSPLIT's ``CONTROL`` file takes at most 12 meteorological files
when its number of input data grids is one number (each file its own grid) and
at most 128 per grid when it is two numbers, ``1 N`` (one grid with ``N`` files),
so longer per-file coverage keeps a long run under either limit. See the HYSPLIT
user guide, "Compilation Limits":
https://www.ready.noaa.gov/hysplitusersguide/S441.htm

The original idea and the HRRR use case come from Derek Mallia's
``concat_hrrr_daily.py`` script.
"""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Iterable
from contextlib import ExitStack
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from arlmet._io import atomic_output
from arlmet.exceptions import ARLFormatError
from arlmet.file import File

__all__ = ["concat", "concat_by_time"]


def concat(
    paths: Iterable[str | os.PathLike[str]],
    dest: str | os.PathLike[str],
) -> Path:
    """
    Concatenate multiple ARL files into a single ARL file.

    Every time step of the inputs is copied to the output byte for byte, in
    valid-time order, preserving every record (including diff records and
    checksums) exactly. HYSPLIT finds a time step by its position in the file,
    so the output must hold its time steps in order: inputs whose hours
    interleave are merged hour by hour, and a time step an input repeats byte
    for byte (as some NOAA HRRR archive files do) is written once. The inputs
    are first scanned to ensure they share one grid and vertical axis and do
    not repeat valid times, since a concatenated ARL file must be a single
    coherent record stream.

    Parameters
    ----------
    paths : iterable of path-like
        Input ARL files to join. Must contain at least one path. A bare string
        or path is rejected — wrap a single file in a list.
    dest : path-like
        Output ARL file. It is written to a hidden ``.partial`` file beside
        it and renamed once complete, so it replaces an existing file only on
        success. Must not be one of ``paths``.

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

    Combine every 6-hourly file for one day discovered by glob (the output is
    in valid-time order, so glob order does not matter):

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

    steps = _scan_inputs(input_paths)

    # Write a hidden partial file and rename it only once every input is in, so
    # an interrupted run never leaves a short file under the final name.
    with (
        atomic_output(dest) as tmp_path,
        open(tmp_path, "wb") as out,
        ExitStack() as stack,
    ):
        sources = {
            path: stack.enter_context(open(path, "rb"))
            for path in dict.fromkeys(path for _, path, _, _ in steps)
        }
        for _, path, position, n_bytes in steps:
            _copy_step(sources[path], out, position, n_bytes)

    return dest


#: Bytes copied per read: a few HRRR records, so memory stays small.
_CHUNK = 16 * 1024 * 1024


def _copy_step(src: BinaryIO, out: BinaryIO, position: int, n_bytes: int) -> None:
    """Copy the ``n_bytes`` of one time step, starting at ``position``, to ``out``."""
    src.seek(position)
    remaining = n_bytes
    while remaining:
        block = src.read(min(remaining, _CHUNK))
        if not block:
            # File() checked the size when it scanned the input.
            raise ARLFormatError(f"{src.name} ended inside a time step.")
        out.write(block)
        remaining -= len(block)


def _scan_inputs(paths: list[Path]) -> list[tuple[pd.Timestamp, Path, int, int]]:
    """
    Read each input's index records to validate compatibility and order by time.

    Returns every time step to write as ``(time, path, position, n_bytes)``,
    in valid-time order, each once. Raises if any input is empty, the grids or
    vertical axes disagree, or a valid time is shared across inputs.
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
            # File raised if a repeated time step differs, so a repeat is a
            # byte-identical copy: write the first one only.
            steps: dict[pd.Timestamp, tuple[pd.Timestamp, Path, int, int]] = {}
            for time, position, n_bytes in src._steps:
                steps.setdefault(time, (time, path, position, n_bytes))

        scanned.append((path, times, grid, axis, steps))

    # paths is non-empty (checked by concat), so scanned[0] exists.
    reference_path, _, reference_grid, reference_axis, _ = scanned[0]
    for path, _times, grid, axis, _ in scanned[1:]:
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

    _reject_duplicate_times([(path, times) for path, times, _, _, _ in scanned])

    # Valid times are unique across the inputs now, so this is a total order.
    return sorted(
        (step for *_, steps in scanned for step in steps.values()),
        key=lambda step: step[0],
    )


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
        outputs.append(concat(groups[bin_start], out_path))
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
