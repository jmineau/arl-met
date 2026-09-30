"""Subset extraction helpers for ARL meteorology files."""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

from arlmet._io import atomic_output, reject_same_file
from arlmet.exceptions import ARLFormatError
from arlmet.file import File
from arlmet.grid import Grid, GridWindow
from arlmet.header import Header, record_length_from_grid, split_grid_component
from arlmet.index import IndexRecord, LvlInfo, VarInfo, _derive_index_forecast
from arlmet.packing import calculate_checksum
from arlmet.record import DataRecord
from arlmet.vertical import VerticalAxis

if TYPE_CHECKING:
    from arlmet.recordset import RecordSet


def normalize_levels(
    vertical_axis: VerticalAxis, levels: Iterable[int] | None
) -> tuple[int, ...]:
    """
    Normalize a level selection to sorted unique ARL level indices.
    """
    if levels is None:
        return tuple(range(len(vertical_axis.levels)))

    normalized = tuple(sorted({int(level) for level in levels}))
    if not normalized:
        raise ValueError("levels must include at least one level index.")

    max_index = len(vertical_axis.levels) - 1
    if normalized[0] < 0 or normalized[-1] > max_index:
        raise ValueError(f"levels must be between 0 and {max_index}, got {normalized}.")
    return normalized


def resolve_window(
    file: File, bbox: tuple[float, float, float, float] | None
) -> GridWindow:
    """
    Resolve a bbox selection to a grid window.
    """
    if bbox is None:
        return file.grid.full_window()
    return file.grid.window_from_bbox(bbox)


def select_records(
    records: Sequence[DataRecord],
    *,
    levels: set[int] | None = None,
    variables: set[str] | None = None,
) -> list[DataRecord]:
    """
    Filter records by ARL level index and variable name.
    """
    return [
        record
        for record in records
        if (levels is None or record.level in levels)
        and (variables is None or record.variable in variables)
    ]


def _source_index_entries(
    selected_records: Sequence[DataRecord], level_map: dict[int, int]
) -> list[tuple[int, str, VarInfo]]:
    """
    Index manifest entries ``(output level, name, info)`` for selected records.

    Each record is followed by its DIF record, if any, as in the output file.
    Checksums and reserved flags come from the source index; they size the
    output index record for validation.
    """
    entries = []
    for record in selected_records:
        level = level_map[record.level]
        for rec in (record,) if record.diff is None else (record, record.diff):
            entries.append(
                (
                    level,
                    rec.variable,
                    VarInfo(checksum=rec.checksum, reserved=(rec._reserved or "")[:1]),
                )
            )
    return entries


def _build_subset_index_record(
    recordset: RecordSet,
    *,
    subset_grid: Grid,
    subset_axis: VerticalAxis,
    selected_records: Sequence[DataRecord],
    entries: Iterable[tuple[int, str, VarInfo]],
) -> IndexRecord:
    """
    Build the output index record for one subsetted time step.

    ``entries`` lists ``(output level, variable name, VarInfo)`` in file order.
    """
    forecast = _derive_index_forecast(
        (record.forecast for record in selected_records),
        recordset.forecast,
    )

    level_records: dict[int, OrderedDict[str, VarInfo]] = {
        level: OrderedDict() for level in range(len(subset_axis.levels))
    }
    for level, name, info in entries:
        level_records[level][name] = info

    grid_x = split_grid_component(subset_grid.nx)[0]
    grid_y = split_grid_component(subset_grid.ny)[0]
    levels = [
        LvlInfo(
            level=level,
            height=float(height),
            variables=level_records[level],
        )
        for level, height in enumerate(subset_axis.levels)
    ]
    projection = subset_grid.projection
    time = recordset.time
    return IndexRecord(
        header=Header(
            year=time.year,
            month=time.month,
            day=time.day,
            hour=time.hour,
            forecast=forecast,
            level=0,
            grid=(grid_x, grid_y),
            variable="INDX",
            exponent=0,
            precision=0.0,
            initial_value=0.0,
        ),
        source=recordset.source,
        forecast=forecast,
        minutes=time.minute,
        pole_lat=projection.pole_lat,
        pole_lon=projection.pole_lon,
        tangent_lat=projection.tangent_lat,
        tangent_lon=projection.tangent_lon,
        grid_size=projection.grid_size,
        orientation=projection.orientation,
        cone_angle=projection.cone_angle,
        sync_x=projection.sync_x,
        sync_y=projection.sync_y,
        sync_lat=projection.sync_lat,
        sync_lon=projection.sync_lon,
        reserved=subset_axis.offset,
        nx=subset_grid.nx,
        ny=subset_grid.ny,
        nz=len(levels),
        vertical_flag=subset_axis.flag,
        levels=levels,
    )


def validate_subset_record_length(
    selected_recordsets: Sequence[tuple[RecordSet, Sequence[DataRecord]]],
    *,
    subset_grid: Grid,
    subset_axis: VerticalAxis,
    level_map: dict[int, int],
) -> None:
    """
    Fail early when a cropped ARL grid cannot fit its index record.
    """
    record_len = record_length_from_grid(grid=subset_grid)
    for recordset, selected_records in selected_recordsets:
        index = _build_subset_index_record(
            recordset,
            subset_grid=subset_grid,
            subset_axis=subset_axis,
            selected_records=selected_records,
            entries=_source_index_entries(selected_records, level_map),
        )
        index_len = len(index.tobytes())
        if index_len > record_len:
            min_cells = index_len - Header.N_BYTES
            raise ValueError(
                "Subset grid is too small to encode an ARL index record: "
                f"time {recordset.time} needs {index_len} bytes, but each record is "
                f"only {record_len} bytes for grid {subset_grid.nx}x{subset_grid.ny}. "
                f"The bbox must yield at least {min_cells} grid cells (nx*ny). "
                "Expand the bbox or reduce levels/variables."
            )


# Byte offsets of the level field in a record header (see Header.FIELDS).
_LEVEL_START, _LEVEL_STOP = Header.FIELDS["level"][:2]


def _copy_subset_records(
    src: File,
    out: BinaryIO,
    selected_recordsets: Sequence[tuple[RecordSet, Sequence[DataRecord]]],
    *,
    subset_grid: Grid,
    subset_axis: VerticalAxis,
    level_map: dict[int, int],
) -> None:
    """
    Write an uncropped subset by copying record bytes instead of re-packing.

    Without a horizontal crop a record's packed bytes do not depend on which
    other records are kept, so each selected record (and its DIF record) is
    copied verbatim with only the header's level field renumbered. Only the
    index records are rebuilt, with checksums recomputed from the copied
    bytes and blank reserved flags, as a re-pack writes them.

    The output decodes to exactly the source values. For records that
    re-packing reproduces byte for byte (those written by arlmet without a
    DIF record) it is identical to the re-pack path's output; otherwise the
    re-pack path re-quantizes the values and this path is the more faithful.

    Each time step's index record is written last, over a placeholder, once
    the checksums of its records are known, so memory stays at a few records.
    """
    record_length = src.record_length
    src_fh = src.handle
    placeholder = bytes(record_length)

    for recordset, selected_records in selected_recordsets:
        index_position = out.tell()
        out.write(placeholder)

        entries: list[tuple[int, str, VarInfo]] = []
        for record in selected_records:
            level = level_map[record.level]
            level_field = f"{level:2d}".encode("ascii")
            for rec in (record,) if record.diff is None else (record, record.diff):
                src_fh.seek(rec.position)
                raw = src_fh.read(record_length)
                if len(raw) != record_length:
                    raise ARLFormatError(
                        f"{src.path}: record {rec.variable!r} at byte "
                        f"{rec.position} is truncated."
                    )
                header = Header.from_bytes(raw[: Header.N_BYTES])
                if header.variable != rec.variable or header.level != rec.level:
                    raise ARLFormatError(
                        f"DataRecord header mismatch at position {rec.position}: "
                        f"expected variable '{rec.variable}' level {rec.level}, "
                        f"got variable '{header.variable}' level '{header.level}'"
                    )
                # Write the record with its level renumbered, without copying it.
                view = memoryview(raw)
                out.write(view[:_LEVEL_START])
                out.write(level_field)
                out.write(view[_LEVEL_STOP:])
                checksum = calculate_checksum(view[Header.N_BYTES :])
                entries.append((level, rec.variable, VarInfo(checksum, reserved="")))

        index = _build_subset_index_record(
            recordset,
            subset_grid=subset_grid,
            subset_axis=subset_axis,
            selected_records=selected_records,
            entries=entries,
        )
        out.seek(index_position)
        out.write(index.to_record_bytes(record_length))
        out.seek(0, os.SEEK_END)


def _repack_subset_records(
    src: File,
    dest: Path,
    selected_recordsets: Sequence[tuple[RecordSet, Sequence[DataRecord]]],
    *,
    window: GridWindow,
    subset_grid: Grid,
    subset_axis: VerticalAxis,
    level_map: dict[int, int],
) -> None:
    """
    Write a cropped subset by unpacking each record's window and re-packing it.
    """
    with File(
        dest,
        mode="w",
        source=src.source,
        grid=subset_grid,
        vertical_axis=subset_axis,
    ) as out:
        for src_recordset, selected_records in selected_recordsets:
            dst_recordset = out.create_recordset(
                src_recordset.time,
                forecast=src_recordset.forecast,
            )
            for record in selected_records:
                # record.read() returns the full-precision value
                # (parent + diff when a diff is attached); the diff branch
                # below relies on this so that create_datarecord(diff=...)
                # can recompute the diff against the newly packed parent.
                data = record.read(window=window)
                dst_recordset.create_datarecord(
                    variable=record.variable,
                    level=level_map[record.level],
                    forecast=record.forecast,
                    data=data,
                    diff=record.diff.variable if record.diff is not None else None,
                )
            # Write each time step as soon as it is filled so peak memory
            # is one time step, not the whole output.
            out.flush()


def _is_full_window(grid: Grid, window: GridWindow) -> bool:
    """Return True when ``window`` covers all of ``grid`` (no horizontal crop)."""
    return window == grid.full_window()


def extract_subset(
    path: str | os.PathLike[str],
    dest: str | os.PathLike[str],
    *,
    bbox: tuple[float, float, float, float] | None = None,
    levels: Iterable[int] | None = None,
    variables: Iterable[str] | None = None,
) -> Path:
    """
    Extract a spatial/vertical subset from an ARL file into a new ARL file.

    Parameters
    ----------
    path : path-like
        Input ARL file.
    dest : path-like
        Output ARL file. Overwrites any existing file. Must not be ``path``.
    bbox : tuple[float, float, float, float], optional
        Geographic bounding box ``(west, south, east, north)`` in degrees.
    levels : iterable of int, optional
        ARL level indices to keep. Output levels are compacted and renumbered
        from zero while preserving the selected level heights.
    variables : iterable of str, optional
        Variable names to keep. All variables are included by default.

    Returns
    -------
    pathlib.Path
        The output path, ``Path(dest)``. Open it with :class:`~arlmet.File`
        or :func:`~arlmet.open_dataset` to read the subset.

    Raises
    ------
    ValueError
        If ``dest`` is the same file as ``path``, or the cropped grid is too
        small to hold the ARL index record.

    Notes
    -----
    The subset is written to a temporary file next to ``dest`` and renamed
    into place once complete, so an interrupted run never leaves a truncated
    file under the final name.

    Without a horizontal crop (``bbox`` is None or covers the whole grid),
    the selected records are copied byte for byte, with only their level
    numbers and the index records rewritten, which is much faster than
    unpacking and re-packing them. A crop re-packs each record's window.

    Examples
    --------
    >>> import arlmet
    >>> out = arlmet.extract_subset(
    ...     "met.arl",
    ...     "subset.arl",
    ...     bbox=(-114.0, 39.0, -110.0, 42.0),
    ...     levels=[0, 1, 2],
    ... )
    >>> ds = arlmet.open_dataset(out)
    """
    reject_same_file(path, dest)
    variable_names = None if variables is None else set(variables)

    with File(path) as src:
        window = resolve_window(src, bbox)
        selected_levels = normalize_levels(src.vertical_axis, levels)
        selected_level_set = set(selected_levels)
        level_map = {
            old_level: new_level for new_level, old_level in enumerate(selected_levels)
        }

        subset_grid = src.grid.subset(window)
        subset_axis = VerticalAxis.from_flag(
            src.vertical_axis.flag,
            levels=src.vertical_axis.levels[list(selected_levels)].tolist(),
            offset=src.vertical_axis.offset,
        )
        selected_recordsets = []
        for time in src.times:
            src_recordset = src[time]
            selected_records = select_records(
                src_recordset.records,
                levels=selected_level_set,
                variables=variable_names,
            )
            if selected_records:
                selected_recordsets.append((src_recordset, selected_records))

        validate_subset_record_length(
            selected_recordsets,
            subset_grid=subset_grid,
            subset_axis=subset_axis,
            level_map=level_map,
        )

        # Write to a temporary file that replaces dest only once
        # complete, so an interrupted run never leaves a truncated output.
        with atomic_output(dest) as tmp_path:
            if _is_full_window(src.grid, window):
                with open(tmp_path, "wb") as out:
                    _copy_subset_records(
                        src,
                        out,
                        selected_recordsets,
                        subset_grid=subset_grid,
                        subset_axis=subset_axis,
                        level_map=level_map,
                    )
            else:
                _repack_subset_records(
                    src,
                    tmp_path,
                    selected_recordsets,
                    window=window,
                    subset_grid=subset_grid,
                    subset_axis=subset_axis,
                    level_map=level_map,
                )

    return Path(dest)
