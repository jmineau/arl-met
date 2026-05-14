"""Subset extraction helpers for ARL meteorology files."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from arlmet.file import File
from arlmet.grid import Grid, GridWindow
from arlmet.header import Header, record_length_from_grid, split_grid_component
from arlmet.index import IndexRecord, LvlInfo, VarInfo, _derive_index_forecast
from arlmet.packing import unpack
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


def _build_subset_index_record(
    recordset: RecordSet,
    *,
    subset_grid: Grid,
    subset_axis: VerticalAxis,
    selected_records: Sequence[DataRecord],
    level_map: dict[int, int],
) -> IndexRecord:
    """Build the destination index record for one subsetted time step."""
    forecast = _derive_index_forecast(
        (record.forecast for record in selected_records),
        recordset.forecast,
    )

    level_records: dict[int, OrderedDict[str, VarInfo]] = {
        level: OrderedDict() for level in range(len(subset_axis.levels))
    }
    for record in selected_records:
        level_records[level_map[record.level]][record.variable] = VarInfo(
            checksum=record.checksum,
            reserved=(record._reserved or "")[:1],
        )

    grid_x, nx = split_grid_component(subset_grid.nx)
    grid_y, ny = split_grid_component(subset_grid.ny)
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
        nx=nx,
        ny=ny,
        nz=len(levels),
        vertical_flag=subset_axis.flag,
        index_length=0,
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
            level_map=level_map,
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


def extract_subset(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    levels: Iterable[int] | None = None,
    variables: Iterable[str] | None = None,
) -> None:
    """
    Extract a spatial/vertical subset from an ARL file into a new ARL file.

    Parameters
    ----------
    source_path, destination_path : path-like
        Input and output ARL file paths.
    bbox : tuple[float, float, float, float], optional
        Geographic bounding box ``(west, south, east, north)`` in degrees.
    levels : iterable of int, optional
        ARL level indices to keep. Output levels are compacted and renumbered
        from zero while preserving the selected level heights.
    variables : iterable of str, optional
        Variable names to keep. All variables are included by default.

    Examples
    --------
    >>> import arlmet
    >>> arlmet.extract_subset(
    ...     "met.arl",
    ...     "subset.arl",
    ...     bbox=(-114.0, 39.0, -110.0, 42.0),
    ...     levels=[0, 1, 2],
    ... )
    """
    variable_names = None if variables is None else set(variables)

    with File(source_path) as source:
        window = resolve_window(source, bbox)
        selected_levels = normalize_levels(source.vertical_axis, levels)
        selected_level_set = set(selected_levels)
        level_map = {
            old_level: new_level for new_level, old_level in enumerate(selected_levels)
        }

        subset_grid = source.grid.subset(window)
        subset_axis = VerticalAxis(
            flag=source.vertical_axis.flag,
            levels=source.vertical_axis.levels[list(selected_levels)].tolist(),
            offset=source.vertical_axis.offset,
        )
        selected_recordsets = []
        for time in source.times:
            src_recordset = source[time]
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

        with File(
            destination_path,
            mode="w",
            source=source.source,
            grid=subset_grid,
            vertical_axis=subset_axis,
        ) as destination:
            for src_recordset, selected_records in selected_recordsets:
                dst_recordset = destination.create_recordset(
                    src_recordset.time,
                    forecast=src_recordset.forecast,
                )
                for record in selected_records:
                    if record.diff is None:
                        data = record.read(window=window)
                        dst_recordset.create_datarecord(
                            variable=record.variable,
                            level=level_map[record.level],
                            forecast=record.forecast,
                            data=data,
                        )
                        continue

                    parent_data = np.asarray(
                        unpack(
                            packed=record.bytes[Header.N_BYTES :],
                            nx=source.grid.nx,
                            ny=source.grid.ny,
                            precision=record.header.precision,
                            exponent=record.header.exponent,
                            initial_value=record.header.initial_value,
                            window=window,
                            driver=np,
                        ),
                        dtype=np.float32,
                    )
                    diff_data = np.asarray(
                        record.diff.read(window=window), dtype=np.float32
                    )
                    destination.register_diff_binding(
                        diff_name=record.diff.variable,
                        parent_name=record.variable,
                    )
                    dst_parent = dst_recordset.create_datarecord(
                        variable=record.variable,
                        level=level_map[record.level],
                        forecast=record.forecast,
                        data=parent_data,
                    )
                    dst_diff = dst_parent._create_diff(
                        position=-1,
                        variable=record.diff.variable,
                        forecast=record.diff.forecast,
                    )
                    dst_diff[:] = diff_data
