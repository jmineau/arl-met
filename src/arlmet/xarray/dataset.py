from __future__ import annotations

import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr
from xarray.core import indexing

from arlmet._time import ensure_timestamp
from arlmet.subset import normalize_levels, resolve_window, select_records
from arlmet.vertical import VerticalAxis

from ._backend import ArlVariableArray
from ._coords import (
    _extract_dataset_forecast_hours,
    _extract_dataset_vertical_axis,
    add_cf_spatial_attrs,
    grid_from_coord,
    grid_to_coord,
    physical_coord_for_vaxis,
)

if TYPE_CHECKING:
    from arlmet.file import File
    from arlmet.record import DataRecord


def _build_dataset_from_file(
    met: File,
    *,
    drop_variables=None,
    bbox: tuple[float, float, float, float] | None = None,
    levels: list[int] | tuple[int, ...] | None = None,
) -> xr.Dataset:
    drop_variables = set(drop_variables or [])
    window = resolve_window(met, bbox)
    read_window = None if bbox is None else window
    selected_grid = met.grid if bbox is None else met.grid.subset(window)
    requested_levels = (
        None if levels is None else normalize_levels(met.vertical_axis, levels)
    )
    requested_level_set = None if requested_levels is None else set(requested_levels)

    selected_recordsets: list[tuple[pd.Timestamp, int | None, list[DataRecord]]] = []
    present_levels: dict[int, None] = {}

    for time in met.times:
        recordset = met[time]
        selected_records = [
            record
            for record in select_records(recordset.records, levels=requested_level_set)
            if record.variable not in drop_variables
        ]
        if not selected_records:
            continue

        for record in selected_records:
            present_levels.setdefault(record.level, None)
        selected_recordsets.append(
            (ensure_timestamp(time), recordset.forecast, selected_records)
        )

    selected_levels = (
        list(requested_levels) if requested_levels is not None else list(present_levels)
    )
    level_pos = {level: pos for pos, level in enumerate(selected_levels)}

    ds = xr.Dataset(coords=selected_grid.calculate_coords())
    ds = add_cf_spatial_attrs(ds)
    # Integer 1-N level coord; surface (index 0) is handled separately below
    ds = ds.assign_coords(
        level=("level", list(selected_levels), {"standard_name": "model_level_number"})
    )

    times: list[pd.Timestamp] = []
    index_forecasts: list[int] = []
    records_by_variable: dict[str, dict[tuple[int, int], Any]] = defaultdict(dict)

    for time_pos, (time, _index_forecast, selected_records) in enumerate(
        selected_recordsets
    ):
        times.append(time)
        index_forecasts.append(0 if _index_forecast is None else int(_index_forecast))
        for record in selected_records:
            records_by_variable[record.variable][
                (time_pos, level_pos[record.level])
            ] = record

    if times:
        ds = ds.assign_coords(
            time=xr.Variable(
                "time", times, attrs={"standard_name": "time", "axis": "T"}
            )
        )
        ds["forecast_hour"] = xr.Variable(
            "time",
            [int(value) for value in index_forecasts],
            attrs={
                "long_name": "ARL index record forecast hour",
                "description": (
                    "Forecast hour stored in the INDX record for each time step. "
                    "Individual variable record forecasts may differ and are not represented "
                    "in this Dataset API."
                ),
                "units": "hours",
            },
        )

    shape = (len(times), len(selected_levels), selected_grid.ny, selected_grid.nx)
    dims = ("time", "level", *selected_grid.dims)
    for name in sorted(records_by_variable):
        backend = ArlVariableArray(
            records=records_by_variable[name],
            shape=shape,
            window=read_window,
        )
        ds[name] = xr.Variable(dims, indexing.LazilyIndexedArray(backend))

    sfc_pos = level_pos.get(0)
    if sfc_pos is not None:
        sfc_only_vars = {
            name
            for name, recs in records_by_variable.items()
            if all(z == sfc_pos for (_, z) in recs)
        }
        for name in sfc_only_vars:
            ds[name] = ds[name].isel(level=sfc_pos).drop_vars("level")

        upper_level_indices = [i for i in range(len(selected_levels)) if i != sfc_pos]
        if upper_level_indices:
            ds = ds.isel(level=upper_level_indices)
        else:
            ds = ds.drop_dims("level")

    ds = ds.assign_coords(arl_grid=grid_to_coord(selected_grid))
    ds.attrs["source"] = met.source
    ds.attrs["vertical_flag"] = met.vertical_axis.flag
    if "level" in ds.coords:
        # Add physical non-dim coord for upper-air levels
        upper_level_ints = [int(v) for v in ds.coords["level"].values.tolist()]
        try:
            phys_name, phys_var = physical_coord_for_vaxis(
                met.vertical_axis, upper_level_ints
            )
            ds = ds.assign_coords({phys_name: phys_var})
        except NotImplementedError:
            warnings.warn(
                f"Physical coordinate for vertical_flag={met.vertical_axis.flag} is not implemented; "
                "the 'level' coordinate will contain raw ARL level indices only.",
                stacklevel=2,
            )
    return ds


def open_dataset(
    filename_or_obj,
    drop_variables=None,
    bbox: tuple[float, float, float, float] | None = None,
    levels: list[int] | tuple[int, ...] | None = None,
) -> xr.Dataset:
    """
    Open an ARL meteorology file as an xarray Dataset.

    Surface-only variables (e.g. ``SHGT``, ``T02M``, ``PRSS``) have dimensions
    ``(time, lat, lon)`` with no ``level`` dimension. Upper-air variables
    (e.g. ``UWND``, ``VWND``, ``TEMP``) have dimensions
    ``(time, level, lat, lon)``. There is no NaN padding.

    ARL metadata is accessible via the ``.arl`` accessor::

        ds = arlmet.open_dataset("met.arl")
        ds.arl.grid  # Grid object (survives isel/sel)
        ds.arl.vertical_axis  # VerticalAxis
        ds.isel(level=0)  # selects level 0 for upper vars; sfc vars unchanged

    Parameters
    ----------
    filename_or_obj : path-like
        Path to the ARL file.
    drop_variables : iterable of str, optional
        Variable names to omit from the resulting dataset.
    bbox : tuple[float, float, float, float], optional
        Geographic bounding box ``(west, south, east, north)`` in degrees.
    levels : list[int] or tuple[int, ...], optional
        ARL level indices to keep.

    Returns
    -------
    xarray.Dataset
    """
    from arlmet.file import File

    with File(filename_or_obj) as met:
        return met.to_dataset(
            drop_variables=drop_variables,
            bbox=bbox,
            levels=levels,
        )


def write_dataset(
    ds: xr.Dataset,
    filename_or_obj,
    *,
    vertical_axis: VerticalAxis | None = None,
) -> None:
    """
    Write a simple flat Dataset representation to an ARL file.

    This writer supports the common-case Dataset contract returned by
    ``open_dataset()``:

    - surface variables use dims ``(time, y, x)`` or ``(time, lat, lon)``
    - upper-air variables use dims ``(time, level, y, x)`` or ``(time, level, lat, lon)``
    - all upper-air variables share the same ``level`` coordinate

    Per-variable forecast heterogeneity is intentionally not represented in the
    flat Dataset API. ``forecast_hour(time)`` supplies only the ARL index-record
    forecast written for each time step.
    """
    from arlmet.file import File

    if not isinstance(ds, xr.Dataset):
        raise TypeError("write_dataset() requires an xarray.Dataset.")

    if "time" not in ds.coords:
        raise ValueError("Dataset must define a 'time' coordinate.")
    times = pd.to_datetime(ds.coords["time"].values)

    source = ds.attrs.get("source")
    if not isinstance(source, str) or not source:
        raise ValueError("Dataset is missing attrs['source'].")

    if "arl_grid" not in ds.coords:
        raise ValueError("Dataset is missing an 'arl_grid' coordinate.")
    grid = grid_from_coord(ds.coords["arl_grid"])
    h_dims = grid.dims

    resolved_vertical_axis = _extract_dataset_vertical_axis(ds, vertical_axis)
    forecast_hours = _extract_dataset_forecast_hours(ds, times)

    data_vars = [name for name in ds.data_vars if name != "forecast_hour"]
    if not data_vars:
        raise ValueError("Dataset contains no data variables to write.")

    diff_names: dict[str, str] = {}
    for name in data_vars:
        da = ds[str(name)]
        diff_name = da.attrs.get("diff")
        if diff_name is None:
            continue
        if not isinstance(diff_name, str) or not diff_name:
            raise ValueError(
                f"Variable '{name}' attrs['diff'] must be a non-empty string when provided."
            )
        if not diff_name.startswith("DIF"):
            raise ValueError(
                f"Variable '{name}' attrs['diff'] must start with 'DIF', got '{diff_name}'."
            )
        diff_names[str(name)] = diff_name

    with File(
        filename_or_obj,
        mode="w",
        source=source,
        grid=grid,
        vertical_axis=resolved_vertical_axis,
    ) as arl:
        for time_index, time in enumerate(times):
            recordset = arl.create_recordset(
                pd.Timestamp(time),
                forecast=forecast_hours[time_index],
            )

            for var_name in data_vars:
                var_name = str(var_name)
                if len(var_name) > 4:
                    raise ValueError(
                        f"Variable names must be 4 characters or fewer, got '{var_name}'."
                    )
                if var_name.startswith("DIF"):
                    raise ValueError(
                        "Dataset DIF generation is parent-led; set attrs['diff'] on the parent variable instead of writing a DIF* data variable."
                    )

                da = ds[var_name]
                for dim, size in zip(h_dims, (grid.ny, grid.nx), strict=True):
                    if dim not in da.dims:
                        raise ValueError(
                            f"Variable '{var_name}' is missing horizontal dimension '{dim}'."
                        )
                    if da.sizes[dim] != size:
                        raise ValueError(
                            f"Variable '{var_name}' dimension '{dim}' has size {da.sizes[dim]}, expected {size}."
                        )

                if "time" not in da.dims:
                    raise ValueError(
                        f"Variable '{var_name}' data must include the 'time' dimension."
                    )
                data_times = pd.to_datetime(da.coords["time"].values)
                if not data_times.equals(times):
                    raise ValueError(
                        f"Variable '{var_name}' time coordinate must align exactly with the dataset time coordinate."
                    )

                if "level" in da.dims:
                    if "level" not in ds.coords:
                        raise ValueError(
                            f"Variable '{var_name}' uses a 'level' dimension but the dataset has no shared level coordinate."
                        )
                    level_ints = np.atleast_1d(
                        np.asarray(ds.coords["level"].values, dtype=int)
                    ).tolist()
                    transposed = da.transpose(
                        "time", "level", *h_dims, missing_dims="raise"
                    )
                    write_iter = enumerate(level_ints)
                else:
                    transposed = da.transpose("time", *h_dims, missing_dims="raise")
                    write_iter = [(None, 0)]

                for slot, level_index in write_iter:
                    if slot is None:
                        data = np.asarray(
                            transposed.isel(time=time_index).values,
                            dtype=np.float32,
                        )
                    else:
                        data = np.asarray(
                            transposed.isel(time=time_index, level=slot).values,
                            dtype=np.float32,
                        )

                    if data.shape != (grid.ny, grid.nx):
                        raise ValueError(
                            f"Variable '{var_name}' slice has shape {data.shape}, expected {(grid.ny, grid.nx)}."
                        )
                    if np.isnan(data).any():
                        raise ValueError(
                            f"Variable '{var_name}' contains missing values at time {pd.Timestamp(time)} "
                            f"level {level_index}; write_dataset() requires complete slices."
                        )
                    if not np.isfinite(data).all():
                        raise ValueError(
                            f"Variable '{var_name}' contains non-finite values that cannot be packed."
                        )

                    recordset.create_datarecord(
                        variable=var_name,
                        level=int(level_index),
                        forecast=forecast_hours[time_index],
                        data=data,
                        diff=diff_names.get(var_name),
                    )
