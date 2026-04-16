"""xarray-facing read, write, and conversion helpers for ARL meteorology files."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

from arlmet.core import File
from arlmet.grid import Grid, Projection
from arlmet.vertical import VerticalAxis

if TYPE_CHECKING:
    from arlmet.core import DataRecord, RecordCollection, VariableView

PROJECTION_ATTRS = (
    "pole_lat",
    "pole_lon",
    "tangent_lat",
    "tangent_lon",
    "grid_size",
    "orientation",
    "cone_angle",
    "sync_x",
    "sync_y",
    "sync_lat",
    "sync_lon",
)


def grid_to_attrs(grid: Grid) -> dict[str, Any]:
    attrs = {
        "arl_nx": grid.nx,
        "arl_ny": grid.ny,
    }
    projection = grid.projection
    attrs.update({f"arl_{name}": getattr(projection, name) for name in PROJECTION_ATTRS})
    return attrs


def grid_from_attrs(attrs: Mapping[str, Any]) -> Grid:
    try:
        projection = Projection(
            **{name: float(attrs[f"arl_{name}"]) for name in PROJECTION_ATTRS}
        )
        nx = int(attrs["arl_nx"])
        ny = int(attrs["arl_ny"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Dataset is missing ARL grid metadata. Expected a Grid in attrs['grid'] "
            "or serialized 'arl_*' projection attributes."
        ) from exc
    return Grid(projection=projection, nx=nx, ny=ny)


def vertical_axis_to_attrs(vertical_axis: VerticalAxis) -> dict[str, Any]:
    return {
        "arl_vertical_flag": vertical_axis.flag,
        "arl_vertical_levels": vertical_axis.heights.tolist(),
        "arl_vertical_offset": vertical_axis.offset,
    }


def vertical_axis_from_attrs(attrs: Mapping[str, Any]) -> VerticalAxis:
    try:
        flag = int(attrs["arl_vertical_flag"])
        levels = attrs["arl_vertical_levels"]
        offset = float(attrs.get("arl_vertical_offset", 0.0))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Dataset is missing ARL vertical metadata. Expected a VerticalAxis in "
            "attrs['vertical_axis'] or serialized 'arl_vertical_*' attributes."
        ) from exc

    try:
        return VerticalAxis(flag=flag, levels=levels, offset=offset)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Dataset is missing ARL vertical metadata. Expected a VerticalAxis in "
            "attrs['vertical_axis'] or serialized 'arl_vertical_*' attributes."
        ) from exc


def record_collection_attrs(source: "RecordCollection") -> dict[str, Any]:
    attrs = {
        "source": source.source,
        "arl_source": source.source,
        "grid": source.grid,
    }
    attrs.update(grid_to_attrs(source.grid))

    try:
        vertical_axis = source.vertical_axis
    except (AttributeError, ValueError):
        vertical_axis = None

    if vertical_axis is not None:
        attrs["vertical_axis"] = vertical_axis
        attrs.update(vertical_axis_to_attrs(vertical_axis))

    return attrs


def attach_record_collection_metadata(
    source: "RecordCollection", ds: xr.Dataset
) -> xr.Dataset:
    ds.attrs.update(record_collection_attrs(source))
    if not source.records:
        return ds

    forecast_by_time = source.forecast_by_time
    if "time" in ds.dims:
        time_index = pd.to_datetime(ds.coords["time"].values)
        return ds.assign_coords(
            forecast=(
                "time",
                [forecast_by_time[pd.Timestamp(time)] for time in time_index],
            )
        )

    only_forecast = next(iter(forecast_by_time.values()))
    return ds.assign_coords(forecast=only_forecast)


def datarecord_to_xarray(record: "DataRecord", squeeze: bool = True) -> xr.DataArray:
    """
    Convert a DataRecord into an xarray.DataArray.
    """
    da = xr.DataArray(
        data=record.data,
        dims=record.grid.dims,
        coords=record.grid.calculate_coords(),
        name=record.variable,
    )

    da = da.expand_dims(("time", "level"))

    z_coords = record.recordset.vertical_axis.calculate_coords()
    level = record.level
    height = z_coords["height"][level]

    da = da.assign_coords(
        time=[record.time],
        forecast=("time", [record.forecast]),
        level=[level],
        height=("level", [height]),
    )
    da.attrs.update(**record.recordset.attrs)
    return da.squeeze() if squeeze else da


def record_collection_to_xarray(
    source: "RecordCollection", drop_variables=None, squeeze: bool = True
) -> xr.Dataset | xr.DataArray:
    """
    Convert a RecordCollection into an xarray Dataset.
    """
    drop_variables = drop_variables or []

    ds = xr.combine_by_coords(
        [
            dr.to_xarray(squeeze=False).drop_vars("forecast")
            for dr in source.records
            if dr.variable not in drop_variables
        ],
        join="outer",
        compat="no_conflicts",
        coords="different",
    )

    ds = attach_record_collection_metadata(source, ds)
    return ds.squeeze() if squeeze else ds


def variable_view_to_xarray(view: "VariableView", squeeze: bool = True) -> xr.DataArray:
    """
    Convert a VariableView into an xarray DataArray.
    """
    da = xr.combine_by_coords(
        [
            dr.to_xarray(squeeze=False)
            for dr in view.source.records
            if dr.variable == view.name
        ],
        join="outer",
        compat="no_conflicts",
        coords="different",
    )
    if isinstance(da, xr.Dataset):
        da = da[view.name]
    return da.squeeze() if squeeze else da


def _expand_scalar_dim(ds: xr.Dataset, dim: str) -> xr.Dataset:
    if dim in ds.dims:
        return ds
    if dim not in ds.coords:
        raise ValueError(f"Dataset must include a '{dim}' coordinate.")

    coord = ds.coords[dim]
    if coord.ndim != 0:
        raise ValueError(
            f"Dataset coordinate '{dim}' must be a dimension or scalar coordinate."
        )

    return ds.expand_dims({dim: [coord.item()]})


def normalize_dataset_for_write(ds: xr.Dataset) -> xr.Dataset:
    if not isinstance(ds, xr.Dataset):
        raise TypeError("write_dataset() requires an xarray.Dataset.")

    normalized = ds
    for dim in ("time", "level"):
        normalized = _expand_scalar_dim(normalized, dim)
    return normalized


def extract_dataset_source(attrs: Mapping[str, Any]) -> str:
    source = attrs.get("source", attrs.get("arl_source"))
    if not isinstance(source, str) or not source:
        raise ValueError(
            "Dataset is missing a source identifier. Set attrs['source'] or attrs['arl_source']."
        )
    return source


def extract_dataset_grid(attrs: Mapping[str, Any]) -> Grid:
    grid = attrs.get("grid")
    if isinstance(grid, Grid):
        return grid
    return grid_from_attrs(attrs)


def extract_dataset_vertical_axis(attrs: Mapping[str, Any]) -> VerticalAxis:
    vertical_axis = attrs.get("vertical_axis")
    if isinstance(vertical_axis, VerticalAxis):
        return VerticalAxis(
            flag=vertical_axis.flag,
            levels=vertical_axis.heights.tolist(),
            offset=vertical_axis.offset,
        )
    return vertical_axis_from_attrs(attrs)


def extract_dataset_forecasts(ds: xr.Dataset) -> np.ndarray:
    if "forecast" not in ds.coords:
        raise ValueError(
            "Dataset must include a 'forecast' coordinate aligned to the time dimension."
        )

    forecast = ds.coords["forecast"]
    if forecast.ndim == 0:
        return np.asarray([int(forecast.item())], dtype=int)
    if forecast.dims != ("time",):
        raise ValueError("The 'forecast' coordinate must use only the 'time' dimension.")
    return np.asarray(forecast.values, dtype=int)


def open_dataset(filename_or_obj, drop_variables=None, squeeze=True):
    """
    Open an ARLMet file and convert it to an xarray Dataset.
    """
    met = File(filename_or_obj)
    ds = met.to_xarray(drop_variables=drop_variables, squeeze=squeeze)
    return ds if isinstance(ds, xr.Dataset) else ds.to_dataset()


def write_dataset(ds: xr.Dataset, filename_or_obj) -> None:
    """
    Write an xarray Dataset to ARL format.
    """
    ds = normalize_dataset_for_write(ds)
    source = extract_dataset_source(ds.attrs)
    grid = extract_dataset_grid(ds.attrs)
    vertical_axis = extract_dataset_vertical_axis(ds.attrs)
    forecasts = extract_dataset_forecasts(ds)

    h_dims = grid.dims
    for dim, size in zip(h_dims, (grid.ny, grid.nx), strict=True):
        if dim not in ds.dims:
            raise ValueError(f"Dataset is missing required horizontal dimension '{dim}'.")
        if ds.sizes[dim] != size:
            raise ValueError(
                f"Dataset dimension '{dim}' has size {ds.sizes[dim]}, expected {size}."
            )

    level_values = np.asarray(ds.coords["level"].values)
    if not np.allclose(level_values, level_values.astype(int)):
        raise ValueError("The 'level' coordinate must contain integer ARL level indices.")
    level_indices = level_values.astype(int)
    if len(np.unique(level_indices)) != len(level_indices):
        raise ValueError("The 'level' coordinate must not contain duplicate indices.")
    if np.any(level_indices < 0) or np.any(level_indices >= len(vertical_axis.heights)):
        raise ValueError("The 'level' coordinate references indices outside the vertical axis.")

    height_coord = ds.coords.get("height")
    if (
        height_coord is not None
        and height_coord.ndim == 1
        and height_coord.dims == ("level",)
    ):
        expected_heights = vertical_axis.heights[level_indices]
        if not np.allclose(np.asarray(height_coord.values, dtype=float), expected_heights):
            raise ValueError(
                "The 'height' coordinate does not match the configured vertical axis levels."
            )

    times = pd.to_datetime(ds.coords["time"].values)
    if forecasts.shape[0] != len(times):
        raise ValueError("The 'forecast' coordinate must have one value per time.")

    with File(
        filename_or_obj,
        mode="w",
        source=source,
        grid=grid,
        vertical_axis=vertical_axis,
    ) as arl:
        for time_index, time in enumerate(times):
            forecast = int(forecasts[time_index])
            pending_records: list[tuple[str, int, np.ndarray]] = []

            for name, da in ds.data_vars.items():
                if len(name) > 4:
                    raise ValueError(
                        f"Variable names must be 4 characters or fewer, got '{name}'."
                    )
                if name.startswith("DIF"):
                    raise NotImplementedError("Writing DIF* variables is not implemented.")

                transposed = da.transpose("time", "level", *h_dims, missing_dims="raise")
                for level_pos, level_idx in enumerate(level_indices):
                    data = np.asarray(
                        transposed.isel(time=time_index, level=level_pos).values,
                        dtype=np.float32,
                    )
                    if data.shape != (grid.ny, grid.nx):
                        raise ValueError(
                            f"Variable '{name}' slice has shape {data.shape}, expected {(grid.ny, grid.nx)}."
                        )

                    nan_mask = np.isnan(data)
                    if nan_mask.all():
                        continue
                    if nan_mask.any():
                        raise ValueError(
                            f"Variable '{name}' contains a partially-missing slice at time "
                            f"{pd.Timestamp(time)} level {level_idx}. Use all-NaN to omit a slice."
                        )
                    if not np.isfinite(data).all():
                        raise ValueError(
                            f"Variable '{name}' contains non-finite values that cannot be packed."
                        )

                    pending_records.append((name, int(level_idx), data))

            if not pending_records:
                continue

            recordset = arl.create_recordset(pd.Timestamp(time))
            for name, level_idx, data in pending_records:
                recordset.create_datarecord(
                    variable=name,
                    level=level_idx,
                    forecast=forecast,
                    data=data,
                )
