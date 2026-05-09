"""xarray-facing read, write, and conversion helpers for ARL meteorology files."""

from collections import OrderedDict, defaultdict
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr
from xarray.backends.common import BackendArray
from xarray.core import indexing

from arlmet.file import File
from arlmet.grid import Grid, Projection
from arlmet.subset import normalize_levels, resolve_window, select_records
from arlmet.vertical import VerticalAxis

if TYPE_CHECKING:
    from arlmet.record import DataRecord
    from arlmet.recordset import RecordCollection, VariableView

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


class ArlVariableArray(BackendArray):
    """
    Backend-style lazy array for a single ARL variable.
    """

    def __init__(
        self,
        *,
        records: dict[tuple[int, int], "DataRecord"],
        shape: tuple[int, int, int, int],
        window=None,
    ):
        self.records = records
        self.shape = shape
        self.dtype = np.dtype(np.float32)
        self.window = window

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.OUTER_1VECTOR,
            self._getitem,
        )

    def _getitem(self, key):
        """Materialize the requested outer-indexed slice from the backing records."""
        if len(key) != 4:
            raise IndexError(f"ARL variable arrays expect 4 indexers, got {len(key)}.")

        t_idx, t_scalar = _normalize_backend_indexer(key[0], self.shape[0])
        z_idx, z_scalar = _normalize_backend_indexer(key[1], self.shape[1])
        y_idx, y_scalar = _normalize_backend_indexer(key[2], self.shape[2])
        x_idx, x_scalar = _normalize_backend_indexer(key[3], self.shape[3])

        out = np.full(
            (len(t_idx), len(z_idx), len(y_idx), len(x_idx)),
            np.nan,
            dtype=np.float32,
        )
        for time_pos, time_idx in enumerate(t_idx):
            for level_pos, level_idx in enumerate(z_idx):
                record = self.records.get((int(time_idx), int(level_idx)))
                if record is None:
                    continue
                field = record.read(window=self.window)
                out[time_pos, level_pos] = field[np.ix_(y_idx, x_idx)]

        for axis, is_scalar in reversed(
            list(enumerate((t_scalar, z_scalar, y_scalar, x_scalar)))
        ):
            if is_scalar:
                out = np.take(out, 0, axis=axis)
        return out


def _normalize_backend_indexer(indexer, size: int) -> tuple[np.ndarray, bool]:
    """Normalize one backend indexer to explicit integer indices and scalar status."""
    base = np.arange(size, dtype=int)
    if isinstance(indexer, slice):
        return np.asarray(base[indexer], dtype=int), False
    if isinstance(indexer, np.ndarray):
        return np.asarray(base[indexer], dtype=int).reshape(-1), False
    return np.asarray([base[indexer]], dtype=int), True


def grid_to_attrs(grid: Grid) -> dict[str, Any]:
    """Serialize Grid metadata into JSON-safe dataset attributes."""
    attrs = {
        "arl_nx": grid.nx,
        "arl_ny": grid.ny,
    }
    projection = grid.projection
    attrs.update(
        {f"arl_{name}": getattr(projection, name) for name in PROJECTION_ATTRS}
    )
    return attrs


def grid_from_attrs(attrs: Mapping[str, Any]) -> Grid:
    """Reconstruct a Grid from serialized ARL dataset attributes."""
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
    """Serialize a VerticalAxis into JSON-safe dataset attributes."""
    return {
        "arl_vertical_flag": vertical_axis.flag,
        "arl_vertical_levels": vertical_axis.levels.tolist(),
        "arl_vertical_offset": vertical_axis.offset,
    }


def vertical_axis_from_attrs(attrs: Mapping[str, Any]) -> VerticalAxis:
    """Reconstruct a VerticalAxis from serialized dataset attributes."""
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
    """Collect source, grid, and vertical metadata for xarray attrs."""
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
    """Attach serialized ARL metadata and forecast coordinates to a dataset."""
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
    level_value = z_coords["level"][record.level]

    da = da.assign_coords(
        time=[record.time],
        forecast=("time", [record.forecast]),
        level=[level_value],
    )
    da.attrs.update(**record_collection_attrs(record.recordset))
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
            datarecord_to_xarray(dr, squeeze=False).drop_vars("forecast")
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
            datarecord_to_xarray(dr, squeeze=False)
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


def _assign_dataset_metadata(
    ds: xr.Dataset,
    *,
    source: str,
    grid: Grid,
    vertical_axis: VerticalAxis,
    forecast_by_time: dict[pd.Timestamp, int],
) -> xr.Dataset:
    """Attach ARL metadata and forecast coordinates to a constructed dataset."""
    ds.attrs.update(
        {
            "source": source,
            "arl_source": source,
            "grid": grid,
            "vertical_axis": vertical_axis,
        }
    )
    ds.attrs.update(grid_to_attrs(grid))
    ds.attrs.update(vertical_axis_to_attrs(vertical_axis))

    if forecast_by_time:
        if "time" in ds.dims:
            times = pd.to_datetime(ds.coords["time"].values)
            ds = ds.assign_coords(
                forecast=(
                    "time",
                    [forecast_by_time[pd.Timestamp(time)] for time in times],
                )
            )
        else:
            ds = ds.assign_coords(forecast=next(iter(forecast_by_time.values())))
    return ds


def _build_dataset_from_file(
    met: File,
    *,
    bbox: tuple[float, float, float, float] | None,
    levels: list[int] | tuple[int, ...] | None,
    drop_variables=None,
    squeeze: bool = True,
) -> xr.Dataset:
    """Build a lazy xarray.Dataset from a File selection."""
    drop_variables = set(drop_variables or [])
    window = resolve_window(met, bbox)
    read_window = None if bbox is None else window
    selected_grid = met.grid if bbox is None else met.grid.subset(window)
    requested_levels = (
        None if levels is None else normalize_levels(met.vertical_axis, levels)
    )
    requested_level_set = None if requested_levels is None else set(requested_levels)

    selected_recordsets: list[tuple[pd.Timestamp, int, list[DataRecord]]] = []
    present_levels: OrderedDict[int, None] = OrderedDict()

    for time in met.times:
        selected_records = [
            record
            for record in select_records(met[time].records, levels=requested_level_set)
            if record.variable not in drop_variables
        ]
        if not selected_records:
            continue

        recordset = met[time]
        if recordset.forecast is not None:
            forecast = recordset.forecast
        else:
            forecasts = {record.forecast for record in selected_records}
            if len(forecasts) != 1:
                raise ValueError(
                    f"Selection contains multiple forecast hours for time {time}."
                )
            forecast = forecasts.pop()

        for record in selected_records:
            present_levels.setdefault(record.level, None)
        selected_recordsets.append(
            (pd.Timestamp(time), forecast, selected_records)
        )

    selected_levels = (
        list(requested_levels) if requested_levels is not None else list(present_levels)
    )
    level_pos = {level: pos for pos, level in enumerate(selected_levels)}

    coords = selected_grid.calculate_coords()
    ds = xr.Dataset(coords=coords)
    level_values = met.vertical_axis.levels[list(selected_levels)]
    ds = ds.assign_coords(level=("level", level_values.tolist()))

    times: list[pd.Timestamp] = []
    forecast_by_time: OrderedDict[pd.Timestamp, int] = OrderedDict()
    records_by_variable: dict[str, dict[tuple[int, int], Any]] = defaultdict(dict)

    for time_pos, (time, forecast, selected_records) in enumerate(selected_recordsets):
        times.append(time)
        forecast_by_time[time] = forecast
        for record in selected_records:
            records_by_variable[record.variable][
                (time_pos, level_pos[record.level])
            ] = record

    if times:
        ds = ds.assign_coords(time=("time", times))

    shape = (len(times), len(selected_levels), selected_grid.ny, selected_grid.nx)
    dims = ("time", "level", *selected_grid.dims)
    for name in sorted(records_by_variable):
        backend = ArlVariableArray(
            records=records_by_variable[name],
            shape=shape,
            window=read_window,
        )
        ds[name] = xr.Variable(dims, indexing.LazilyIndexedArray(backend))

    ds = _assign_dataset_metadata(
        ds,
        source=met.source,
        grid=selected_grid,
        vertical_axis=met.vertical_axis,
        forecast_by_time=forecast_by_time,
    )
    return ds.squeeze() if squeeze else ds


def _expand_scalar_dim(ds: xr.Dataset, dim: str) -> xr.Dataset:
    """Promote a scalar coordinate to a length-1 dimension when needed."""
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
    """Normalize a dataset to the dimensional form required by write_dataset()."""
    if not isinstance(ds, xr.Dataset):
        raise TypeError("write_dataset() requires an xarray.Dataset.")

    normalized = ds
    for dim in ("time", "level"):
        normalized = _expand_scalar_dim(normalized, dim)
    return normalized


def extract_dataset_source(attrs: Mapping[str, Any]) -> str:
    """Extract the ARL source identifier from dataset attributes."""
    source = attrs.get("source", attrs.get("arl_source"))
    if not isinstance(source, str) or not source:
        raise ValueError(
            "Dataset is missing a source identifier. Set attrs['source'] or attrs['arl_source']."
        )
    return source


def extract_dataset_grid(attrs: Mapping[str, Any]) -> Grid:
    """Extract a Grid instance from dataset attributes."""
    grid = attrs.get("grid")
    if isinstance(grid, Grid):
        return grid
    return grid_from_attrs(attrs)


def extract_dataset_vertical_axis(attrs: Mapping[str, Any]) -> VerticalAxis:
    """Extract a VerticalAxis instance from dataset attributes."""
    vertical_axis = attrs.get("vertical_axis")
    if isinstance(vertical_axis, VerticalAxis):
        return VerticalAxis(
            flag=vertical_axis.flag,
            levels=vertical_axis.levels.tolist(),
            offset=vertical_axis.offset,
        )
    return vertical_axis_from_attrs(attrs)


def extract_dataset_forecasts(ds: xr.Dataset) -> np.ndarray:
    """Extract forecast hours from a dataset as a one-dimensional integer array."""
    if "forecast" not in ds.coords:
        raise ValueError(
            "Dataset must include a 'forecast' coordinate aligned to the time dimension."
        )

    forecast = ds.coords["forecast"]
    if forecast.ndim == 0:
        return np.asarray([int(forecast.item())], dtype=int)
    if forecast.dims != ("time",):
        raise ValueError(
            "The 'forecast' coordinate must use only the 'time' dimension."
        )
    return np.asarray(forecast.values, dtype=int)


def open_dataset(
    filename_or_obj,
    drop_variables=None,
    squeeze=True,
    bbox: tuple[float, float, float, float] | None = None,
    levels: list[int] | tuple[int, ...] | None = None,
):
    """
    Open an ARL meteorology file as an xarray.Dataset.

    Parameters
    ----------
    filename_or_obj : path-like
        Path to the ARL file.
    drop_variables : iterable of str, optional
        Variable names to omit from the resulting dataset.
    squeeze : bool, default True
        Remove length-1 dimensions from the returned dataset.
    bbox : tuple[float, float, float, float], optional
        Geographic bounding box ``(west, south, east, north)`` in degrees.
        When provided, records are cropped before unpacking.
    levels : list[int] or tuple[int, ...], optional
        ARL level indices to keep.

    Returns
    -------
    xarray.Dataset
        Lazy dataset containing the selected ARL variables and coordinates.

    Examples
    --------
    >>> import arlmet
    >>> ds = arlmet.open_dataset("met.arl")
    >>> subset = arlmet.open_dataset(
    ...     "met.arl",
    ...     bbox=(-114.0, 39.0, -110.0, 42.0),
    ...     levels=[0, 1, 2],
    ... )
    """
    met = File(filename_or_obj)
    return _build_dataset_from_file(
        met,
        bbox=bbox,
        levels=levels,
        drop_variables=drop_variables,
        squeeze=squeeze,
    )


def write_dataset(ds: xr.Dataset, filename_or_obj) -> None:
    """
    Write an xarray.Dataset to ARL format.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset produced by :func:`open_dataset` or another dataset carrying
        equivalent ``arl_*`` metadata.
    filename_or_obj : path-like
        Output path for the ARL file.

    Returns
    -------
    None
        This function writes the dataset to disk.

    Examples
    --------
    >>> import arlmet
    >>> ds = arlmet.open_dataset("met.arl", levels=[0, 1, 2])
    >>> arlmet.write_dataset(ds, "subset.arl")
    """
    ds = normalize_dataset_for_write(ds)
    source = extract_dataset_source(ds.attrs)
    grid = extract_dataset_grid(ds.attrs)
    vertical_axis = extract_dataset_vertical_axis(ds.attrs)
    forecasts = extract_dataset_forecasts(ds)

    h_dims = grid.dims
    for dim, size in zip(h_dims, (grid.ny, grid.nx), strict=True):
        if dim not in ds.dims:
            raise ValueError(
                f"Dataset is missing required horizontal dimension '{dim}'."
            )
        if ds.sizes[dim] != size:
            raise ValueError(
                f"Dataset dimension '{dim}' has size {ds.sizes[dim]}, expected {size}."
            )

    level_values = np.asarray(ds.coords["level"].values, dtype=float)
    all_levels = vertical_axis.levels
    level_indices = []
    for v in level_values:
        matches = np.where(np.isclose(all_levels, v))[0]
        if len(matches) == 0:
            raise ValueError(
                f"Level value {v!r} not found in the vertical axis height values."
            )
        level_indices.append(int(matches[0]))
    level_indices = np.asarray(level_indices, dtype=int)
    if len(np.unique(level_indices)) != len(level_indices):
        raise ValueError("The 'level' coordinate must not contain duplicate values.")

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
                    raise NotImplementedError(
                        "Writing DIF* variables is not implemented."
                    )

                transposed = da.transpose(
                    "time", "level", *h_dims, missing_dims="raise"
                )
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
