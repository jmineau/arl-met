"""Build an ARL pressure-level Dataset from fields on a regular lat/lon grid."""

from __future__ import annotations

from collections.abc import Callable, Container, Hashable, Iterable, Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from arlmet._grid import Grid
from arlmet._version import __version__
from arlmet._xarray._coords import (
    add_cf_spatial_attrs,
    grid_to_coord,
    physical_coord_for_vaxis,
)
from arlmet.vertical import PressureAxis, VerticalAxis

from ._latlon import latlon_grid

__all__ = [
    "Field",
    "arl_dataset",
    "field_variable",
    "pressure_axis",
    "pressure_level_dataset",
    "scalar_coord",
    "stack_times",
    "standardize",
]

TIME_DIMS = ("valid_time", "time")
LEVEL_DIMS = ("pressure_level", "isobaricInhPa", "level")
LAT_DIMS = ("latitude", "lat")
LON_DIMS = ("longitude", "lon")


class Field(NamedTuple):
    """How one input variable becomes an ARL variable: ``ARL = input * scale``."""

    name: str
    scale: float
    units: str
    diff: str | None = None


def _pick(
    names: tuple[str, ...], available: Container[Hashable], what: str, label: str
) -> str:
    """Return the first of ``names`` in ``available``, or raise naming them."""
    for name in names:
        if name in available:
            return name
    raise ValueError(
        f"{label} has no {what} dimension (looked for {', '.join(names)})."
    )


def scalar_coord(ds: xr.Dataset, name: str) -> Any:
    """Return the value of a scalar coordinate of ``ds`` (any dtype), or None without one."""
    coord = ds.coords.get(name)
    if coord is None or coord.ndim != 0:
        return None
    return coord.values[()]


def stack_times(
    found: Mapping[tuple[pd.Timestamp, str], xr.DataArray],
    times: Sequence[pd.Timestamp],
) -> xr.Dataset:
    """
    Join each field's time steps along ``time``, NaN where a time step lacks it.

    ``found`` maps (valid time, name) to the field at that time; the result
    has one variable per name, on ``times``.
    """
    variables = {}
    for name in sorted({name for _, name in found}):
        template = next(da for (_, other), da in found.items() if other == name)
        slices = [
            found[(time, name)]
            if (time, name) in found
            else xr.full_like(template, np.nan, dtype=float)
            for time in times
        ]
        try:
            variables[name] = xr.concat(slices, dim="time", join="exact")
        except ValueError as exc:
            raise ValueError(
                f"The time steps hold '{name}' on different grids or pressure levels."
            ) from exc
    return xr.Dataset(variables).assign_coords(
        time=pd.DatetimeIndex(list(times)).values
    )


def standardize(
    ds: xr.Dataset,
    names: Iterable[str],
    *,
    upper: bool,
    label: str,
    level_dims: tuple[str, ...] = LEVEL_DIMS,
    level_kind: str = "pressure-level",
) -> xr.Dataset:
    """
    Keep the variables ``names`` with dims ``time``, (``level``,) ``lat``, ``lon``.

    The coordinates are the valid times (``valid_time`` when there is one,
    as cfgrib gives a forecast's), the levels (the first of ``level_dims``
    the Dataset has: pressures in hPa by default), and the cell centers
    from south to north; every other coordinate (``number``, ``expver``,
    ``step``, ...) is dropped. A single time step may be a scalar coordinate.
    """
    if not isinstance(ds, xr.Dataset):
        raise TypeError(f"{label} must be an xarray.Dataset, got {type(ds).__name__}.")
    wanted = list(names)
    known = [name for name in wanted if name in ds.data_vars]
    if not known:
        raise ValueError(
            f"{label} holds none of the variables arlmet converts "
            f"({', '.join(wanted)}); found "
            f"{', '.join(map(str, ds.data_vars)) or 'none'}."
        )
    ds = ds[known]

    if not any(name in ds.dims for name in TIME_DIMS):
        # One time step, as a scalar coordinate.
        scalar = _pick(TIME_DIMS, ds.coords, "time", label)
        ds = ds.expand_dims(scalar)
    time_dim = _pick(TIME_DIMS, ds.dims, "time", label)
    valid = ds.coords.get("valid_time")
    times = (
        valid if valid is not None and valid.dims == (time_dim,) else ds[time_dim]
    ).values

    dims = {time_dim: "time"}
    coords: dict[str, np.ndarray] = {"time": pd.to_datetime(times).values}
    if upper:
        level_dim = _pick(level_dims, ds.dims, level_kind, label)
        dims[level_dim] = "level"
        coords["level"] = np.asarray(ds[level_dim].values, dtype=float)
    lat_dim = _pick(LAT_DIMS, ds.dims, "latitude", label)
    lon_dim = _pick(LON_DIMS, ds.dims, "longitude", label)
    dims[lat_dim], dims[lon_dim] = "lat", "lon"
    coords["lat"] = np.asarray(ds[lat_dim].values, dtype=float)
    coords["lon"] = np.asarray(ds[lon_dim].values, dtype=float)

    for name in known:
        extra = set(map(str, ds[name].dims)) - set(dims)
        if extra:
            hint = (
                " (a 'step' dimension: select or stack valid times first)"
                if "step" in extra
                else ""
            )
            raise ValueError(
                f"{label} variable '{name}' has dimensions {sorted(extra)} besides "
                f"time, {level_kind.replace('-', ' ') + ', ' if upper else ''}"
                f"latitude and longitude{hint}."
            )
        if dims.keys() - set(map(str, ds[name].dims)):
            raise ValueError(
                f"{label} variable '{name}' has dims {ds[name].dims}; expected "
                f"{tuple(dims)}."
            )

    renames = {old: new for old, new in dims.items() if old != new}
    ds = ds.drop_vars(list(ds.coords)).rename_dims(renames).assign_coords(coords)
    if pd.Index(ds["time"].values).has_duplicates:
        raise ValueError(f"{label} has repeated valid times.")
    return ds.sortby("lat")


def field_variable(
    da: xr.DataArray, field: Field, dims: tuple[str, ...]
) -> xr.Variable:
    """Scale ``da`` to the ARL field, keeping a lazy array lazy where possible."""
    if field.scale != 1.0:
        da = da * field.scale
    variable = da.transpose(*dims).variable.copy(deep=False)
    attrs: dict[str, str] = {"units": field.units}
    if field.diff is not None:
        attrs["diff"] = field.diff
    variable.attrs = attrs
    return variable


def arl_dataset(
    grid: Grid,
    vertical_axis: VerticalAxis,
    times: npt.ArrayLike,
    *,
    source: str,
    forecast_hours: Sequence[int] | None = None,
) -> xr.Dataset:
    """
    Return an empty Dataset in :func:`arlmet.open_dataset`'s layout, to add fields to.

    It has the grid's coordinates and ``arl_grid``, ``time``, the index records'
    ``forecast_hour`` (0 unless given), ``level`` numbered 1 to N with the
    physical coordinate of ``vertical_axis`` (whose first level is the
    surface), and the ``source``, ``vertical_flag`` and ``arlmet_version`` attrs.
    """
    times = np.asarray(times)
    level_numbers = list(range(1, len(vertical_axis.levels)))
    hours = (
        np.zeros(len(times), dtype=int) if forecast_hours is None else forecast_hours
    )
    ds = add_cf_spatial_attrs(xr.Dataset(coords=grid.calculate_coords()))
    phys_name, phys_coord = physical_coord_for_vaxis(vertical_axis, level_numbers)
    ds = ds.assign_coords(
        time=xr.Variable("time", times, attrs={"standard_name": "time", "axis": "T"}),
        forecast_hour=xr.Variable(
            "time",
            np.asarray(hours, dtype=int),
            attrs={"long_name": "ARL index record forecast hour", "units": "hours"},
        ),
        level=xr.Variable(
            "level", level_numbers, attrs={"standard_name": "model_level_number"}
        ),
        arl_grid=grid_to_coord(grid),
    )
    ds = ds.assign_coords({phys_name: phys_coord})
    ds.attrs["source"] = source
    ds.attrs["vertical_flag"] = vertical_axis.flag
    ds.attrs["arlmet_version"] = __version__
    return ds


def pressure_axis(pressures: Sequence[float]) -> VerticalAxis:
    """Return the ARL pressure axis (flag 2) of ``pressures`` [hPa], the surface first as 0."""
    return PressureAxis(levels=[0.0, *pressures])


def pressure_level_dataset(
    upper: xr.Dataset,
    surface: xr.Dataset,
    upper_fields: Mapping[str, Field],
    surface_fields: Mapping[str, Field],
    *,
    source: str,
    forecast_hours: Sequence[int] | None = None,
    labels: tuple[str, str] = ("pressure_levels", "single_levels"),
    vertical_axis: Callable[[Sequence[float]], VerticalAxis] = pressure_axis,
) -> xr.Dataset:
    """
    Build the ARL Dataset from two :func:`standardize`'d Datasets.

    ``surface`` must be on ``upper``'s grid and hold each of its valid times;
    its other times are left out. Each variable of ``upper_fields`` and
    ``surface_fields`` the inputs hold is scaled and named as its
    :class:`Field` says. Levels are written from the largest ``level`` value
    up (the highest pressure, or the lowest model level), and their axis is
    ``vertical_axis`` of those values (by default pressure, flag 2).
    ``forecast_hours`` (one per valid time, else 0) is the index records'.
    """
    upper_label, surface_label = labels
    for axis in ("lat", "lon"):
        a, b = upper[axis].values, surface[axis].values
        if a.shape != b.shape or not np.allclose(
            (a - b + 180.0) % 360.0 - 180.0, 0.0, atol=1e-6
        ):
            raise ValueError(
                f"{upper_label} and {surface_label} are on different grids "
                f"({axis} differs)."
            )
    missing = pd.Index(upper["time"].values).difference(
        pd.Index(surface["time"].values)
    )
    if len(missing):
        raise ValueError(
            f"{surface_label} lacks {len(missing)} valid time(s) of {upper_label}, "
            f"the first {missing[0]}."
        )
    surface = surface.sel(time=upper["time"].values)
    upper = upper.sortby("level", ascending=False)

    grid = latlon_grid(upper["lat"].values, upper["lon"].values)
    axis = vertical_axis([float(value) for value in upper["level"].values])
    ds = arl_dataset(
        grid,
        axis,
        upper["time"].values,
        source=source,
        forecast_hours=forecast_hours,
    )

    upper = upper.drop_vars(["lat", "lon", "level", "time"])
    surface = surface.drop_vars(["lat", "lon", "time"])
    for name, field in upper_fields.items():
        if name in upper:
            ds[field.name] = field_variable(
                upper[name], field, ("time", "level", "lat", "lon")
            )
    for name, field in surface_fields.items():
        if name in surface:
            ds[field.name] = field_variable(
                surface[name], field, ("time", "lat", "lon")
            )
    return ds
