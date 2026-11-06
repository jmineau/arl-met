"""Convert ERA5 pressure-level and single-level fields, as HYSPLIT's ``era52arl``."""

from __future__ import annotations

from collections.abc import Container, Hashable
from typing import NamedTuple

import numpy as np
import pandas as pd
import xarray as xr

from arlmet.vertical import PressureAxis
from arlmet.xarray._coords import (
    add_cf_spatial_attrs,
    grid_to_coord,
    physical_coord_for_vaxis,
)

from ._latlon import latlon_grid

__all__ = ["from_era5"]

#: ECMWF's gravity (m s-2): geopotential height is geopotential over it.
GRAVITY = 9.80665
#: Seconds in the hour each ERA5 hourly accumulation covers.
HOUR = 3600.0


class _Field(NamedTuple):
    """How one ERA5 variable becomes an ARL variable: ``ARL = ERA5 * scale``."""

    name: str
    scale: float
    units: str
    diff: str | None = None


#: ERA5 pressure-level variables (netCDF and cfgrib names) and their ARL fields.
PRESSURE_LEVEL_FIELDS: dict[str, _Field] = {
    "z": _Field("HGTS", 1.0 / GRAVITY, "m"),
    "t": _Field("TEMP", 1.0, "K"),
    "u": _Field("UWND", 1.0, "m s-1"),
    "v": _Field("VWND", 1.0, "m s-1"),
    "w": _Field("WWND", 0.01, "hPa s-1", "DIFW"),
    "r": _Field("RELH", 1.0, "%"),
    "q": _Field("SPHU", 1.0, "kg kg-1"),
}

#: ERA5 single-level variables and their ARL fields. Accumulations are over the
#: hour before the valid time; ECMWF's fluxes are positive downward, HYSPLIT's
#: heat fluxes positive upward.
SINGLE_LEVEL_FIELDS: dict[str, _Field] = {
    "sp": _Field("PRSS", 0.01, "hPa"),
    "z": _Field("SHGT", 1.0 / GRAVITY, "m"),
    "t2m": _Field("T02M", 1.0, "K"),
    "d2m": _Field("DP2M", 1.0, "K"),
    "u10": _Field("U10M", 1.0, "m s-1"),
    "v10": _Field("V10M", 1.0, "m s-1"),
    "blh": _Field("PBLH", 1.0, "m"),
    "cape": _Field("CAPE", 1.0, "J kg-1"),
    "tcc": _Field("TCLD", 100.0, "%"),
    "tp": _Field("TPP1", 1.0, "m", "DIFR"),
    "sshf": _Field("SHTF", -1.0 / HOUR, "W m-2"),
    "slhf": _Field("LHTF", -1.0 / HOUR, "W m-2"),
    "ssrd": _Field("DSWF", 1.0 / HOUR, "W m-2"),
    "zust": _Field("USTR", 1.0, "m s-1"),
}

_TIME_DIMS = ("valid_time", "time")
_LEVEL_DIMS = ("pressure_level", "isobaricInhPa", "level")
_LAT_DIMS = ("latitude", "lat")
_LON_DIMS = ("longitude", "lon")


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


def _standardize(
    ds: xr.Dataset, fields: dict[str, _Field], *, upper: bool, label: str
) -> xr.Dataset:
    """
    Keep the variables of ``fields`` with dims ``time``, (``level``,) ``lat``, ``lon``.

    The coordinates are the valid times, the pressures (hPa), and the cell
    centers; every other coordinate (``number``, ``expver``, ...) is dropped.
    """
    if not isinstance(ds, xr.Dataset):
        raise TypeError(f"{label} must be an xarray.Dataset, got {type(ds).__name__}.")
    known = [name for name in fields if name in ds.data_vars]
    if not known:
        raise ValueError(
            f"{label} holds none of the ERA5 variables arlmet converts "
            f"({', '.join(fields)}); found {', '.join(map(str, ds.data_vars)) or 'none'}."
        )
    ds = ds[known]

    if not any(name in ds.dims for name in _TIME_DIMS):
        # One time step, as a scalar coordinate.
        scalar = _pick(_TIME_DIMS, ds.coords, "time", label)
        ds = ds.expand_dims(scalar)
    time_dim = _pick(_TIME_DIMS, ds.dims, "time", label)
    valid = ds.coords.get("valid_time")
    times = (
        valid if valid is not None and valid.dims == (time_dim,) else ds[time_dim]
    ).values

    dims = {time_dim: "time"}
    coords: dict[str, np.ndarray] = {"time": pd.to_datetime(times).values}
    if upper:
        level_dim = _pick(_LEVEL_DIMS, ds.dims, "pressure-level", label)
        dims[level_dim] = "level"
        coords["level"] = np.asarray(ds[level_dim].values, dtype=float)
    lat_dim = _pick(_LAT_DIMS, ds.dims, "latitude", label)
    lon_dim = _pick(_LON_DIMS, ds.dims, "longitude", label)
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
                f"time, {'pressure level, ' if upper else ''}latitude and longitude{hint}."
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


def _field_variable(
    da: xr.DataArray, field: _Field, dims: tuple[str, ...]
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


def from_era5(pressure_levels: xr.Dataset, single_levels: xr.Dataset) -> xr.Dataset:
    """
    Convert ERA5 hourly fields to a Dataset that :func:`arlmet.write_dataset` writes.

    This is HYSPLIT's ``era52arl`` for ERA5 on pressure levels, taking the two
    Datasets of a Copernicus Climate Data Store download (``reanalysis-era5-
    pressure-levels`` and ``reanalysis-era5-single-levels``) as xarray opens
    them, from netCDF or, with ``cfgrib``, from GRIB. The result is an ARL
    pressure-level Dataset (vertical flag 2) on the same lat/lon grid, rows
    south to north, levels from the highest pressure up; nothing is read until
    it is written, when the inputs are lazy (opened with ``chunks=``).

    The variables converted are those of ``era52arl``'s default list, plus
    specific humidity. Each is written when the input has it; others are
    ignored.

    ============  ======  ===========================================
    ERA5          ARL     conversion
    ============  ======  ===========================================
    ``z``         HGTS    geopotential / 9.80665 (m)
    ``t``         TEMP    K
    ``u``, ``v``  UWND,   m s-1
                  VWND
    ``w``         WWND    Pa s-1 to hPa s-1, with a ``DIFW`` record
    ``r``         RELH    %
    ``q``         SPHU    kg kg-1
    ``sp``        PRSS    Pa to hPa
    ``z``         SHGT    surface geopotential / 9.80665 (m)
    ``t2m``       T02M    K
    ``d2m``       DP2M    K
    ``u10``,      U10M,   m s-1
    ``v10``       V10M
    ``blh``       PBLH    m
    ``cape``      CAPE    J kg-1
    ``tcc``       TCLD    fraction to %
    ``tp``        TPP1    m in the hour, with a ``DIFR`` record
    ``sshf``      SHTF    J m-2 in the hour to W m-2, positive upward
    ``slhf``      LHTF    J m-2 in the hour to W m-2, positive upward
    ``ssrd``      DSWF    J m-2 in the hour to W m-2
    ``zust``      USTR    m s-1
    ============  ======  ===========================================

    Parameters
    ----------
    pressure_levels : xarray.Dataset
        ERA5 fields on pressure levels, with dims valid time (``valid_time``
        or ``time``), pressure (``pressure_level``, ``isobaricInhPa`` or
        ``level``, in hPa), ``latitude`` and ``longitude``.
    single_levels : xarray.Dataset
        ERA5 single-level fields on the same grid, at every valid time of
        ``pressure_levels`` (it may hold others, which are left out). A CDS
        netCDF download comes as two files, the instantaneous and the
        accumulated fields: merge them first (:func:`xarray.merge`).

    Returns
    -------
    xarray.Dataset
        The fields in :func:`arlmet.open_dataset`'s layout, with source
        ``"ERA5"``, each variable's ARL ``units`` in its attrs, and
        ``attrs["diff"]`` on ``WWND`` and ``TPP1``.

    Raises
    ------
    ValueError
        If a Dataset has none of the variables above, a dimension is missing
        or extra (an ensemble ``number``, a forecast ``step``), the latitudes
        or longitudes are not equally spaced, the two grids differ, or
        ``single_levels`` lacks a valid time of ``pressure_levels``.

    Notes
    -----
    Where this differs from ``era52arl``: ``era52arl`` divides geopotential
    by 9.8 (a factor of 0.102), writes ``TCLD`` as a fraction where HYSPLIT
    reads percent, keeps ECMWF's downward-positive heat fluxes where HYSPLIT
    takes them positive upward, and names the latent heat flux ``LTHF``,
    a name HYSPLIT does not read.

    Examples
    --------
    >>> import xarray as xr
    >>> import arlmet
    >>> pl = xr.open_dataset("era5_pressure_levels.nc", chunks={})  # doctest: +SKIP
    >>> sfc = xr.merge(  # doctest: +SKIP
    ...     [xr.open_dataset("era5_instant.nc"), xr.open_dataset("era5_accum.nc")],
    ...     compat="override",
    ... )
    >>> ds = arlmet.converters.from_era5(pl, sfc)  # doctest: +SKIP
    >>> arlmet.write_dataset(ds, "era5.arl")  # doctest: +SKIP
    """
    upper = _standardize(
        pressure_levels, PRESSURE_LEVEL_FIELDS, upper=True, label="pressure_levels"
    )
    surface = _standardize(
        single_levels, SINGLE_LEVEL_FIELDS, upper=False, label="single_levels"
    )

    for axis in ("lat", "lon"):
        a, b = upper[axis].values, surface[axis].values
        if a.shape != b.shape or not np.allclose(
            (a - b + 180.0) % 360.0 - 180.0, 0.0, atol=1e-6
        ):
            raise ValueError(
                "pressure_levels and single_levels are on different grids "
                f"({axis} differs)."
            )
    missing = pd.Index(upper["time"].values).difference(
        pd.Index(surface["time"].values)
    )
    if len(missing):
        raise ValueError(
            f"single_levels lacks {len(missing)} valid time(s) of pressure_levels, "
            f"the first {missing[0]}."
        )
    surface = surface.sel(time=upper["time"].values)
    upper = upper.sortby("level", ascending=False)

    grid = latlon_grid(upper["lat"].values, upper["lon"].values)
    pressures = [float(p) for p in upper["level"].values]
    level_numbers = list(range(1, len(pressures) + 1))
    vertical_axis = PressureAxis(levels=[0.0, *pressures])
    times = upper["time"].values

    ds = add_cf_spatial_attrs(xr.Dataset(coords=grid.calculate_coords()))
    phys_name, phys_coord = physical_coord_for_vaxis(vertical_axis, level_numbers)
    ds = ds.assign_coords(
        time=xr.Variable("time", times, attrs={"standard_name": "time", "axis": "T"}),
        forecast_hour=xr.Variable(
            "time",
            np.zeros(len(times), dtype=int),
            attrs={"long_name": "ARL index record forecast hour", "units": "hours"},
        ),
        level=xr.Variable(
            "level", level_numbers, attrs={"standard_name": "model_level_number"}
        ),
        arl_grid=grid_to_coord(grid),
    )
    ds = ds.assign_coords({phys_name: phys_coord})

    upper = upper.drop_vars(["lat", "lon", "level", "time"])
    surface = surface.drop_vars(["lat", "lon", "time"])
    for name, field in PRESSURE_LEVEL_FIELDS.items():
        if name in upper:
            ds[field.name] = _field_variable(
                upper[name], field, ("time", "level", "lat", "lon")
            )
    for name, field in SINGLE_LEVEL_FIELDS.items():
        if name in surface:
            ds[field.name] = _field_variable(
                surface[name], field, ("time", "lat", "lon")
            )

    ds.attrs["source"] = "ERA5"
    ds.attrs["vertical_flag"] = vertical_axis.flag
    return ds
