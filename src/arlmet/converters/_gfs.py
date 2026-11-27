"""Convert NCEP's GFS and GDAS GRIB2 files on pressure levels, as HYSPLIT's ``api2arl``."""

from __future__ import annotations

from collections.abc import Iterable
from typing import NamedTuple

import numpy as np
import pandas as pd
import xarray as xr

from ._pressure import (
    Field,
    forecast_hours,
    keep_shortest_step,
    pressure_level_dataset,
    scalar_coord,
    stack_times,
    standardize,
    surface_dataset,
)

__all__ = ["GFS_HPBL", "from_gfs"]

#: GFS's pressure-level variables (cfgrib names, ``isobaricInhPa``) and their ARL fields.
PRESSURE_LEVEL_FIELDS: dict[str, Field] = {
    "gh": Field("HGTS", 1.0, "m"),
    "t": Field("TEMP", 1.0, "K"),
    "u": Field("UWND", 1.0, "m s-1"),
    "v": Field("VWND", 1.0, "m s-1"),
    "w": Field("WWND", 0.01, "hPa s-1"),
    "r": Field("RELH", 1.0, "%"),
    "q": Field("SPHU", 1.0, "kg kg-1"),
}


class _Source(NamedTuple):
    """Where a single-level field is in a GRIB2 file, as cfgrib names it."""

    type_of_level: str
    level: float | None  # the level's value, for heightAboveGround
    names: tuple[
        str, ...
    ]  # cfgrib's names for it (they changed between ecCodes versions)
    step_type: str  # GRIB's stepType: instant, accum or avg
    field: Field


#: GFS's single-level fields and their ARL fields. NCEP's fluxes are positive
#: upward already; accumulations and averages run from the start of the
#: forecast's 6-hour bucket (``TPP6``: HYSPLIT empties it every 6 hours).
SINGLE_LEVEL_SOURCES: tuple[_Source, ...] = (
    _Source("surface", None, ("sp",), "instant", Field("PRSS", 0.01, "hPa")),
    _Source("meanSea", None, ("prmsl",), "instant", Field("MSLP", 0.01, "hPa")),
    _Source("surface", None, ("orog",), "instant", Field("SHGT", 1.0, "m")),
    _Source("surface", None, ("t",), "instant", Field("TMPS", 1.0, "K")),
    _Source("heightAboveGround", 2.0, ("t2m",), "instant", Field("T02M", 1.0, "K")),
    _Source("heightAboveGround", 2.0, ("r2",), "instant", Field("RH2M", 1.0, "%")),
    _Source(
        "heightAboveGround", 10.0, ("u10",), "instant", Field("U10M", 1.0, "m s-1")
    ),
    _Source(
        "heightAboveGround", 10.0, ("v10",), "instant", Field("V10M", 1.0, "m s-1")
    ),
    _Source("atmosphere", None, ("tcc",), "instant", Field("TCLD", 1.0, "%")),
    _Source("surface", None, ("cape",), "instant", Field("CAPE", 1.0, "J kg-1")),
    _Source("surface", None, ("cin",), "instant", Field("CINH", 1.0, "J kg-1")),
    _Source("surface", None, ("fricv",), "instant", Field("USTR", 1.0, "m s-1")),
    _Source("surface", None, ("hpbl",), "instant", Field("PBLH", 1.0, "m")),
    _Source("surface", None, ("tp",), "accum", Field("TPP6", 0.001, "m")),
    _Source("surface", None, ("avg_ishf", "shtfl"), "avg", Field("SHTF", 1.0, "W m-2")),
    _Source(
        "surface", None, ("avg_slhtf", "lhtfl"), "avg", Field("LHTF", 1.0, "W m-2")
    ),
    _Source("surface", None, ("sdswrf", "dswrf"), "avg", Field("DSWF", 1.0, "W m-2")),
    _Source("surface", None, ("avg_utaua", "uflx"), "avg", Field("UMOF", 1.0, "N m-2")),
    _Source("surface", None, ("avg_vtaua", "vflx"), "avg", Field("VMOF", 1.0, "N m-2")),
)

#: The single-level fields by ARL name, scaled as they are found.
_SINGLE_LEVEL_FIELDS = {s.field.name: s.field for s in SINGLE_LEVEL_SOURCES}

#: NCEP's planetary boundary layer height, ``HPBL``, by its GRIB2 numbers.
#: ecCodes has no name for this local parameter, and cfgrib's
#: ``open_datasets`` drops it from a forecast file, where a second unnamed
#: field shares its level. Open it on its own with these as both
#: ``filter_by_keys`` and ``read_keys`` (:func:`from_gfs`).
GFS_HPBL: dict[str, int] = {
    "discipline": 0,
    "parameterCategory": 3,
    "parameterNumber": 196,
}


def _is_hpbl(da: xr.DataArray) -> bool:
    """Whether *da* is ``HPBL``, by the GRIB2 numbers cfgrib gives it with ``read_keys``."""
    return all(da.attrs.get(f"GRIB_{key}") == value for key, value in GFS_HPBL.items())


def _arl_name(ds: xr.Dataset, name: str, da: xr.DataArray) -> str | None:
    """Return the ARL name ``ds[name]`` converts to, or None if it is not converted."""
    type_of_level = da.attrs.get("GRIB_typeOfLevel")
    step_type = da.attrs.get("GRIB_stepType", "instant")
    if type_of_level == "isobaricInhPa":
        field = PRESSURE_LEVEL_FIELDS.get(name)
        if (
            field is not None
            and step_type == "instant"
            and "isobaricInhPa" in da.coords
        ):
            return field.name
        return None
    if type_of_level == "surface" and step_type == "instant" and _is_hpbl(da):
        return "PBLH"
    for source in SINGLE_LEVEL_SOURCES:
        if (
            type_of_level == source.type_of_level
            and name in source.names
            and step_type == source.step_type
        ):
            if source.level is not None:
                level = scalar_coord(ds, source.type_of_level)
                if level is None or float(level) != source.level:
                    continue
            return source.field.name
    return None


def _crop(da: xr.DataArray, bbox: tuple[float, float, float, float]) -> xr.DataArray:
    """
    Return the cells of *da* inside *bbox*, ``(west, south, east, north)`` in degrees.

    GRIB's longitudes run from 0 to 360; the box's may be on either
    convention and may cross 0 or 180. The kept longitudes are counted east
    from ``west``, so a box across 0 keeps its cells in order.
    """
    west, south, east, north = bbox
    if not south < north:
        raise ValueError(f"bbox needs south < north; got {bbox}.")
    lat = da["latitude"].values
    rows = np.flatnonzero((lat >= south) & (lat <= north))
    from_west = (da["longitude"].values - west) % 360.0
    width = (east - west) % 360.0 or 360.0
    cols = np.flatnonzero(from_west <= width)
    if len(rows) == 0 or len(cols) == 0:
        raise ValueError(f"bbox {bbox} holds no cell of the GFS grid.")
    cols = cols[np.argsort(from_west[cols], kind="stable")]
    cropped = da.isel(latitude=rows, longitude=cols)
    # GRIB's own longitudes, unless the box crosses 0: then counted from
    # west, so that they increase.
    lon = (west + from_west[cols]) % 360.0
    if (np.diff(lon) <= 0).any():
        lon = west + from_west[cols]
    return cropped.assign_coords(longitude=lon)


def from_gfs(
    datasets: Iterable[xr.Dataset],
    *,
    bbox: tuple[float, float, float, float] | None = None,
) -> xr.Dataset:
    """
    Convert GFS or GDAS GRIB2 files on pressure levels to a Dataset for :func:`arlmet.write_dataset`.

    This is HYSPLIT's ``api2arl`` for NCEP's ``pgrb2`` files on a lat/lon
    grid (GFS forecasts and GDAS analyses at 0.25, 0.5 or 1 degree), taking
    the Datasets :func:`cfgrib.open_datasets` returns for one or several
    files. Each valid time is one time step. Where two files hold a field at
    the same valid time, the one with the shorter forecast step is used: an
    analysis (``f000``) gives the instantaneous fields and the previous
    cycle's 6-hour forecast the precipitation and fluxes the analysis lacks,
    as NOAA's GDAS ARL files are made. A field a time step has no file for is
    written as a missing record there. The index record's forecast hour is
    the step of the file that gave the pressure-level fields.

    =====================  ======  ============================================
    GRIB2 (cfgrib name)    ARL     conversion
    =====================  ======  ============================================
    ``gh``                 HGTS    geopotential height (m)
    ``t``, ``u``, ``v``    TEMP,   K, m s-1
                           UWND,
                           VWND
    ``w``                  WWND    Pa s-1 to hPa s-1
    ``r``                  RELH    %
    ``q``                  SPHU    kg kg-1
    ``sp``                 PRSS    Pa to hPa
    ``prmsl``              MSLP    Pa to hPa
    ``orog``               SHGT    m
    ``t`` (surface)        TMPS    K
    ``t2m``, ``r2``        T02M,   K, %
                           RH2M
    ``u10``, ``v10``       U10M,   m s-1
                           V10M
    ``tcc`` (instant)      TCLD    %
    ``cape``, ``cin``      CAPE,   J kg-1
                           CINH
    ``fricv``              USTR    m s-1
    ``HPBL`` (see below)   PBLH    m
    ``tp``                 TPP6    kg m-2 to m, since the bucket's start
    ``avg_ishf``           SHTF    W m-2, positive upward
    ``avg_slhtf``          LHTF    W m-2, positive upward
    ``sdswrf``             DSWF    W m-2
    ``avg_utaua``,         UMOF,   N m-2
    ``avg_vtaua``          VMOF
    =====================  ======  ============================================

    Older ecCodes name the averaged fluxes ``shtfl``, ``lhtfl``, ``dswrf``,
    ``uflx`` and ``vflx``; both names are read.

    NCEP's planetary boundary layer height, ``HPBL``, is a local GRIB2
    parameter that ecCodes (2.50 and earlier) has no name for, so cfgrib
    calls it ``unknown``, and :func:`cfgrib.open_datasets` drops it from a
    forecast file, where a second unnamed field shares its level. Open it on
    its own with :data:`GFS_HPBL` (Examples): it is found by its numbers
    and written as ``PBLH``. Without it no ``PBLH`` is written. A future
    ecCodes that names it ``hpbl`` needs no second open.

    Parameters
    ----------
    datasets : iterable of xarray.Dataset
        The Datasets of one or more GRIB2 files, as :func:`cfgrib.open_datasets`
        returns them (each with a scalar ``valid_time``), one list after
        another. Datasets and variables not in the table are left out. Every
        time step must have the same pressure levels; select them first
        (``ds.sel(isobaricInhPa=...)``) to write fewer.
    bbox : tuple of float, optional
        A bounding box ``(west, south, east, north)`` in degrees, as
        :func:`arlmet.open_dataset` takes: only the grid cells inside it are
        converted. Longitudes may be from -180 to 180 or from 0 to 360, and
        the box may cross 0. ``None`` converts the whole grid.

    Returns
    -------
    xarray.Dataset
        The fields in :func:`arlmet.open_dataset`'s layout, with source
        ``"GFS"`` and each variable's ARL ``units`` in its attrs.

    Raises
    ------
    ValueError
        If no Dataset holds pressure-level fields, a Dataset has no scalar
        valid time, the time steps have different pressure levels, or the
        grids differ.

    Notes
    -----
    A field missing at every point (the GDAS files' instantaneous ``tcc``) is
    left out when it is written.
    No ``DIF*`` records are written, as ``api2arl`` writes none.

    Each field is read from its file once, cropped to *bbox* first, and is
    then in memory: a global 0.25 degree time step is about 3 GB, the
    western United States about 0.2 GB.

    Examples
    --------
    >>> import cfgrib  # doctest: +SKIP
    >>> import arlmet
    >>> files = ["gdas.t18z.pgrb2.1p00.f006", "gdas.t00z.pgrb2.1p00.f000"]
    >>> import xarray as xr
    >>> hpbl = {
    ...     "filter_by_keys": arlmet.converters.GFS_HPBL,
    ...     "read_keys": list(arlmet.converters.GFS_HPBL),
    ... }
    >>> datasets = [
    ...     ds for f in files for ds in cfgrib.open_datasets(f)
    ... ]  # doctest: +SKIP
    >>> datasets += [
    ...     xr.open_dataset(f, engine="cfgrib", backend_kwargs=hpbl) for f in files
    ... ]  # doctest: +SKIP
    >>> ds = arlmet.converters.from_gfs(datasets)  # doctest: +SKIP
    >>> arlmet.write_dataset(ds, "gdas.arl")  # doctest: +SKIP
    """
    # (valid time, ARL name) -> (forecast step, field without its scalar coords)
    upper: dict[tuple[pd.Timestamp, str], tuple[pd.Timedelta, xr.DataArray]] = {}
    surface: dict[tuple[pd.Timestamp, str], tuple[pd.Timedelta, xr.DataArray]] = {}
    for ds in datasets:
        if not isinstance(ds, xr.Dataset):
            raise TypeError(
                f"from_gfs() takes xarray.Datasets, got {type(ds).__name__}."
            )
        names = {
            str(name): arl
            for name, da in ds.data_vars.items()
            if (arl := _arl_name(ds, str(name), da)) is not None
        }
        if not names:
            continue
        valid_time = scalar_coord(ds, "valid_time")
        if valid_time is None:
            raise ValueError(
                "Each Dataset needs a scalar 'valid_time', as cfgrib.open_datasets() "
                f"gives one per GRIB file; got one with dims {dict(ds.sizes)}."
            )
        time = pd.Timestamp(valid_time)
        step_value = scalar_coord(ds, "step")
        step = pd.Timedelta(0 if step_value is None else step_value)
        for name, arl in names.items():
            da = ds[name]
            if "isobaricInhPa" in da.coords and "isobaricInhPa" not in da.dims:
                # One level: cfgrib squeezes it to a scalar, which the next line drops.
                da = da.expand_dims("isobaricInhPa")
            da = da.reset_coords(drop=True)
            if bbox is not None:
                da = _crop(da, bbox)
            # Read once: cfgrib reads a whole message on every access.
            da = da.load()
            target = (
                upper
                if arl in {f.name for f in PRESSURE_LEVEL_FIELDS.values()}
                else surface
            )
            keep_shortest_step(target, (time, arl), step, da)

    times = sorted({time for time, _ in upper})
    if not times:
        raise ValueError(
            "None of the Datasets holds GFS pressure-level fields "
            f"({', '.join(PRESSURE_LEVEL_FIELDS)} on isobaricInhPa)."
        )
    hours = forecast_hours(upper, times)

    upper_ds = standardize(
        stack_times({k: da for k, (_, da) in upper.items()}, times),
        [f.name for f in PRESSURE_LEVEL_FIELDS.values()],
        upper=True,
        label="pressure levels",
    )
    surface_ds = surface_dataset(
        {k: da for k, (_, da) in surface.items()}, times, upper_ds, _SINGLE_LEVEL_FIELDS
    )
    upper_fields = {f.name: f for f in PRESSURE_LEVEL_FIELDS.values()}
    return pressure_level_dataset(
        upper_ds,
        surface_ds,
        upper_fields,
        _SINGLE_LEVEL_FIELDS,
        source="GFS",
        forecast_hours=hours,
        labels=("the pressure-level fields", "the single-level fields"),
    )
