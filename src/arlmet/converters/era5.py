"""Convert ERA5 fields on pressure or model levels, as HYSPLIT's ``era52arl`` and ``api2arl``."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr

from arlmet.indexrecord import format_fixed_width_float
from arlmet.vertical import HybridAxis, VerticalAxis

from ._l137 import L137_A, L137_B
from ._pressure import Field, pressure_level_dataset, standardize

__all__ = ["from_era5"]

#: ECMWF's gravity (m s-2): geopotential height is geopotential over it.
GRAVITY = 9.80665
#: Seconds in the hour each ERA5 hourly accumulation covers.
HOUR = 3600.0


#: ERA5 pressure-level variables (netCDF and cfgrib names) and their ARL fields.
PRESSURE_LEVEL_FIELDS: dict[str, Field] = {
    "z": Field("HGTS", 1.0 / GRAVITY, "m"),
    "t": Field("TEMP", 1.0, "K"),
    "u": Field("UWND", 1.0, "m s-1"),
    "v": Field("VWND", 1.0, "m s-1"),
    "w": Field("WWND", 0.01, "hPa s-1", "DIFW"),
    "r": Field("RELH", 1.0, "%"),
    "q": Field("SPHU", 1.0, "kg kg-1"),
}

#: ERA5 model-level variables (netCDF and cfgrib names) and their ARL fields,
#: as HYSPLIT's ``api2arl`` converts them.
MODEL_LEVEL_FIELDS: dict[str, Field] = {
    "t": Field("TEMP", 1.0, "K"),
    "u": Field("UWND", 1.0, "m s-1"),
    "v": Field("VWND", 1.0, "m s-1"),
    "w": Field("WWND", 0.01, "hPa s-1", "DIFW"),
    "q": Field("SPHU", 1.0, "kg kg-1"),
}

#: Names of the model-level dimension: the CDS's netCDF, cfgrib.
MODEL_LEVEL_DIMS = ("model_level", "hybrid")

#: ERA5 single-level variables and their ARL fields. Accumulations are over the
#: hour before the valid time; ECMWF's fluxes are positive downward, HYSPLIT's
#: heat fluxes positive upward.
SINGLE_LEVEL_FIELDS: dict[str, Field] = {
    "sp": Field("PRSS", 0.01, "hPa"),
    "z": Field("SHGT", 1.0 / GRAVITY, "m"),
    "t2m": Field("T02M", 1.0, "K"),
    "d2m": Field("DP2M", 1.0, "K"),
    "u10": Field("U10M", 1.0, "m s-1"),
    "v10": Field("V10M", 1.0, "m s-1"),
    "blh": Field("PBLH", 1.0, "m"),
    "cape": Field("CAPE", 1.0, "J kg-1"),
    "tcc": Field("TCLD", 100.0, "%"),
    "tp": Field("TPP1", 1.0, "m", "DIFR"),
    "sshf": Field("SHTF", -1.0 / HOUR, "W m-2"),
    "slhf": Field("LHTF", -1.0 / HOUR, "W m-2"),
    "ssrd": Field("DSWF", 1.0 / HOUR, "W m-2"),
    "zust": Field("USTR", 1.0, "m s-1"),
}


def _hybrid_levels(model_levels: Sequence[int]) -> list[float]:
    """
    Return the ARL level values of ERA5 model levels, as HYSPLIT's ``api2arl`` writes them.

    A hybrid (flag 4) ARL level is one number: its whole part is a pressure
    offset in hPa and its fraction a sigma, so that the level's pressure is
    ``offset + sigma * surface pressure`` (HYSPLIT's ``prfecm``). For model
    level ``n`` of ECMWF's 137 (1 at the top, 137 at the ground) the offset is
    the mean of the ``a`` coefficients of half levels ``n - 1`` and ``n`` in
    hPa, rounded to the nearest whole hPa, and sigma the mean of their ``b``.
    Raises if a level is not a whole number from 1 to 137.
    """
    values = []
    for level in model_levels:
        n = int(level)
        if n != level or not 1 <= n <= 137:
            raise ValueError(
                f"ERA5 model levels are whole numbers from 1 to 137, got {level}."
            )
        offset = np.floor(0.5 * (L137_A[n - 1] + L137_A[n]) / 100.0 + 0.5)
        values.append(float(offset + 0.5 * (L137_B[n - 1] + L137_B[n])))
    return values


def _hybrid_axis(model_levels: Sequence[float]) -> VerticalAxis:
    """
    Return the hybrid axis of ``model_levels`` (lowest first), the ground first as 1.

    Raises if two levels are stored as the same pressure: a level above about
    model level 24 rounds to the same whole hPa as the one below it.
    """
    stored = [
        float(format_fixed_width_float(value, 6))
        for value in _hybrid_levels([int(n) for n in model_levels])
    ]
    for below, above, n in zip(stored, stored[1:], model_levels[1:], strict=False):
        if _pressure_at(above) >= _pressure_at(below):
            raise ValueError(
                f"ERA5 model level {int(n)} is stored as the same pressure as the "
                f"level below it: an ARL hybrid level holds its pressure offset in "
                f"whole hPa. Leave out level {int(n)} and those above it (keep "
                f"levels {int(n) + 1} to 137)."
            )
    return HybridAxis(levels=[1.0, *stored])


def _pressure_at(value: float, surface_pressure: float = 1013.25) -> float:
    """Return the pressure [hPa] HYSPLIT reads from a hybrid level value."""
    offset = np.floor(value)
    return float(offset + (value - offset) * surface_pressure)


def from_era5(levels: xr.Dataset, single_levels: xr.Dataset) -> xr.Dataset:
    """
    Convert ERA5 hourly fields to a Dataset that :func:`arlmet.write_dataset` writes.

    This is HYSPLIT's ``era52arl`` for ERA5 on pressure levels and its
    ``api2arl`` for ERA5 on model levels, taking two Datasets of a Copernicus
    Climate Data Store download as xarray opens them, from netCDF or, with
    ``cfgrib``, from GRIB: ``levels`` (``reanalysis-era5-pressure-levels``,
    or model levels from ``reanalysis-era5-complete``) and ``single_levels``
    (``reanalysis-era5-single-levels``). The result is on the same lat/lon
    grid, rows south to north, levels from the ground up; nothing is read
    until it is written, when the inputs are lazy (opened with ``chunks=``).

    - **Pressure levels** give an ARL pressure-level Dataset (vertical flag
      2), its levels from the highest pressure up.
    - **Model levels** (a ``model_level`` or cfgrib's ``hybrid`` dimension,
      1 at the top to 137 at the ground) give an ARL hybrid Dataset (flag 4),
      its levels from model level 137 up, each written as ``api2arl``
      writes it (Notes). HYSPLIT finds each level's pressure
      from the surface pressure, so ``single_levels`` must hold ``sp``.
      Levels above about model level 24 cannot be written: their pressure
      offsets round to the same whole hPa (see Raises).

    The variables converted are those of ``era52arl``'s default list, plus
    specific humidity, and on model levels those of ``api2arl``'s. Each is
    written when the input has it; others are ignored.

    ==============  ======  ===========================================
    ERA5            ARL     conversion
    ==============  ======  ===========================================
    ``z``           HGTS    geopotential / 9.80665 (m; pressure levels)
    ``t``           TEMP    K
    ``u``, ``v``    UWND,   m s-1
                    VWND
    ``w``           WWND    Pa s-1 to hPa s-1, with a ``DIFW`` record
    ``r``           RELH    % (pressure levels)
    ``q``           SPHU    kg kg-1
    ``sp``          PRSS    Pa to hPa
    ``z``           SHGT    surface geopotential / 9.80665 (m)
    ``t2m``         T02M    K
    ``d2m``         DP2M    K
    ``u10``,        U10M,   m s-1
    ``v10``         V10M
    ``blh``         PBLH    m
    ``cape``        CAPE    J kg-1
    ``tcc``         TCLD    fraction to %
    ``tp``          TPP1    m in the hour, with a ``DIFR`` record
    ``sshf``        SHTF    J m-2 in the hour to W m-2, positive upward
    ``slhf``        LHTF    J m-2 in the hour to W m-2, positive upward
    ``ssrd``        DSWF    J m-2 in the hour to W m-2
    ``zust``        USTR    m s-1
    ==============  ======  ===========================================

    Parameters
    ----------
    levels : xarray.Dataset
        ERA5 fields on pressure levels or on model levels, with dims valid
        time (``valid_time`` or ``time``), the level (``pressure_level``,
        ``isobaricInhPa`` or ``level``, in hPa; or ``model_level`` or
        ``hybrid``), ``latitude`` and ``longitude``.
    single_levels : xarray.Dataset
        ERA5 single-level fields on the same grid, at every valid time of
        ``levels`` (it may hold others, which are left out). A CDS
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
        ``single_levels`` lacks a valid time of ``levels``. On model levels,
        also if ``single_levels`` has no ``sp``, or two of the levels would be
        stored as the same pressure (any above about model level 24): select
        the levels below, for example ``levels.sel(model_level=slice(24,
        None))``.

    Notes
    -----
    Where this differs from ``era52arl``: ``era52arl`` divides geopotential
    by 9.8 (a factor of 0.102), turns the hourly totals of the heat fluxes and
    solar radiation into W m-2 with a factor of 0.00028 for 1/3600 (so its
    ``DSWF`` is 0.8 % high), writes ``TCLD`` as a fraction where HYSPLIT
    reads percent, keeps ECMWF's downward-positive heat fluxes where HYSPLIT
    takes them positive upward (``stbanl.f``; HYSPLIT changes the sign only
    for ``ETA`` files, in ``prfcom.f``), so HYSPLIT reads a sunny afternoon in
    an ``era52arl`` file as stable, and names the latent heat flux ``LTHF``,
    a name HYSPLIT does not read. ``era52arl`` does not convert model levels.

    An ARL hybrid level is one number, a pressure offset in whole hPa plus a
    sigma, the level's pressure being ``offset + sigma * surface pressure``
    (HYSPLIT's ``prfecm``). As in ``api2arl``, model level ``n``'s offset is
    the mean ``a`` of its two half levels in hPa, rounded, and its sigma their
    mean ``b`` (ECMWF's L137 coefficients). The index record holds each level
    in six characters, so above an offset of 100 hPa sigma keeps two
    decimals: HYSPLIT reads some mid-troposphere levels up to about 5 hPa
    from ERA5's pressure (model level 100: 584.6 hPa for 589.7 at a surface
    pressure of 1013.25 hPa). ``api2arl``'s files are rounded the same way.

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

    On model levels, from GRIB (``cfgrib`` names the dimension ``hybrid``):

    >>> ml = xr.open_dataset(  # doctest: +SKIP
    ...     "era5_model_levels.grib",
    ...     engine="cfgrib",
    ...     backend_kwargs={"filter_by_keys": {"typeOfLevel": "hybrid"}},
    ... )
    >>> ds = arlmet.converters.from_era5(
    ...     ml.sel(hybrid=slice(60, None)), sfc
    ... )  # doctest: +SKIP
    """
    if not isinstance(levels, xr.Dataset):
        raise TypeError(
            f"levels must be an xarray.Dataset, got {type(levels).__name__}."
        )
    surface = standardize(
        single_levels, SINGLE_LEVEL_FIELDS, upper=False, label="single_levels"
    )
    if not any(dim in levels.dims for dim in MODEL_LEVEL_DIMS):
        upper = standardize(levels, PRESSURE_LEVEL_FIELDS, upper=True, label="levels")
        return pressure_level_dataset(
            upper,
            surface,
            PRESSURE_LEVEL_FIELDS,
            SINGLE_LEVEL_FIELDS,
            source="ERA5",
            labels=("levels", "single_levels"),
        )
    if "sp" not in surface:
        raise ValueError(
            "single_levels has no surface pressure 'sp', which ERA5 on model "
            "levels needs: HYSPLIT finds the levels' pressures from it."
        )
    upper = standardize(
        levels,
        MODEL_LEVEL_FIELDS,
        upper=True,
        label="levels",
        level_dims=MODEL_LEVEL_DIMS,
        level_kind="model-level",
    )
    return pressure_level_dataset(
        upper,
        surface,
        MODEL_LEVEL_FIELDS,
        SINGLE_LEVEL_FIELDS,
        source="ERA5",
        labels=("levels", "single_levels"),
        vertical_axis=_hybrid_axis,
    )
