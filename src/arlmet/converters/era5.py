"""Convert ERA5 pressure-level and single-level fields, as HYSPLIT's ``era52arl``."""

from __future__ import annotations

import xarray as xr

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
    upper = standardize(
        pressure_levels, PRESSURE_LEVEL_FIELDS, upper=True, label="pressure_levels"
    )
    surface = standardize(
        single_levels, SINGLE_LEVEL_FIELDS, upper=False, label="single_levels"
    )
    return pressure_level_dataset(
        upper, surface, PRESSURE_LEVEL_FIELDS, SINGLE_LEVEL_FIELDS, source="ERA5"
    )
