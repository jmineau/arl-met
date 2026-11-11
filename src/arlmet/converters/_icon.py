"""Convert DWD's ICON forecasts on a regular lat/lon grid (ICON-EU, ICON-D2 GRIB2)."""

from __future__ import annotations

import functools
import operator
from collections import defaultdict
from collections.abc import Iterable
from typing import Literal, NamedTuple

import pandas as pd
import xarray as xr

from ._pressure import (
    Field,
    pressure_level_dataset,
    scalar_coord,
    stack_times,
    standardize,
)

__all__ = ["from_icon"]

GRAVITY = 9.80665  # m s-2, the WMO standard ICON's geopotential is divided by

#: ICON's pressure-level variables (cfgrib names, ``isobaricInhPa``) and their ARL fields.
PRESSURE_LEVEL_FIELDS: dict[str, Field] = {
    "z": Field("HGTS", 1.0 / GRAVITY, "m"),
    "t": Field("TEMP", 1.0, "K"),
    "u": Field("UWND", 1.0, "m s-1"),
    "v": Field("VWND", 1.0, "m s-1"),
    "w": Field("WWND", 0.01, "hPa s-1"),
    "r": Field("RELH", 1.0, "%"),
}

#: How a single-level field is stored: at the valid time, as a mean or a total
#: from the start of the forecast, or once for every time (``HSURF``).
_Kind = Literal["instant", "mean", "accum", "invariant"]


class _Source(NamedTuple):
    """One ICON single-level variable and the ARL field it gives."""

    names: tuple[str, ...]  # cfgrib's names for it: ecCodes', DWD's, lower case
    kind: _Kind
    field: Field


#: ICON's single-level fields and their ARL fields. ICON's fluxes are positive
#: downward, so the heat fluxes change sign; ``DSWF`` is the direct plus the
#: diffuse downward short-wave flux.
SINGLE_LEVEL_SOURCES: tuple[_Source, ...] = (
    _Source(("sp", "PS", "ps"), "instant", Field("PRSS", 0.01, "hPa")),
    _Source(("prmsl", "PMSL", "pmsl"), "instant", Field("MSLP", 0.01, "hPa")),
    _Source(("HSURF", "hsurf"), "invariant", Field("SHGT", 1.0, "m")),
    _Source(("T_G", "t_g"), "instant", Field("TMPS", 1.0, "K")),
    _Source(("t2m", "T_2M", "t_2m"), "instant", Field("T02M", 1.0, "K")),
    _Source(("r2", "RELHUM_2M", "relhum_2m"), "instant", Field("RH2M", 1.0, "%")),
    _Source(("u10", "U_10M", "u_10m"), "instant", Field("U10M", 1.0, "m s-1")),
    _Source(("v10", "V_10M", "v_10m"), "instant", Field("V10M", 1.0, "m s-1")),
    _Source(("CLCT", "clct"), "instant", Field("TCLD", 1.0, "%")),
    _Source(("CAPE_ML", "cape_ml"), "instant", Field("CAPE", 1.0, "J kg-1")),
    _Source(("CIN_ML", "cin_ml"), "instant", Field("CINH", 1.0, "J kg-1")),
    _Source(("mld", "MH", "mh"), "instant", Field("PBLH", 1.0, "m")),
    _Source(("fsr", "Z0", "z0"), "instant", Field("RGHS", 1.0, "m")),
    _Source(("tp", "TOT_PREC", "tot_prec"), "accum", Field("TPP1", 0.001, "m")),
    _Source(("avg_ishf", "ASHFL_S", "ashfl_s"), "mean", Field("SHTF", -1.0, "W m-2")),
    _Source(("avg_slhtf", "ALHFL_S", "alhfl_s"), "mean", Field("LHTF", -1.0, "W m-2")),
    _Source(("ASWDIR_S", "aswdir_s"), "mean", Field("DSWF", 1.0, "W m-2")),
    _Source(("ASWDIFD_S", "aswdifd_s"), "mean", Field("DSWF", 1.0, "W m-2")),
    _Source(("AUMFL_S", "aumfl_s"), "mean", Field("UMOF", 1.0, "N m-2")),
    _Source(("AVMFL_S", "avmfl_s"), "mean", Field("VMOF", 1.0, "N m-2")),
)

#: The single-level fields by ARL name, scaled as they are found.
_SINGLE_LEVEL_FIELDS = {s.field.name: s.field for s in SINGLE_LEVEL_SOURCES}
#: The GRIB stepType each kind is stored with.
_STEP_TYPES: dict[_Kind, str] = {
    "instant": "instant",
    "mean": "avg",
    "accum": "accum",
    "invariant": "instant",
}
_HOUR = pd.Timedelta("1h")

# (forecast start, step, name) -> field
_Found = dict[tuple[pd.Timestamp, pd.Timedelta, str], xr.DataArray]


def _source(name: str, da: xr.DataArray) -> _Source | None:
    """Return the single-level source ``da`` (cfgrib's ``name``) is, or None."""
    step_type = da.attrs.get("GRIB_stepType", "instant")
    for source in SINGLE_LEVEL_SOURCES:
        if name in source.names and step_type == _STEP_TYPES[source.kind]:
            return source
    return None


def _since_step_before(found: _Found, kind: _Kind) -> _Found:
    """
    Turn means or totals from the forecast start into ones since the step before.

    Each step of a run is paired with the run's step before it among those
    given (the forecast start, where the value is 0, for the first). A mean
    becomes the mean over that interval, a total the total per hour of it.
    Step 0 has no interval and is left out.
    """
    by_run: dict[tuple[pd.Timestamp, str], dict[pd.Timedelta, xr.DataArray]]
    by_run = defaultdict(dict)
    for (start, step, name), da in found.items():
        by_run[(start, name)][step] = da
    out: _Found = {}
    for (start, name), steps in by_run.items():
        previous, before = pd.Timedelta(0), None
        for step in sorted(steps):
            da = steps[step]
            if step > previous:
                hours = (step - previous) / _HOUR
                if kind == "mean":
                    total = da * (step / _HOUR)
                    if before is not None:
                        total = total - before * (previous / _HOUR)
                    value = total / hours
                else:
                    value = (da if before is None else da - before) / hours
                out[(start, step, name)] = value
            previous, before = step, da
    return out


def from_icon(datasets: Iterable[xr.Dataset]) -> xr.Dataset:
    """
    Convert ICON forecasts on a regular lat/lon grid to a Dataset for :func:`arlmet.write_dataset`.

    This reads DWD's ICON-EU and ICON-D2 GRIB2 files on their regular
    lat/lon grids (DWD's open data, ``regular-lat-lon_pressure-level`` and
    ``single-level``, one file per variable, level and forecast step), taking
    the Datasets :func:`cfgrib.open_datasets` returns for them. HYSPLIT has no
    converter for ICON. Each valid time is one time step, on ICON's pressure
    levels (vertical flag 2); the levels of each variable are stacked.

    ICON's precipitation and surface fluxes are totals and means from the
    start of the forecast. Each is written as the total or mean since the
    step before among the files given, of the same forecast (since the start
    of the forecast for its first): ``TPP1`` is that precipitation per hour,
    as ``arw2arl`` writes WRF's. A forecast's step 0 has none. Where two
    files hold a field at the same valid time, the one with the shorter
    forecast step is used, as in :func:`from_gfs`, so the forecasts of
    several runs make one series, each step 0 taking its fluxes from the run
    before. A field a time step has no file for is written as a missing
    record there. The index record's forecast hour is the step of the
    pressure-level fields.

    =====================  ======  ============================================
    ICON (cfgrib name)     ARL     conversion
    =====================  ======  ============================================
    ``FI`` (``z``)         HGTS    geopotential (m2 s-2) / 9.80665
    ``T``, ``U``, ``V``    TEMP,   K, m s-1
                           UWND,
                           VWND
    ``OMEGA`` (``w``)      WWND    Pa s-1 to hPa s-1
    ``RELHUM`` (``r``)     RELH    %
    ``PS``                 PRSS    Pa to hPa
    ``PMSL``               MSLP    Pa to hPa
    ``HSURF``              SHGT    m, the same at every time step
    ``T_G``                TMPS    K
    ``T_2M``, ``RELHUM``   T02M,   K, %
    ``_2M``                RH2M
    ``U_10M``, ``V_10M``   U10M,   m s-1
                           V10M
    ``CLCT``               TCLD    %
    ``CAPE_ML``,           CAPE,   J kg-1
    ``CIN_ML``             CINH
    ``MH``                 PBLH    m
    ``Z0``                 RGHS    m
    ``TOT_PREC``           TPP1    kg m-2 to m, per hour since the step before
    ``ASHFL_S``            SHTF    W m-2, sign changed (positive upward)
    ``ALHFL_S``            LHTF    W m-2, sign changed (positive upward)
    ``ASWDIR_S`` +         DSWF    W m-2, direct plus diffuse downward
    ``ASWDIFD_S``
    ``AUMFL_S``,           UMOF,   N m-2
    ``AVMFL_S``            VMOF
    =====================  ======  ============================================

    cfgrib names some of these by ecCodes' standard names (``sp``, ``t2m``,
    ``tp``, ``avg_slhtf``, ``mld``, ``fsr``, ...) and the rest by DWD's;
    both, and DWD's in lower case, are read.

    Parameters
    ----------
    datasets : iterable of xarray.Dataset
        The Datasets of ICON GRIB2 files, as :func:`cfgrib.open_datasets`
        returns them (each with scalar ``time``, ``step`` and ``valid_time``
        coordinates), one list after another. Datasets and variables not in
        the table are left out. Every time step must have the same pressure
        levels.

    Returns
    -------
    xarray.Dataset
        The fields in :func:`arlmet.open_dataset`'s layout, with source
        ``"ICON"`` and each variable's ARL ``units`` in its attrs.

    Raises
    ------
    ValueError
        If no Dataset holds pressure-level fields, a Dataset lacks its scalar
        times, a field is given twice for one step, the time steps have
        different pressure levels, or the grids differ.

    Notes
    -----
    ICON's own grid (icosahedral, as in DWD's global ICON files) and its
    model levels (heights that follow the terrain, which no ARL vertical
    coordinate describes) are not converted; regrid to a lat/lon grid and
    interpolate to pressure levels first. No ``DIF*`` records are written.

    Examples
    --------
    >>> import glob
    >>> import cfgrib  # doctest: +SKIP
    >>> import arlmet
    >>> files = sorted(glob.glob("icon-eu_europe_regular-lat-lon_*.grib2"))
    >>> datasets = [
    ...     ds for f in files for ds in cfgrib.open_datasets(f)
    ... ]  # doctest: +SKIP
    >>> ds = arlmet.converters.from_icon(datasets)  # doctest: +SKIP
    >>> arlmet.write_dataset(ds, "icon-eu.arl")  # doctest: +SKIP
    """
    # (forecast start, step, name) -> {pressure: field}
    upper_levels: dict[
        tuple[pd.Timestamp, pd.Timedelta, str], dict[float, xr.DataArray]
    ]
    upper_levels = defaultdict(dict)
    single: dict[_Kind, _Found] = {kind: {} for kind in _STEP_TYPES}
    for ds in datasets:
        if not isinstance(ds, xr.Dataset):
            raise TypeError(
                f"from_icon() takes xarray.Datasets, got {type(ds).__name__}."
            )
        wanted: dict[str, _Source | None] = {}
        for key, da in ds.data_vars.items():
            name = str(key)
            if (
                name in PRESSURE_LEVEL_FIELDS
                and da.attrs.get("GRIB_typeOfLevel") == "isobaricInhPa"
                and da.attrs.get("GRIB_stepType", "instant") == "instant"
            ):
                wanted[name] = None
            elif (source := _source(name, da)) is not None:
                wanted[name] = source
        if not wanted:
            continue
        start, step = scalar_coord(ds, "time"), scalar_coord(ds, "step")
        if start is None or step is None:
            raise ValueError(
                "Each Dataset needs scalar 'time' (the forecast start) and 'step' "
                "coordinates, as cfgrib.open_datasets() gives one per GRIB file; "
                f"got one with dims {dict(ds.sizes)}."
            )
        start, step = pd.Timestamp(start), pd.Timedelta(step)
        for name, source in wanted.items():
            da = ds[name]
            if source is None:
                if "isobaricInhPa" in da.dims:
                    levels = [
                        da.sel(isobaricInhPa=p) for p in da["isobaricInhPa"].values
                    ]
                else:
                    levels = [da]
                for level in levels:
                    pressure = float(level["isobaricInhPa"])
                    kept = upper_levels[(start, step, name)]
                    if pressure in kept:
                        raise ValueError(
                            f"ICON's '{name}' at {pressure:g} hPa is given twice for "
                            f"the forecast from {start} at step {step}."
                        )
                    kept[pressure] = level.reset_coords(drop=True)
                continue
            key = (start, step, source.names[0])
            if key in single[source.kind]:
                raise ValueError(
                    f"ICON's '{name}' is given twice for the forecast from {start} "
                    f"at step {step}."
                )
            single[source.kind][key] = da.reset_coords(drop=True)

    def by_valid_time(
        found: _Found,
    ) -> dict[tuple[pd.Timestamp, str], tuple[pd.Timedelta, xr.DataArray]]:
        """Return the shortest forecast step's field at each valid time."""
        out: dict[tuple[pd.Timestamp, str], tuple[pd.Timedelta, xr.DataArray]] = {}
        for (start, step, name), da in found.items():
            kept = out.get((start + step, name))
            if kept is None or step < kept[0]:
                out[(start + step, name)] = (step, da)
        return out

    upper_found: _Found = {
        key: xr.concat(
            [levels[p] for p in sorted(levels)],
            dim=pd.Index(sorted(levels), name="isobaricInhPa"),
        )
        for key, levels in upper_levels.items()
    }
    upper = by_valid_time(upper_found)
    times = sorted({time for time, _ in upper})
    if not times:
        raise ValueError(
            "None of the Datasets holds ICON pressure-level fields "
            f"({', '.join(PRESSURE_LEVEL_FIELDS)} on isobaricInhPa)."
        )
    hours = [
        int(min(step for (t, _), (step, _) in upper.items() if t == time) / _HOUR)
        for time in times
    ]

    parts = {
        **by_valid_time(single["instant"]),
        **by_valid_time(_since_step_before(single["mean"], "mean")),
        **by_valid_time(_since_step_before(single["accum"], "accum")),
    }
    invariant = {name: da for (_, _, name), da in single["invariant"].items()}
    surface: dict[tuple[pd.Timestamp, str], xr.DataArray] = {}
    for arl in _SINGLE_LEVEL_FIELDS:
        sources = [s.names[0] for s in SINGLE_LEVEL_SOURCES if s.field.name == arl]
        for time in times:
            if all(name in invariant for name in sources):
                surface[(time, arl)] = invariant[sources[0]]
            elif all((time, name) in parts for name in sources):
                fields = [parts[(time, name)][1] for name in sources]
                surface[(time, arl)] = functools.reduce(operator.add, fields)

    upper_ds = standardize(
        stack_times({k: da for k, (_, da) in upper.items()}, times),
        list(PRESSURE_LEVEL_FIELDS),
        upper=True,
        label="pressure levels",
    )
    surface_ds = (
        standardize(
            stack_times(surface, times),
            list(_SINGLE_LEVEL_FIELDS),
            upper=False,
            label="single levels",
        )
        if surface
        else upper_ds.isel(level=0, drop=True)[[]]
    )
    return pressure_level_dataset(
        upper_ds,
        surface_ds,
        PRESSURE_LEVEL_FIELDS,
        _SINGLE_LEVEL_FIELDS,
        source="ICON",
        forecast_hours=hours,
        labels=("the pressure-level fields", "the single-level fields"),
    )
