"""Convert WRF-ARW output, as HYSPLIT's ``arw2arl``."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

from arlmet._header import format_fixed_width_float
from arlmet.grid import Grid, Projection
from arlmet.vertical import SigmaAxis, VerticalAxis, WrfAxis

from ._pressure import Field, arl_dataset, field_variable

__all__ = ["from_wrf"]

#: The constants ``arw2arl`` uses: gravity (m s-2), the gas constant of dry
#: air and its specific heat (J kg-1 K-1).
GRAVITY = 9.81
R_DRY = 287.04
CP = 1004.0

#: WRF's single-level variables and their ARL fields (``arw2arl``'s default list).
SURFACE_FIELDS: dict[str, Field] = {
    "HGT": Field("SHGT", 1.0, "m"),
    "PSFC": Field("PRSS", 0.01, "hPa"),
    "PBLH": Field("PBLH", 1.0, "m"),
    "UST": Field("USTR", 1.0, "m s-1"),
    "SWDOWN": Field("DSWF", 1.0, "W m-2"),
    "HFX": Field("SHTF", 1.0, "W m-2"),
    "LH": Field("LHTF", 1.0, "W m-2"),
    "T2": Field("T02M", 1.0, "K"),
    "U10": Field("U10M", 1.0, "m s-1"),
    "V10": Field("V10M", 1.0, "m s-1"),
}

#: The derived fields, by ARL name.
PRECIPITATION = Field("TPP1", 1.0, "m", "DIFR")
LEVEL_FIELDS: dict[str, Field] = {
    "PRES": Field("PRES", 1.0, "hPa"),
    "TEMP": Field("TEMP", 1.0, "K"),
    "UWND": Field("UWND", 1.0, "m s-1"),
    "VWND": Field("VWND", 1.0, "m s-1"),
    "WWND": Field("WWND", 1.0, "hPa s-1", "DIFW"),
    "SPHU": Field("SPHU", 1.0, "kg kg-1"),
}

_REQUIRED = ("Times", "ZNU", "P_TOP", "P", "PB", "T", "QVAPOR")
#: The winds of each of ``arw2arl``'s variable lists: (U, V, W) and, for the
#: mass-coupled averages, the fields HYSPLIT decouples them with.
_WINDS = {
    "instantaneous": (("U", "V", "W"), ()),
    "averaged": (
        ("AVGFLX_RUM", "AVGFLX_RVM", "AVGFLX_WWM"),
        ("MUU", "MUV", "MAPFAC_M", "MAPFAC_U", "MAPFAC_V"),
    ),
}

#: The surface fields the averaged winds add (``arw2arl -c2``), by WRF name:
#: (ARL field, the staggered dimension to cut to the mass points).
AVERAGED_SURFACE_FIELDS: dict[str, tuple[Field, str | None]] = {
    "ZNT": (Field("RGHS", 1.0, "m"), None),
    "MUU": (Field("MUU0", 1.0, "Pa"), "west_east_stag"),
    "MUV": (Field("MUV0", 1.0, "Pa"), "south_north_stag"),
    "MAPFAC_M": (Field("MSFT", 1.0, "1"), None),
    "MAPFAC_U": (Field("MSFU", 1.0, "1"), "west_east_stag"),
    "MAPFAC_V": (Field("MSFV", 1.0, "1"), "south_north_stag"),
}
_DIMS = {"Time": "time", "bottom_top": "level", "south_north": "y", "west_east": "x"}


def _eqvlat(lat1: float, lat2: float) -> float:
    """Return the latitude of the cone tangent to two true latitudes (HYSPLIT's ``eqvlat.f``)."""
    s1, s2 = math.sin(math.radians(lat1)), math.sin(math.radians(lat2))
    if abs(s1 - s2) > 0.001:
        al1 = math.log((1.0 - s1) / (1.0 - s2))
        al2 = math.log((1.0 + s1) / (1.0 + s2))
    else:  # the true latitudes (nearly) equal
        tau = ((s1 - s2) / (2.0 - s1 - s2)) ** 2
        al1 = 2.0 / (2.0 - s1 - s2) * (1 + tau * (1 / 3 + tau * (1 / 5 + tau / 7)))
        tau = ((s1 - s2) / (2.0 + s1 + s2)) ** 2
        al2 = -2.0 / (2.0 + s1 + s2) * (1 + tau * (1 / 3 + tau * (1 / 5 + tau / 7)))
    return math.degrees(math.asin((al1 + al2) / (al1 - al2)))


def wrf_grid(attrs: Mapping[str, Any], nx: int, ny: int) -> Grid:
    """
    Describe a WRF domain as ``arw2arl``'s ``setmap.f`` does.

    The sync point is the domain's center (``CEN_LAT``, ``CEN_LON``) at grid
    point ``((nx + 1) / 2, (ny + 1) / 2)``, the spacing the mean of ``DX`` and
    ``DY``, and the projection ``MAP_PROJ``'s: 1 Lambert conformal (the cone
    tangent at ``eqvlat(TRUELAT1, TRUELAT2)``), 2 polar stereographic, 3
    Mercator.
    """

    def value(name: str) -> float:
        """Return the global attribute ``name`` as a float, or raise naming it."""
        if name not in attrs:
            raise ValueError(f"The WRF Dataset has no '{name}' attribute.")
        return float(attrs[name])

    map_proj = int(value("MAP_PROJ"))
    clat, clon = value("CEN_LAT"), value("CEN_LON")
    orient, tlat1, tlat2 = value("STAND_LON"), value("TRUELAT1"), value("TRUELAT2")
    grid_size = 0.5 * (value("DX") + value("DY")) / 1000.0
    if map_proj == 1:
        # As the index record stores it (7 characters), so this is the grid written.
        cone = float(format_fixed_width_float(_eqvlat(tlat1, tlat2), 7))
        pole_lat = cone
    elif map_proj == 2:
        if tlat1 != tlat2:
            raise ValueError(
                "A polar stereographic WRF grid needs TRUELAT1 equal to TRUELAT2, "
                f"got {tlat1} and {tlat2}."
            )
        pole_lat = cone = -90.0 if clat < 0.0 else 90.0
    elif map_proj == 3:
        if not -180.0 <= orient <= 180.0:
            orient = clon
        pole_lat, cone = (-90.0 if clat < 0.0 else 0.0), 0.0
    else:
        raise ValueError(
            f"WRF MAP_PROJ {map_proj} is not converted (1 Lambert conformal, "
            "2 polar stereographic, 3 Mercator)."
        )
    projection = Projection(
        pole_lat=pole_lat,
        pole_lon=orient,
        tangent_lat=tlat1,
        tangent_lon=orient,
        grid_size=grid_size,
        orientation=0.0,
        cone_angle=cone,
        sync_x=0.5 * (nx + 1),
        sync_y=0.5 * (ny + 1),
        sync_lat=clat,
        sync_lon=clon,
    )
    return Grid(projection=projection, nx=nx, ny=ny)


def _valid_times(ds: xr.Dataset) -> pd.DatetimeIndex:
    """Return the valid times of WRF's ``Times`` strings (``2013-06-30_01:00:00``)."""
    strings = [
        t.decode() if isinstance(t, bytes) else str(t)
        for t in np.atleast_1d(ds["Times"].values)
    ]
    return pd.to_datetime(strings, format="%Y-%m-%d_%H:%M:%S")


def _minutes(ds: xr.Dataset, times: pd.DatetimeIndex) -> np.ndarray:
    """Return the minutes since the simulation's start (WRF's ``XTIME``) at each time."""
    if "XTIME" in ds:
        xtime = np.atleast_1d(ds["XTIME"].values)
        if not np.issubdtype(xtime.dtype, np.datetime64):
            return xtime.astype(float)
    start = ds.attrs.get("START_DATE")
    if start is None:
        raise ValueError(
            "The WRF Dataset has neither a numeric 'XTIME' nor 'START_DATE'."
        )
    offset = times - pd.to_datetime(str(start), format="%Y-%m-%d_%H:%M:%S")
    return np.asarray(offset / pd.Timedelta(minutes=1), dtype=float)


def from_wrf(
    ds: xr.Dataset, *, winds: Literal["instantaneous", "averaged"] = "instantaneous"
) -> xr.Dataset:
    """
    Convert WRF-ARW output to a Dataset that :func:`arlmet.write_dataset` writes.

    This is HYSPLIT's ``arw2arl`` with its default variables (instantaneous
    winds) or, with ``winds="averaged"``, its mass-coupled time-averaged winds
    (``arw2arl -c2``), taking WRF's ``wrfout`` files as xarray opens them: one file, or
    several joined along ``Time`` (``xr.open_mfdataset(files,
    combine="nested", concat_dim="Time")``). The result has source ``AWRF``
    (which HYSPLIT and :func:`arlmet.sample_points` read as WRF), WRF's own
    grid and model levels, and the index records' forecast hour WRF's
    ``XTIME`` (minutes since the start of the simulation) in hours.

    ======================  ======  ==========================================
    WRF                     ARL     conversion
    ======================  ======  ==========================================
    ``HGT``                 SHGT    m
    ``PSFC``                PRSS    Pa to hPa
    ``RAINC`` + ``RAINNC``  TPP1    mm to m in the hour before, with ``DIFR``
    ``PBLH``                PBLH    m
    ``UST``                 USTR    m s-1
    ``SWDOWN``              DSWF    W m-2
    ``HFX``, ``LH``         SHTF,   W m-2, positive upward
                            LHTF
    ``T2``                  T02M    K
    ``U10``, ``V10``        U10M,   m s-1, along the grid's axes
                            V10M
    ``P`` + ``PB``          PRES    Pa to hPa
    ``T``                   TEMP    ``(T + 300) (P / 1000 hPa) ** (R / cp)``
    ``U``, ``V``            UWND,   m s-1, as staggered: the west and south
                            VWND    faces of each cell
    ``W``                   WWND    ``-W g P / (R TEMP (1 + 0.6077 QVAPOR))``
                                    in hPa s-1, at the bottom of each layer,
                                    with ``DIFW``
    ``QVAPOR``              SPHU    kg kg-1
    ======================  ======  ==========================================

    The levels are WRF's half levels (``ZNU``) with the surface first: a sigma
    axis (vertical flag 1) whose offset is the model top ``P_TOP``, or with
    WRF's hybrid coordinate (``HYBRID_OPT`` 2) a :class:`arlmet.WrfAxis`
    (flag 5). The precipitation of a time step is what fell since the one
    before, scaled to one hour; the first time step has none, so join every
    file of a run before converting.

    With ``winds="averaged"`` (a run with ``do_avgflx_em = 1``), ``UWND``,
    ``VWND`` and ``WWND`` are WRF's ``AVGFLX_RUM``, ``AVGFLX_RVM`` and
    ``AVGFLX_WWM`` as stored: averaged over the output interval and coupled to
    the column mass (no ``DIFW``, as ``arw2arl -c2`` writes them). The file
    also gets what HYSPLIT decouples them with: ``MUU0`` and ``MUV0`` (WRF's
    ``MUU``, ``MUV``), ``MSFT``, ``MSFU`` and ``MSFV`` (``MAPFAC_M``,
    ``MAPFAC_U``, ``MAPFAC_V``), all at the mass points' faces as stored,
    and ``RGHS`` (``ZNT``) where the run has it. HYSPLIT and
    :func:`arlmet.sample_points` use ``UWND * MSFU / MUU0``, ...;
    :func:`arlmet.open_dataset` returns the stored values.

    Parameters
    ----------
    ds : xarray.Dataset
        WRF output with its global attributes (``MAP_PROJ``, ``CEN_LAT``,
        ``DX``, ...) and at least ``Times``, ``ZNU``, ``P_TOP``, ``P``, ``PB``,
        ``T``, ``QVAPOR`` and the winds. Single-level variables not in it are
        left out.
    winds : {"instantaneous", "averaged"}, default "instantaneous"
        Which winds to write: WRF's ``U``, ``V`` and ``W`` at the output time,
        or the mass-coupled averages over the interval before it (above).

    Returns
    -------
    xarray.Dataset
        The fields in :func:`arlmet.open_dataset`'s layout (dims ``time``,
        ``level``, ``y``, ``x``), with each variable's ARL ``units`` and
        ``attrs["diff"]`` on ``TPP1`` and, for instantaneous winds, ``WWND``.
        Every model level is kept
        (``arw2arl`` keeps 50 unless told otherwise); ``ds.isel(level=...)``
        keeps fewer.

    Raises
    ------
    ValueError
        If a required variable or attribute is missing, ``MAP_PROJ`` is not
        1, 2 or 3, or ``winds`` is neither "instantaneous" nor "averaged".

    Notes
    -----
    ``arw2arl``'s TKE list is not converted. :func:`arlmet.open_dataset` and
    :func:`arlmet.sample_points` read the winds of the result where WRF put
    them (``AWRF``).

    Examples
    --------
    >>> import xarray as xr
    >>> import arlmet
    >>> wrf = xr.open_mfdataset(  # doctest: +SKIP
    ...     "wrfout_d01_2013-06-30_*", combine="nested", concat_dim="Time"
    ... )
    >>> ds = arlmet.converters.from_wrf(wrf)  # doctest: +SKIP
    >>> arlmet.write_dataset(ds, "wrf.arl")  # doctest: +SKIP
    """
    if not isinstance(ds, xr.Dataset):
        raise TypeError(f"from_wrf() takes an xarray.Dataset, got {type(ds).__name__}.")
    if winds not in _WINDS:
        raise ValueError(f"winds must be 'instantaneous' or 'averaged', got {winds!r}.")
    (u_name, v_name, w_name), couplers = _WINDS[winds]
    required = (*_REQUIRED, u_name, v_name, w_name, *couplers)
    missing = [name for name in required if name not in ds]
    if missing:
        raise ValueError(f"The WRF Dataset lacks {', '.join(missing)}.")
    if "Time" not in ds["T"].dims:
        ds = ds.expand_dims("Time")  # one time step, as a scalar
    nx, ny, nz = ds.sizes["west_east"], ds.sizes["south_north"], ds.sizes["bottom_top"]

    grid = wrf_grid(ds.attrs, nx, ny)
    first = {"Time": 0} if "Time" in ds["ZNU"].dims else {}
    levels = [1.0, *(float(v) for v in ds["ZNU"].isel(first).values)]
    top = float(np.atleast_1d(ds["P_TOP"].values)[0]) / 100.0
    hybrid = int(ds.attrs.get("HYBRID_OPT", 0)) == 2
    axis: VerticalAxis = (
        WrfAxis(levels=levels, offset=top)
        if hybrid
        else SigmaAxis(levels=levels, offset=top)
    )
    times = _valid_times(ds)
    minutes = _minutes(ds, times)
    hours = [int(round(m / 60.0)) for m in minutes]

    out = arl_dataset(grid, axis, times.values, source="AWRF", forecast_hours=hours)
    sizes_ok = {"bottom_top": nz, "south_north": ny, "west_east": nx}
    upper_dims = ("time", "level", "y", "x")
    surface_dims = ("time", "y", "x")

    def arl(da: xr.DataArray) -> xr.DataArray:
        """``da`` without its coordinates, on the ARL dimension names."""
        da = da.reset_coords(drop=True).drop_vars(list(da.coords), errors="ignore")
        return da.rename({k: v for k, v in _DIMS.items() if k in da.dims})

    for name, field in SURFACE_FIELDS.items():
        if name in ds:
            out[field.name] = field_variable(arl(ds[name]), field, surface_dims)
        if name == "PSFC" and "RAINC" in ds and "RAINNC" in ds:
            # Hourly totals from the run's accumulations, as arw2arl makes them.
            rain = arl(ds["RAINC"] + ds["RAINNC"]) * 0.001
            step = np.diff(minutes, prepend=np.nan)
            fell = xr.concat(
                [xr.zeros_like(rain.isel(time=[0])), rain.diff("time")], dim="time"
            )
            per_hour = xr.DataArray(
                np.where(np.isfinite(step), 60.0 / step, 0.0), dims="time"
            )
            out[PRECIPITATION.name] = field_variable(
                fell * per_hour, PRECIPITATION, surface_dims
            )

    pressure = ds["P"] + ds["PB"]  # Pa
    temperature = (ds["T"] + 300.0) * (pressure / 1.0e5) ** (R_DRY / CP)
    w = (
        ds[w_name]
        .isel(bottom_top_stag=slice(0, nz))
        .rename(bottom_top_stag="bottom_top")
    )
    if winds == "instantaneous":
        w = (
            -w
            * GRAVITY
            * pressure
            * 0.01
            / (R_DRY * temperature * (1.0 + 0.6077 * ds["QVAPOR"]))
        )
    u = ds[u_name].isel(west_east_stag=slice(0, nx)).rename(west_east_stag="west_east")
    v = (
        ds[v_name]
        .isel(south_north_stag=slice(0, ny))
        .rename(south_north_stag="south_north")
    )
    fields = {
        "PRES": pressure * 0.01,
        "TEMP": temperature,
        "UWND": u,
        "VWND": v,
        "WWND": w,
        "SPHU": ds["QVAPOR"],
    }
    for name, da in fields.items():
        bad = {d: s for d, s in sizes_ok.items() if d in da.dims and da.sizes[d] != s}
        if bad:
            raise ValueError(f"WRF's {name} has unexpected sizes {bad}.")
        field = LEVEL_FIELDS[name]
        if winds == "averaged" and name in {"UWND", "VWND", "WWND"}:
            # Mass-coupled, as stored (WRF's units).
            field = Field(name, 1.0, "Pa s-1" if name == "WWND" else "Pa m s-1")
        out[name] = field_variable(arl(da), field, upper_dims)
    if winds == "averaged":
        for name, (field, staggered) in AVERAGED_SURFACE_FIELDS.items():
            if name not in ds:
                continue
            da = ds[name]
            if staggered is not None:
                size = nx if staggered == "west_east_stag" else ny
                da = da.isel({staggered: slice(0, size)}).rename(
                    {staggered: staggered.removesuffix("_stag")}
                )
            out[field.name] = field_variable(arl(da), field, surface_dims)
    return out
