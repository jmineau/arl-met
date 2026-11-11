"""
Helpers for the docs pages whose examples run when the docs are built.

arl-met has no sample ARL file: real ones are gigabytes, and an example never
downloads one. The examples run on small files of made-up values instead, which
:func:`sample_file` writes with ``arlmet.write_dataset``, the same call the user
guides teach.

A page starts with a block that is run but not shown (``:suppress:``)::

    import docs_examples

    docs_cwd = docs_examples.temp_workdir()
    docs_examples.sample_file("met.arl")

so ``"met.arl"`` in the examples below it is that file, in a folder of its own,
and ends with one that returns to the docs folder: ``os.chdir(docs_cwd)``.
"""

import atexit
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import arlmet
from arlmet._xarray._coords import grid_to_coord

#: The pressure levels of a sample file, in hPa (the surface is level 0).
PRESSURES = [850.0, 700.0, 500.0, 300.0]

#: The valid times of a sample file: four, six hours apart.
TIMES = pd.date_range("2024-07-18 00:00", periods=4, freq="6h")

_temporary_folders: list[Path] = []


def latlon_grid() -> arlmet.Grid:
    """Return a 0.1 degree grid, from 115 W to 109 W and from 38 N to 43 N."""
    projection = arlmet.Projection(
        pole_lat=90.0,
        pole_lon=0.0,
        tangent_lat=0.1,  # the spacing in latitude, for a lat-lon grid
        tangent_lon=0.1,  # and in longitude
        grid_size=0.0,  # 0 marks a lat-lon grid
        orientation=0.0,
        cone_angle=0.0,
        sync_x=1.0,  # grid point (1, 1) is the south-west corner,
        sync_y=1.0,
        sync_lat=38.0,  # at this latitude
        sync_lon=-115.0,  # and longitude
    )
    return arlmet.Grid(projection=projection, nx=61, ny=51)


def lambert_grid() -> arlmet.Grid:
    """Return a Lambert conformal grid of 20 km cells, 600 km on a side, over Utah."""
    projection = arlmet.Projection(
        pole_lat=90.0,
        pole_lon=0.0,
        tangent_lat=38.5,
        tangent_lon=-97.5,
        grid_size=20.0,  # km
        orientation=0.0,
        cone_angle=38.5,
        sync_x=1.0,
        sync_y=1.0,
        sync_lat=37.7,
        sync_lon=-115.1,
    )
    return arlmet.Grid(projection=projection, nx=30, ny=30)


def sample_file(
    path: str | os.PathLike[str],
    times: pd.DatetimeIndex = TIMES,
    *,
    upper: bool = True,
    projected: bool = False,
) -> Path:
    """
    Write a small ARL file of made-up values and return its path.

    The file is a July day over Utah on the pressure levels ``PRESSURES``, with
    the surface variables ``PRSS``, ``SHGT`` and ``T02M`` and, unless ``upper``
    is False, ``HGTS``, ``TEMP``, ``UWND``, ``VWND`` and ``WWND`` on each level.
    Its grid is :func:`latlon_grid`, or :func:`lambert_grid` with ``projected``.
    """
    grid = lambert_grid() if projected else latlon_grid()
    coords = grid.calculate_coords()
    lon, lat = coords["lon"][1], coords["lat"][1]
    if grid.is_latlon:  # 1-D: make them (y, x) like a projected grid's
        lon, lat = np.meshgrid(lon, lat)

    # Each field is made for the axes (time, level, y, x), with length 1 along an
    # axis it does not vary along, and then repeated to the full shape.
    hours = np.asarray((times - times[0]) / pd.Timedelta("1h"))[:, None, None, None]
    hpa = np.asarray(PRESSURES)[None, :, None, None]
    day = np.cos(2 * np.pi * (hours - 21) / 24)  # warmest at 21 UTC
    wave = np.sin(np.deg2rad(4 * lon + 6 * lat))

    terrain = 1300 + 150 * np.exp(-(((lon + 111.3) / 0.5) ** 2))  # m, with a ridge
    height = 44330.8 * (1 - (hpa / 1013.25) ** 0.190263) + 5 * hours  # m
    surface = {
        "PRSS": 1013.25 * (1 - terrain / 44330.8) ** 5.255,  # hPa, at the terrain
        "SHGT": terrain,
        "T02M": 298 - 0.0065 * terrain + 8 * day,  # K
    }
    aloft = {}
    if upper:
        jet = (850 - hpa) / 50  # stronger westerlies higher up
        swing = np.cos(2 * np.pi * (hours - 6) / 24)  # winds that change with time
        aloft = {
            "HGTS": height,
            "TEMP": 296 - 0.0065 * height + 0.5 * (40 - lat) + day,  # K
            "UWND": 5 + 3 * wave + 2 * swing + jet,  # m/s
            "VWND": -2 + 2 * np.cos(np.deg2rad(5 * lat)) + swing,  # m/s
            "WWND": 0.01 * wave + 0 * hpa,  # hPa/s
        }

    nt, nz = len(times), len(PRESSURES)
    ds = xr.Dataset(
        coords={**coords, "time": times, "arl_grid": grid_to_coord(grid)},
        attrs={"source": "DOCS"},
    )
    for name, field in surface.items():
        full = np.broadcast_to(field, (nt, 1, grid.ny, grid.nx))[:, 0]
        ds[name] = (("time", *grid.dims), full.astype("float32"))
    if upper:
        ds = ds.assign_coords(level=np.arange(1, nz + 1))
    for name, field in aloft.items():
        full = np.broadcast_to(field, (nt, nz, grid.ny, grid.nx))
        ds[name] = (("time", "level", *grid.dims), full.astype("float32"))

    axis = arlmet.PressureAxis(levels=[0.0, *PRESSURES])
    arlmet.write_dataset(ds, path, vertical_axis=axis)
    return Path(path)


def era5_sample() -> tuple[xr.Dataset, xr.Dataset]:
    """
    Return made-up ERA5 pressure-level and single-level Datasets.

    They are named and laid out as xarray opens a Climate Data Store netCDF
    download: ``valid_time``, ``pressure_level`` and ``latitude`` from north to
    south, values in ERA5's units (geopotential, Pa, J m-2 in the hour). Three
    hours of a July morning over Utah on a 0.25 degree grid.
    """
    gravity = 9.80665
    times = pd.date_range("2024-07-18 15:00", periods=3, freq="1h")
    lat = np.arange(44.0, 37.99, -0.25)
    lon = np.arange(-116.0, -109.99, 0.25)
    levels = np.array([1000.0, 850.0, 700.0, 500.0])
    hours = np.arange(len(times))[:, None, None]
    lon2, lat2 = np.meshgrid(lon, lat)
    wave = np.sin(np.deg2rad(4 * lon2 + 6 * lat2))
    terrain = 1300 + 150 * np.exp(-(((lon2 + 111.3) / 0.5) ** 2))

    hpa = levels[None, :, None, None]
    height = 44330.8 * (1 - (hpa / 1013.25) ** 0.190263) + 0 * wave
    upper = {
        "z": gravity * height,
        "t": 296 - 0.0065 * height + 0.5 * (40 - lat2),
        "u": 5 + 3 * wave + (850 - hpa) / 50,
        "v": -2 + 2 * np.cos(np.deg2rad(5 * lat2)) + 0 * hpa,
        "w": 1.0 * wave + 0 * hpa,  # Pa s-1
        "r": 40 + 20 * wave + 0 * hpa,
    }
    sunshine = 600 + 100 * hours  # W m-2
    surface = {
        "sp": 101325 * (1 - terrain / 44330.8) ** 5.255 + 0 * hours,
        "z": gravity * terrain + 0 * hours,
        "t2m": 298 - 0.0065 * terrain + 2 * hours,
        "u10": 2 + wave + 0 * hours,
        "v10": -1 + 0 * wave + 0 * hours,
        "tp": 0.0002 * np.clip(wave, 0, None) + 0 * hours,  # m in the hour
        "sshf": -3600 * 0.3 * sunshine * (1 + 0.1 * wave),  # downward positive
        "ssrd": 3600 * sunshine * (1 + 0 * wave),
    }
    coords = {"valid_time": times, "latitude": lat, "longitude": lon}
    shape = (len(times), len(levels), lat.size, lon.size)
    pressure_levels = xr.Dataset(
        {
            name: (
                ("valid_time", "pressure_level", "latitude", "longitude"),
                np.broadcast_to(field, shape).astype("float32"),
            )
            for name, field in upper.items()
        },
        coords={**coords, "pressure_level": levels},
    )
    single_levels = xr.Dataset(
        {
            name: (
                ("valid_time", "latitude", "longitude"),
                np.broadcast_to(field, (len(times), lat.size, lon.size)).astype(
                    "float32"
                ),
            )
            for name, field in surface.items()
        },
        coords=coords,
    )
    return pressure_levels, single_levels


def era5_model_level_sample() -> xr.Dataset:
    """
    Return made-up ERA5 fields on five model levels (100 to 137), on :func:`era5_sample`'s grid.

    Laid out as xarray opens a model-level netCDF download from the Climate
    Data Store: ``valid_time``, ``model_level`` (1 at the top, 137 at the
    ground) and ``latitude`` from north to south, values in ERA5's units.
    """
    pressure_levels, _ = era5_sample()
    levels = np.array([100, 110, 120, 130, 137])
    # Roughly the pressures of those levels over the sample's terrain.
    hpa = np.array([585.0, 710.0, 820.0, 900.0, 870.0])[None, :, None, None]
    sample = pressure_levels.isel(pressure_level=0, drop=True)
    height = 44330.8 * (1 - (hpa / 1013.25) ** 0.190263)
    lat = sample["latitude"].values[None, None, :, None]
    shape = (sample.sizes["valid_time"], len(levels), *sample["t"].shape[1:])
    fields = {
        "t": 296 - 0.0065 * height + 0.5 * (40 - lat),
        "u": 5 + (850 - hpa) / 50 + 0 * lat,
        "v": -2 + 0 * hpa + 0 * lat,
        "w": 0.5 * np.cos(np.deg2rad(6 * lat)) + 0 * hpa,  # Pa s-1
        "q": 0.008 * hpa / 1000 + 0 * lat,  # kg kg-1
    }
    dims = ("valid_time", "model_level", "latitude", "longitude")
    return xr.Dataset(
        {
            name: (dims, np.broadcast_to(field, shape).astype("float32"))
            for name, field in fields.items()
        },
        coords={
            "valid_time": sample["valid_time"].values,
            "model_level": levels,
            "latitude": sample["latitude"].values,
            "longitude": sample["longitude"].values,
        },
    )


def gfs_sample() -> list[xr.Dataset]:
    """
    Return made-up GDAS Datasets, laid out as ``cfgrib.open_datasets`` returns them.

    They stand in for two GRIB2 files valid at 2025-09-01 00 UTC: the 00 UTC
    analysis (``step`` 0) and the 18 UTC cycle's 6-hour forecast, which adds
    the accumulated precipitation and the averaged sensible heat flux. One
    Dataset per kind of level, rows from north to south, on a 1 degree grid
    over the western United States.
    """
    valid = pd.Timestamp("2025-09-01 00:00")
    lat = np.arange(50.0, 29.99, -1.0)
    lon = np.arange(235.0, 259.01, 1.0)
    levels = np.array([1000.0, 850.0, 700.0, 500.0])
    lon2, lat2 = np.meshgrid(lon, lat)
    wave = np.sin(np.deg2rad(8 * lon2 + 10 * lat2))
    terrain = 1300 + 150 * np.exp(-(((lon2 - 248.7) / 2.0) ** 2))
    hpa = levels[:, None, None]
    height = 44330.8 * (1 - (hpa / 1013.25) ** 0.190263) + 0 * wave
    flat = ("latitude", "longitude")
    upper = ("isobaricInhPa", *flat)

    def cube(step, variables, type_of_level, level=0.0, step_type="instant"):
        attrs = {"GRIB_typeOfLevel": type_of_level, "GRIB_stepType": step_type}
        step = pd.Timedelta(hours=step)
        coords = {"time": valid - step, "step": step, "valid_time": valid,
                  "latitude": lat, "longitude": lon, type_of_level: level}  # fmt: skip
        return xr.Dataset(
            {
                name: (dims, np.asarray(values, dtype="float32"), attrs)
                for name, (dims, values) in variables.items()
            },
            coords=coords,
        )

    def fields(step):
        winds = 5 + 3 * wave + (850 - hpa) / 50 + step
        return [
            cube(
                step,
                {
                    "gh": (upper, height),
                    "t": (upper, 296 - 0.0065 * height + 0.5 * (40 - lat2)),
                    "u": (upper, winds),
                    "v": (upper, -2 + 2 * np.cos(np.deg2rad(5 * lat2)) + 0 * hpa),
                    "w": (upper, 0.5 * wave + 0 * hpa),  # Pa s-1
                    "r": (upper, 40 + 20 * wave + 0 * hpa),
                },
                "isobaricInhPa",
                levels,
            ),
            cube(
                step,
                {
                    "sp": (flat, 101325 * (1 - terrain / 44330.8) ** 5.255),
                    "orog": (flat, terrain),
                },
                "surface",
            ),
            cube(
                step, {"t2m": (flat, 298 - 0.0065 * terrain)}, "heightAboveGround", 2.0
            ),
        ]

    forecast = [
        *fields(6),
        cube(
            6, {"tp": (flat, 2 * np.clip(wave, 0, None))}, "surface", step_type="accum"
        ),
        cube(6, {"avg_ishf": (flat, 40 + 20 * wave)}, "surface", step_type="avg"),
    ]
    return [*forecast, *fields(0)]


def icon_sample() -> list[xr.Dataset]:
    """
    Return made-up ICON-EU Datasets, laid out as ``cfgrib.open_datasets`` returns them.

    They stand in for DWD's files of the 2026-11-06 00 UTC forecast, steps 0
    to 3: one Dataset per variable, pressure level and step, rows from south
    to north on a 0.5 degree grid over central Europe, with the terrain height
    in one time-invariant file and the precipitation and latent heat flux as
    a total and a mean from the start of the forecast.
    """
    start = pd.Timestamp("2026-11-06 00:00")
    lat = np.arange(44.0, 56.01, 0.5)
    lon = np.arange(0.0, 20.01, 0.5)
    levels = [1000.0, 850.0, 700.0, 500.0]
    lon2, lat2 = np.meshgrid(lon, lat)
    wave = np.sin(np.deg2rad(12 * lon2 + 15 * lat2))
    terrain = 200 + 1800 * np.exp(-(((lat2 - 47.0) / 1.0) ** 2)) * (lon2 > 6)
    rain = np.clip(wave, 0, None)  # mm in each hour
    flat = ("latitude", "longitude")

    def cube(name, values, step, type_of_level, level=0.0, step_type="instant"):
        step = pd.Timedelta(hours=step)
        attrs = {"GRIB_typeOfLevel": type_of_level, "GRIB_stepType": step_type}
        coords = {"time": start, "step": step, "valid_time": start + step,
                  "latitude": lat, "longitude": lon, type_of_level: level}  # fmt: skip
        data = (flat, np.asarray(values, dtype="float32"), attrs)
        return xr.Dataset({name: data}, coords=coords)

    datasets = [cube("HSURF", terrain, 0, "surface")]
    for step in range(4):
        for p in levels:
            height = 44330.8 * (1 - (p / 1013.25) ** 0.190263) + 20 * wave
            for name, values in {
                "z": 9.80665 * height,
                "t": 284 - 0.0065 * height + 0.3 * (50 - lat2),
                "u": 8 + 4 * wave + (850 - p) / 40,
                "v": -1 + 3 * np.cos(np.deg2rad(6 * lon2)),
                "w": 0.3 * wave,  # Pa s-1
                "r": 60 + 25 * wave,
            }.items():
                datasets.append(cube(name, values, step, "isobaricInhPa", p))
        datasets += [
            cube("sp", 101325 * (1 - terrain / 44330.8) ** 5.255, step, "surface"),
            cube("t2m", 283 - 0.0065 * terrain, step, "heightAboveGround", 2.0),
            cube("tp", step * rain, step, "surface", step_type="accum"),
            # Positive downward, as ICON's fluxes: evaporation is negative.
            cube("avg_slhtf", -40 - 20 * wave, step, "surface", step_type="avg"),
        ]
    return datasets


def wrf_sample() -> xr.Dataset:
    """
    Return a made-up WRF run, laid out as xarray opens ``wrfout`` files.

    Three hours on a 12 km Lambert conformal grid over Utah with four model
    levels, WRF's staggered dimensions and its global map attributes, with
    both the instantaneous and the mass-coupled averaged winds.
    """
    nx, ny, nt = 30, 25, 3
    start = pd.Timestamp("2013-06-29 18:00")
    minutes = np.array([360.0, 420.0, 480.0])
    times = [
        (start + pd.Timedelta(minutes=m)).strftime("%Y-%m-%d_%H:%M:%S").encode()
        for m in minutes
    ]
    y, x = np.meshgrid(np.linspace(0, 3, ny), np.linspace(0, 4, nx), indexing="ij")
    wave = np.sin(x + y)
    hours = np.arange(nt)[:, None, None, None]
    eta = np.array([0.99, 0.95, 0.85, 0.7])
    base = (101325 * eta)[None, :, None, None] + 0 * wave + 0 * hours  # Pa

    def stag(field, axis):
        """Add one more point along ``axis``, as WRF staggers a field."""
        return np.concatenate([field, field.take([-1], axis=axis)], axis=axis)

    winds = 5 + 3 * wave + 2 * hours + 0 * base
    mass = ("Time", "bottom_top", "south_north", "west_east")
    flat = ("Time", "south_north", "west_east")
    rain = np.cumsum(np.clip(wave, 0, None)[None] * np.ones((nt, 1, 1)), axis=0)
    data = {
        "Times": (("Time",), np.array(times, dtype="S19")),
        "XTIME": (("Time",), minutes),
        "ZNU": (("Time", "bottom_top"), np.tile(eta, (nt, 1))),
        "P_TOP": (("Time",), np.full(nt, 5000.0)),
        "P": (mass, 50 * wave + 0 * base),
        "PB": (mass, base),
        "T": (mass, 10 * (1 - eta)[None, :, None, None] + wave + 0 * hours),
        "QVAPOR": (mass, 0.005 * eta[None, :, None, None] + 0 * winds),
        "U": (("Time", "bottom_top", "south_north", "west_east_stag"), stag(winds, 3)),
        "V": (("Time", "bottom_top", "south_north_stag", "west_east"), stag(-0.5 * winds, 2)),
        "W": (("Time", "bottom_top_stag", "south_north", "west_east"), stag(0.1 * wave + 0 * base, 1)),
        "HGT": (flat, 1300 + 300 * wave[None] + 0 * hours[:, 0]),
        "PSFC": (flat, 87000 - 3000 * wave[None] + 0 * hours[:, 0]),
        "RAINC": (flat, 0.2 * rain),
        "RAINNC": (flat, 0.8 * rain),
        "T2": (flat, 295 - 2 * wave[None] + hours[:, 0]),
        # The mass-coupled averages (do_avgflx_em = 1): the winds times the column mass.
        "AVGFLX_RUM": (("Time", "bottom_top", "south_north", "west_east_stag"), stag(8.5e4 * winds, 3)),
        "AVGFLX_RVM": (("Time", "bottom_top", "south_north_stag", "west_east"), stag(-4.2e4 * winds, 2)),
        "AVGFLX_WWM": (("Time", "bottom_top_stag", "south_north", "west_east"), stag(8.5e3 * wave + 0 * base, 1)),
        "MUU": (("Time", "south_north", "west_east_stag"), stag(85000 + 0 * hours[:, 0] + 0 * wave[None], 2)),
        "MUV": (("Time", "south_north_stag", "west_east"), stag(85000 + 0 * hours[:, 0] + 0 * wave[None], 1)),
        "MAPFAC_M": (flat, 1.0 + 0 * hours[:, 0] + 0 * wave[None]),
        "MAPFAC_U": (("Time", "south_north", "west_east_stag"), stag(1.0 + 0 * hours[:, 0] + 0 * wave[None], 2)),
        "MAPFAC_V": (("Time", "south_north_stag", "west_east"), stag(1.0 + 0 * hours[:, 0] + 0 * wave[None], 1)),
    }  # fmt: skip
    return xr.Dataset(
        {name: (dims, np.asarray(values, dtype=np.float32) if values.dtype.kind == "f" else values)
         for name, (dims, values) in data.items()},
        attrs={
            "MAP_PROJ": 1, "CEN_LAT": 40.0, "CEN_LON": -111.5, "STAND_LON": -111.5,
            "TRUELAT1": 30.0, "TRUELAT2": 60.0, "DX": 12000.0, "DY": 12000.0,
            "HYBRID_OPT": 0, "START_DATE": "2013-06-29_18:00:00",
        },
    )  # fmt: skip


def temp_workdir() -> str:
    """Change to a new temporary folder and return the folder you left."""
    previous = os.getcwd()
    work = Path(tempfile.mkdtemp())
    _temporary_folders.append(work)
    os.chdir(work)
    return previous


@atexit.register
def _remove_temporary_folders() -> None:
    os.chdir(Path(__file__).resolve().parents[1])
    for folder in _temporary_folders:
        shutil.rmtree(folder, ignore_errors=True)
