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
from arlmet.xarray._coords import grid_to_coord

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
