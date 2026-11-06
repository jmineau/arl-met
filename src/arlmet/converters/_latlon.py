"""Describe a regular latitude/longitude grid as an ARL Grid."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from arlmet.grid import Grid, Projection

__all__ = ["latlon_grid"]


def _step(values: npt.NDArray[np.float64], name: str, *, modulo: float | None) -> float:
    """Return the constant spacing of ``values``, or raise if it is not constant."""
    if values.size < 2:
        raise ValueError(
            f"The grid needs at least two {name} values, got {values.size}."
        )
    offsets = values - values[0]
    if modulo is not None:
        offsets = np.unwrap(offsets, period=modulo)
    step = float(offsets[-1] / (values.size - 1))
    regular = offsets[0] + step * np.arange(values.size)
    if step <= 0.0 or not np.allclose(offsets, regular, rtol=0.0, atol=1e-3 * step):
        raise ValueError(
            f"{name.capitalize()} values must increase in equal steps; got "
            f"steps from {np.diff(offsets).min():g} to {np.diff(offsets).max():g}."
        )
    return step


def latlon_grid(lat: npt.ArrayLike, lon: npt.ArrayLike) -> Grid:
    """
    Build the ARL grid of cell centers ``lat`` (south to north) and ``lon``.

    The grid is written as HYSPLIT's ``xtrct_grid`` and :meth:`arlmet.Grid.subset`
    write a lat/lon grid: ``sync_lat``/``sync_lon`` the first cell with the
    longitude in 0-360 east, ``pole_lat``/``pole_lon`` the last cell, so
    ``pole_lon`` passes 360 when the columns cross Greenwich from the east.
    Longitudes may wrap (``[350, 355, 0, 5]``); they must not cover more than
    one turn.
    """
    lat_arr = np.asarray(lat, dtype=float)
    lon_arr = np.asarray(lon, dtype=float)
    dlat = _step(lat_arr, "latitude", modulo=None)
    dlon = _step(lon_arr, "longitude", modulo=360.0)
    nx, ny = lon_arr.size, lat_arr.size
    if nx * dlon > 360.0 * (1 + 1e-6):
        raise ValueError(
            f"{nx} longitudes {dlon:g} degrees apart cover more than 360 degrees."
        )
    sync_lat = float(lat_arr[0])
    sync_lon = float(lon_arr[0] % 360.0)
    projection = Projection(
        pole_lat=sync_lat + (ny - 1) * dlat,
        pole_lon=sync_lon + (nx - 1) * dlon,
        tangent_lat=dlat,
        tangent_lon=dlon,
        grid_size=0.0,
        orientation=0.0,
        cone_angle=0.0,
        sync_x=1.0,
        sync_y=1.0,
        sync_lat=sync_lat,
        sync_lon=sync_lon,
    )
    return Grid(projection=projection, nx=nx, ny=ny)
