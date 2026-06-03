"""Encode and decode ARL grid and vertical axis as xarray coordinates."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from arlmet.grid import Grid, Projection
from arlmet.vertical import VerticalAxis

PROJECTION_ATTRS = (
    "pole_lat",
    "pole_lon",
    "tangent_lat",
    "tangent_lon",
    "grid_size",
    "orientation",
    "cone_angle",
    "sync_x",
    "sync_y",
    "sync_lat",
    "sync_lon",
)

# (coord_name, CF standard_name, units) by vertical flag
_PHYS_COORD: dict[int, tuple[str, str, str]] = {
    1: ("sigma", "atmosphere_sigma_coordinate", "1"),
    2: ("pressure", "air_pressure", "hPa"),
    3: ("height", "height", "m"),
    4: ("sigma", "atmosphere_hybrid_sigma_pressure_coordinate", "1"),
}


_CF_SPATIAL: dict[str, dict[str, str]] = {
    "lon": {"standard_name": "longitude", "units": "degrees_east", "axis": "X"},
    "lat": {"standard_name": "latitude", "units": "degrees_north", "axis": "Y"},
    "x": {"standard_name": "projection_x_coordinate", "units": "m", "axis": "X"},
    "y": {"standard_name": "projection_y_coordinate", "units": "m", "axis": "Y"},
}
# 2D auxiliary coords don't get 'axis' per CF conventions
_CF_SPATIAL_AUX: dict[str, dict[str, str]] = {
    "lon": {"standard_name": "longitude", "units": "degrees_east"},
    "lat": {"standard_name": "latitude", "units": "degrees_north"},
}


def add_cf_spatial_attrs(ds: xr.Dataset) -> xr.Dataset:
    """Add CF convention attrs to spatial coordinates of a Dataset."""
    updates: dict[str, xr.Variable] = {}
    for name in ("lon", "lat", "x", "y"):
        if name not in ds.coords:
            continue
        cvar = ds.coords[name].variable
        # lon/lat are aux (2D) when x/y also present
        is_aux = name in ("lon", "lat") and ("x" in ds.coords or "y" in ds.coords)
        base_attrs = _CF_SPATIAL_AUX[name] if is_aux else _CF_SPATIAL[name]
        merged = {**base_attrs, **cvar.attrs}
        updates[name] = xr.Variable(cvar.dims, cvar.data, attrs=merged)
    if updates:
        ds = ds.assign_coords(updates)
    return ds


def grid_to_coord(grid: Grid) -> xr.Variable:
    """Build a scalar ``arl_grid`` coordinate encoding ARL projection parameters."""
    proj = grid.projection
    attrs: dict[str, Any] = {"nx": grid.nx, "ny": grid.ny}
    attrs.update({name: getattr(proj, name) for name in PROJECTION_ATTRS})
    return xr.Variable((), 0, attrs=attrs)


def grid_from_coord(coord: xr.DataArray | xr.Variable) -> Grid:
    """Reconstruct a Grid from an ``arl_grid`` coordinate."""
    attrs = coord.attrs
    try:
        projection = Projection(
            **{name: float(attrs[name]) for name in PROJECTION_ATTRS}
        )
        nx = int(attrs["nx"])
        ny = int(attrs["ny"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Cannot reconstruct Grid: 'arl_grid' coordinate is missing expected attributes."
        ) from exc
    return Grid(projection=projection, nx=nx, ny=ny)


def physical_coord_for_vaxis(
    vaxis: VerticalAxis,
    level_indices: list[int],
) -> tuple[str, xr.Variable]:
    """
    Build the physical non-dimension coordinate for a Dataset field's level axis.

    Returns ``(coord_name, variable)`` where the variable has dims ``("level",)``
    and attrs encoding the coordinate system and surface reference value.
    The surface value (index 0) is stored in ``attrs["surface"]``; the coord
    values are the physical values at each requested (non-zero) level index.
    """
    if vaxis.flag not in _PHYS_COORD:
        raise NotImplementedError(
            f"Physical coordinate for vertical_flag={vaxis.flag} is not implemented."
        )
    coord_name, standard_name, units = _PHYS_COORD[vaxis.flag]
    all_levels = vaxis.levels
    surface_val = float(all_levels[0])
    values = np.asarray([all_levels[i] for i in level_indices], dtype=float)
    attrs: dict[str, Any] = {
        "standard_name": standard_name,
        "units": units,
        "surface": surface_val,
        "offset": vaxis.offset,
    }
    return coord_name, xr.Variable(("level",), values, attrs=attrs)


def vaxis_from_coord(
    flag: int,
    level_coord: xr.DataArray | xr.Variable,
    phys_coord: xr.DataArray | xr.Variable,
) -> VerticalAxis:
    """
    Reconstruct a VerticalAxis from integer level coord and sparse physical coord.

    Used for the flat Dataset (``open_dataset``) case where the physical coord
    has dims ``("level",)`` and carries ``attrs["surface"]`` (the level-0 value).
    Missing intermediate levels are filled as 0.0.
    """
    surface = float(phys_coord.attrs["surface"])
    offset = float(phys_coord.attrs.get("offset", 0.0))
    # level coord is 1-N; the full levels array has surface at index 0
    level_ints = np.asarray(level_coord.values, dtype=int).tolist()
    phys_values = np.asarray(phys_coord.values, dtype=float).tolist()
    n_levels = max(level_ints) + 1  # +1 because indices are 1-based, plus surface at 0
    levels: list[float] = [0.0] * n_levels
    levels[0] = surface
    for idx, phys in zip(level_ints, phys_values, strict=True):
        levels[idx] = phys
    return VerticalAxis.from_flag(flag, levels=levels, offset=offset)


def _extract_dataset_vertical_axis(
    ds: xr.Dataset,
    explicit_vertical_axis: VerticalAxis | None = None,
) -> VerticalAxis:
    """Resolve the file-wide vertical axis for Dataset writes."""
    if explicit_vertical_axis is not None:
        if not isinstance(explicit_vertical_axis, VerticalAxis):
            raise TypeError("vertical_axis must be a VerticalAxis instance.")
        return explicit_vertical_axis

    if "level" not in ds.coords:
        raise ValueError(
            "Surface-only datasets require an explicit vertical_axis= for write_dataset()."
        )

    flag = ds.attrs.get("vertical_flag")
    if flag is None:
        raise ValueError(
            "Dataset is missing attrs['vertical_flag']; pass vertical_axis= to write_dataset()."
        )

    level_coord = ds.coords["level"]
    for _coord_name, coord in ds.coords.items():
        if coord.dims == ("level",) and "surface" in coord.attrs:
            return vaxis_from_coord(int(flag), level_coord, coord)

    raise ValueError(
        "Dataset is missing a physical level coordinate with attrs['surface']; "
        "open files with arlmet.open_dataset() or pass vertical_axis= to write_dataset()."
    )


def _extract_dataset_forecast_hours(
    ds: xr.Dataset, times: pd.DatetimeIndex
) -> list[int]:
    """Return one index-record forecast hour per dataset time step."""
    if "forecast_hour" not in ds.data_vars:
        return [0] * len(times)

    forecast = ds["forecast_hour"]
    if forecast.dims != ("time",):
        raise ValueError("Dataset 'forecast_hour' must have dims ('time',).")

    coord_times = pd.to_datetime(forecast.coords["time"].values)
    if not coord_times.equals(times):
        raise ValueError(
            "Dataset 'forecast_hour' must align exactly with the dataset time coordinate."
        )

    return [int(value) for value in forecast.values.tolist()]
