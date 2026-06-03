"""xarray-aware vertical coordinate helpers for ARL datasets."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import numpy as np
import xarray as xr

from arlmet.vertical import (
    HybridAxis,
    PressureAxis,
    SigmaAxis,
    TerrainAxis,
    hypsometric_z_agl,
)

__all__ = ["pressure", "z_agl", "z_msl"]


def _hypsometric_z_agl(
    p_levels: xr.DataArray,
    p_surface: xr.DataArray,
    temp: xr.DataArray,
) -> xr.DataArray:
    """
    Height AGL (m) at each level via the hypsometric equation.

    Thin xarray wrapper around :func:`arlmet.vertical.hypsometric_z_agl`.
    ``p_levels`` must be ordered from high to low pressure (surface to top).
    For flag=2, ``p_levels`` is 1D ``(level,)`` and is broadcast to match
    ``temp``. For flags 1/4, ``p_levels`` has the same shape as ``temp``.
    """
    if "level" not in temp.dims:
        raise ValueError("'TEMP' must have a 'level' dimension.")

    level_ax = list(temp.dims).index("level")
    z_vals = hypsometric_z_agl(
        p_levels.values,
        p_surface.values,
        temp.values,
        level_axis=level_ax,
    )

    return xr.DataArray(
        z_vals.astype(np.float32),
        dims=temp.dims,
        coords=temp.coords,
        attrs={
            "units": "m",
            "long_name": "height above ground level",
            "standard_name": "height",
        },
    )


def _sigma_hybrid_pressure(
    ds: xr.Dataset, axis: SigmaAxis | HybridAxis
) -> xr.DataArray:
    """Compute spatially varying pressure for sigma/hybrid axes."""
    if "PRSS" not in ds:
        raise ValueError(
            "Sigma/hybrid pressure conversion requires 'PRSS' (surface pressure) in dataset."
        )
    if "level" not in ds.coords:
        raise ValueError("Dataset has no 'level' coordinate.")

    prss = ds["PRSS"]
    level_ints = ds.coords["level"].values.astype(int).tolist()
    n_spatial = len(ds.arl.grid.dims)

    prss_vals = prss.values
    orig_shape = prss_vals.shape
    # to_pressure returns (..., all_levels) — select only the dataset's levels
    p_all = axis.to_pressure(surface_pressure=prss_vals.reshape(-1))
    p_flat = p_all[:, level_ints]  # (n, nlev_dataset)
    nlev = len(level_ints)

    p_arr = p_flat.reshape(*orig_shape, nlev)
    # Move the last (level) axis before the spatial dims
    level_target = len(orig_shape) - n_spatial
    p_arr = np.moveaxis(p_arr, -1, level_target)

    result_dims = (
        list(prss.dims[:level_target]) + ["level"] + list(prss.dims[level_target:])
    )
    coords: dict[Hashable, Any] = {
        d: ds.coords[d] for d in result_dims if d in ds.coords
    }
    for name in ("lat", "lon"):
        if name in ds.coords and name not in coords:
            coords[name] = ds.coords[name]

    return xr.DataArray(
        p_arr.astype(np.float32),
        dims=result_dims,
        coords=coords,
        attrs={
            "units": "hPa",
            "long_name": "air pressure",
            "standard_name": "air_pressure",
        },
    )


def pressure(ds: xr.Dataset) -> xr.DataArray:
    """
    Pressure (hPa) at each level.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset from :func:`arlmet.open_dataset`.

    Returns
    -------
    xr.DataArray
        - **flag=2**: the stored 1D ``pressure`` non-dim coord ``(level,)``.
        - **flag=1/4**: spatially varying array ``(time, level, y, x)`` computed
          from surface pressure (``PRSS``) via sigma-to-pressure conversion.
        - **flag=3**: returns the ``PRES`` data variable (3D pressure field in
          hPa) if present in the dataset. Terrain-following ARL files typically
          store this field. Raises ``ValueError`` if ``PRES`` is absent.
    """
    vaxis = ds.arl.vertical_axis

    if isinstance(vaxis, PressureAxis):
        if "pressure" not in ds.coords:
            raise ValueError("Dataset has no 'pressure' coordinate.")
        return ds.coords["pressure"]

    if isinstance(vaxis, (SigmaAxis, HybridAxis)):
        return _sigma_hybrid_pressure(ds, vaxis)

    if isinstance(vaxis, TerrainAxis):
        if "PRES" not in ds:
            raise ValueError(
                "pressure() for terrain-following (flag=3) files requires a 'PRES' "
                "(3D pressure) variable in the dataset."
            )
        return ds["PRES"]

    raise NotImplementedError(
        f"pressure() is not supported for {type(vaxis).__name__}."
    )


def z_agl(ds: xr.Dataset) -> xr.DataArray:
    """
    Height above ground level (m) at each level.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset from :func:`arlmet.open_dataset`.

    Returns
    -------
    xr.DataArray
        - **flag=2**: ``HGTS - SHGT`` (geopotential height minus terrain).
          Requires ``HGTS`` and ``SHGT`` in the dataset.
        - **flag=1/4**: hypsometric integration from surface pressure and
          temperature. Requires ``PRSS`` and ``TEMP``.
        - **flag=3**: the stored 1D ``height`` non-dim coord ``(level,)`` —
          terrain-following levels are heights AGL.
    """
    vaxis = ds.arl.vertical_axis

    if isinstance(vaxis, PressureAxis):
        for name in ("HGTS", "SHGT"):
            if name not in ds:
                raise ValueError(
                    f"z_agl() for pressure-level (flag=2) files requires '{name}' in dataset."
                )
        return ds["HGTS"] - ds["SHGT"]

    if isinstance(vaxis, (SigmaAxis, HybridAxis)):
        for name in ("PRSS", "TEMP"):
            if name not in ds:
                raise ValueError(
                    f"z_agl() for {vaxis.coord_system} (flag={vaxis.flag}) files "
                    f"requires '{name}' in dataset."
                )
        p_levels = pressure(ds)
        return _hypsometric_z_agl(p_levels, ds["PRSS"], ds["TEMP"])

    if isinstance(vaxis, TerrainAxis):
        if "height" not in ds.coords:
            raise ValueError(
                "Dataset has no 'height' coordinate and no 'HGTS' variable."
            )
        return ds.coords["height"]

    raise NotImplementedError(f"z_agl() is not supported for {type(vaxis).__name__}.")


def z_msl(ds: xr.Dataset) -> xr.DataArray:
    """
    Height above mean sea level (m) at each level.

    Requires ``SHGT`` (surface terrain height in meters) in the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset from :func:`arlmet.open_dataset`.

    Returns
    -------
    xr.DataArray
        ``z_agl(ds) + ds["SHGT"]``.
    """
    if "SHGT" not in ds:
        raise ValueError(
            "z_msl() requires 'SHGT' (terrain height in meters) in dataset."
        )
    result = z_agl(ds) + ds["SHGT"]
    result.attrs = {"units": "m", "long_name": "height above mean sea level"}
    return result
