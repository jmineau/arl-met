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
    WrfAxis,
    _hysplit_z_agl,
)

__all__ = ["pressure", "z_agl", "z_msl"]


def _sigma_hybrid_pressure(
    ds: xr.Dataset, axis: SigmaAxis | HybridAxis
) -> xr.DataArray:
    """Compute spatially varying nominal pressure for sigma/hybrid axes."""
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


def _hysplit_z_agl_da(
    ds: xr.Dataset, axis: SigmaAxis | HybridAxis | WrfAxis
) -> xr.DataArray:
    """
    Height AGL (m) of each level of a sigma, hybrid or WRF dataset, as HYSPLIT finds it.

    HYSPLIT's ``prfsig`` (flag 1, and flag 5 without WRF's hybrid
    coefficients) and ``prfecm`` (flag 4): from the ground
    (``PRSS``, ``T02M`` or ``TMPS``) through :func:`pressure` with the virtual
    temperature from ``TEMP`` and ``SPHU`` (or ``RELH``), each when present.
    """
    for name in ("PRSS", "TEMP"):
        if name not in ds:
            raise ValueError(
                f"z_agl() for {axis.coord_system} (flag={axis.flag}) files "
                f"requires '{name}' in dataset."
            )
    temp = ds["TEMP"]
    if "level" not in temp.dims:
        raise ValueError("'TEMP' must have a 'level' dimension.")
    level_ax = list(temp.dims).index("level")
    p = pressure(ds).transpose(*temp.dims).values
    prss = ds["PRSS"].values
    if isinstance(axis, (SigmaAxis, WrfAxis)):
        first_sigma = axis.levels[int(temp["level"].values[0])]
    else:
        first_sigma = np.take(p, 0, axis=level_ax) / prss
    surface_t = next((ds[n].values for n in ("T02M", "TMPS") if n in ds), None)

    def _upper(name: str) -> Any:
        """*name* on TEMP's dims, or None when the dataset lacks it."""
        return ds[name].transpose(*temp.dims).values if name in ds else None

    z = _hysplit_z_agl(
        p,
        prss,
        temp.values,
        first_sigma=first_sigma,
        surface_temperature=surface_t,
        specific_humidity=_upper("SPHU"),
        relative_humidity=_upper("RELH"),
        level_axis=level_ax,
    )
    return xr.DataArray(
        z.astype(np.float32),
        dims=temp.dims,
        coords=temp.coords,
        attrs={
            "units": "m",
            "long_name": "height above ground level",
            "standard_name": "height",
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
        - **flag=1** (sigma): the ``PRES`` data variable when the dataset has
          it, as HYSPLIT's ``prfsig`` uses it; otherwise, as for flag 4, a
          spatially varying array ``(time, level, y, x)`` computed from surface
          pressure (``PRSS``) via sigma-to-pressure conversion.
        - **flag=4** (hybrid): computed from ``PRSS`` and the stored levels;
          ``PRES`` is not used, as in HYSPLIT's ``prfecm``.
        - **flag=3**: returns the ``PRES`` data variable (3D pressure field in
          hPa) if present in the dataset. Terrain-following ARL files typically
          store this field. Raises ``ValueError`` if ``PRES`` is absent.
        - **flag=5** (WRF): returns the ``PRES`` data variable, as HYSPLIT's
          ``prfwrf`` does; WRF's hybrid levels need coefficients the file does
          not store. Raises ``ValueError`` if ``PRES`` is absent.
    """
    vaxis = ds.arl.vertical_axis

    if isinstance(vaxis, PressureAxis):
        if "pressure" not in ds.coords:
            raise ValueError("Dataset has no 'pressure' coordinate.")
        return ds.coords["pressure"]

    if isinstance(vaxis, SigmaAxis) and "PRES" in ds:
        return ds["PRES"]
    if isinstance(vaxis, (SigmaAxis, HybridAxis)):
        return _sigma_hybrid_pressure(ds, vaxis)

    if isinstance(vaxis, TerrainAxis):
        if "PRES" not in ds:
            raise ValueError(
                "pressure() for terrain-following (flag=3) files requires a 'PRES' "
                "(3D pressure) variable in the dataset."
            )
        return ds["PRES"]

    if isinstance(vaxis, WrfAxis):
        if "PRES" not in ds:
            raise ValueError(
                "pressure() for WRF (flag=5) files requires a 'PRES' (3D pressure) "
                "variable in the dataset."
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
        - **flag=1/4**: hypsometric integration from the ground through
          :func:`pressure` with the virtual temperature, as HYSPLIT's
          ``prfsig`` and ``prfecm`` do: the ground is at ``PRSS`` and ``T02M``
          (else ``TMPS``, else the first level's ``TEMP`` brought down
          adiabatically), and the virtual temperature is ``TEMP (1 + 0.61 q)``
          with ``q`` from ``SPHU``, else ``RELH``, else 0. Requires ``PRSS``
          and ``TEMP``.
        - **flag=3**: the stored 1D ``height`` non-dim coord ``(level,)`` —
          terrain-following levels are heights AGL.
        - **flag=5** (WRF): as flag 1 with ``PRES``, as HYSPLIT's ``prfsig``
          does for a WRF file without WRF's hybrid coefficients; the ground
          temperature without ``T02M`` is the first level's brought down
          adiabatically from its eta. Requires ``PRSS``, ``TEMP`` and
          ``PRES``.
    """
    vaxis = ds.arl.vertical_axis

    if isinstance(vaxis, PressureAxis):
        for name in ("HGTS", "SHGT"):
            if name not in ds:
                raise ValueError(
                    f"z_agl() for pressure-level (flag=2) files requires '{name}' in dataset."
                )
        return ds["HGTS"] - ds["SHGT"]

    if isinstance(vaxis, (SigmaAxis, HybridAxis, WrfAxis)):
        return _hysplit_z_agl_da(ds, vaxis)

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
