"""xarray-aware vertical coordinate helpers for ARL datasets."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import numpy as np
import xarray as xr

from arlmet._ops.sample.fields import temperature_from_theta
from arlmet.vertical import (
    HybridAxis,
    PressureAxis,
    SigmaAxis,
    TerrainAxis,
    WrfAxis,
    _hysplit_z_agl,
    _terrain_following_z_agl,
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
    n_spatial = len(ds.arl.grid.dims)

    prss_vals = prss.values
    orig_shape = prss_vals.shape
    # The Dataset's axis is the surface, then its own levels in order (whatever
    # their numbers in the file), so its levels are axis levels 1 to N.
    p_flat = axis.to_pressure(surface_pressure=prss_vals.reshape(-1))[:, 1:]
    nlev = p_flat.shape[1]

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
    temperature from ``TEMP`` (``THET * (PRES / 1000) ** 0.286`` where the
    dataset has no ``TEMP``) and ``SPHU`` (or ``RELH``), each when present.
    """
    if "PRSS" not in ds:
        raise ValueError(
            f"z_agl() for {axis.coord_system} (flag={axis.flag}) files "
            "requires 'PRSS' in dataset."
        )
    if "TEMP" in ds:
        temp = ds["TEMP"]
    elif "THET" in ds and "PRES" in ds:  # as sample_points reads such a file
        temp = temperature_from_theta(ds["THET"], ds["PRES"])
    else:
        raise ValueError(
            f"z_agl() for {axis.coord_system} (flag={axis.flag}) files "
            "requires 'TEMP' (or 'THET' and 'PRES') in dataset."
        )
    if "level" not in temp.dims:
        raise ValueError("'TEMP' must have a 'level' dimension.")
    level_ax = list(temp.dims).index("level")
    p = pressure(ds).transpose(*temp.dims).values
    prss = ds["PRSS"].values
    if isinstance(axis, (SigmaAxis, WrfAxis)):
        first_sigma = axis.levels[1]  # the Dataset's first level (see pressure)
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
          and ``TEMP`` (or ``THET`` and ``PRES``).
        - **flag=3**: ``Z* (1 - SHGT / 20000)`` from the stored ``height``
          coordinate ``Z*``, as HYSPLIT's ``prfter`` and
          :func:`arlmet.sample_points` put the levels (``ZMDL``, 25000 m at
          ``sample_points``' default ``model_top``, in place of 20000 on a
          file whose source is ``RAMS``). Requires ``SHGT``.
        - **flag=5** (WRF): as flag 1 with ``PRES``, as HYSPLIT's ``prfsig``
          does for a WRF file without WRF's hybrid coefficients; the ground
          temperature without ``T02M`` is the first level's brought down
          adiabatically from its eta. Requires ``PRSS``, ``PRES`` and ``TEMP``
          (or ``THET``, from which ``TEMP`` is ``THET * (PRES / 1000) ** 0.286``,
          as :func:`arlmet.sample_points` reads a file without ``TEMP``).
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
        if "height" not in ds.coords or "SHGT" not in ds:
            raise ValueError(
                "z_agl() for terrain-following (flag=3) files requires the 'height' "
                "coordinate and 'SHGT' in dataset."
            )
        # A RAMS file's ZMDLT is ZMDL: 25000 m at sample_points' default model_top.
        zmdlt = 25000.0 if ds.attrs.get("source") == "RAMS" else 20000.0
        shgt = ds["SHGT"]
        z = _terrain_following_z_agl(ds.coords["height"], shgt, zmdlt)
        z = z.transpose(*shgt.dims[: -len(ds.arl.grid.dims)], "level", ...)
        z.attrs = {
            "units": "m",
            "long_name": "height above ground level",
            "standard_name": "height",
        }
        return z

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
    try:
        agl = z_agl(ds)
    except ValueError as err:
        raise ValueError(
            f"z_msl() adds SHGT to the heights above ground: {err}"
        ) from err
    result = agl + ds["SHGT"]
    result.attrs = {"units": "m", "long_name": "height above mean sea level"}
    return result
