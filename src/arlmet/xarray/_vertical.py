"""xarray-aware vertical coordinate helpers for ARL datasets."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import numpy as np
import xarray as xr

R_D = 287.05  # dry air gas constant [J/(kg·K)]
G = 9.80665  # standard gravity [m/s²]


def _hypsometric_z_agl(
    p_levels: xr.DataArray,
    p_surface: xr.DataArray,
    temp: xr.DataArray,
) -> xr.DataArray:
    """
    Height AGL (m) at each level via the hypsometric equation.

    ``p_levels`` must be ordered from high to low pressure (surface to top).
    For flag=2, ``p_levels`` is 1D ``(level,)`` and is broadcast to match
    ``temp``. For flags 1/4, ``p_levels`` has the same shape as ``temp``.

    The first level is integrated from ``p_surface`` to ``p_levels[0]``
    using the temperature at that level as an approximation. Subsequent layers
    use the mean temperature of the bounding levels.
    """
    if "level" not in temp.dims:
        raise ValueError("'TEMP' must have a 'level' dimension.")

    level_ax = list(temp.dims).index("level")
    temp_vals = temp.values
    prss_vals = p_surface.values
    nlev = temp_vals.shape[level_ax]

    # Broadcast p_levels to match temp shape if it is 1D
    if p_levels.ndim == 1:
        expand_axes = [i for i in range(temp_vals.ndim) if i != level_ax]
        p_vals = p_levels.values
        for ax in sorted(expand_axes):
            p_vals = np.expand_dims(p_vals, ax)
        p_vals = np.broadcast_to(p_vals, temp_vals.shape).copy()
    else:
        p_vals = p_levels.values

    def _take(arr: np.ndarray, i: int) -> np.ndarray:
        idx: list[int | slice] = [slice(None)] * arr.ndim
        idx[level_ax] = i
        return arr[tuple(idx)]

    def _take_range(arr: np.ndarray, start: int | None, stop: int | None) -> np.ndarray:
        idx: list[int | slice | None] = [slice(None)] * arr.ndim
        idx[level_ax] = slice(start, stop)
        return arr[tuple(idx)]

    # Layer 0: from PRSS to p[0], using T[0] as representative temperature
    dz0 = (R_D / G) * _take(temp_vals, 0) * np.log(prss_vals / _take(p_vals, 0))

    dz0_exp = np.expand_dims(dz0, level_ax)  # (..., 1, ...)

    if nlev > 1:
        T_lower = _take_range(temp_vals, None, -1)
        T_upper = _take_range(temp_vals, 1, None)
        p_lower = _take_range(p_vals, None, -1)
        p_upper = _take_range(p_vals, 1, None)
        T_mean = (T_lower + T_upper) / 2.0
        dz_layers = (R_D / G) * T_mean * np.log(p_lower / p_upper)
        dz_all = np.concatenate([dz0_exp, dz_layers], axis=level_ax)
    else:
        dz_all = dz0_exp

    z_vals = np.cumsum(dz_all, axis=level_ax)

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

    if vaxis.flag == 2:
        if "pressure" not in ds.coords:
            raise ValueError("Dataset has no 'pressure' coordinate.")
        return ds.coords["pressure"]

    if vaxis.flag in (1, 4):
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
        p_flat = vaxis.sigma_to_pressure(prss_vals.reshape(-1), level_ints)  # (n, nlev)
        nlev = len(level_ints)

        # p_flat shape: (n, nlev) → reshape to (*orig_shape, nlev)
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
        # Include 2D aux coords (lat/lon for projected grids)
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

    if vaxis.flag == 3:
        if "PRES" not in ds:
            raise ValueError(
                "pressure() for terrain-following (flag=3) files requires a 'PRES' "
                "(3D pressure) variable in the dataset."
            )
        return ds["PRES"]

    raise NotImplementedError(
        f"pressure() is not supported for flag={vaxis.flag} ({vaxis.coord_system})."
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
        - **If ``HGTS`` is present** (any flag): returns the ``HGTS`` data
          variable directly (3D heights AGL in metres). HYSPLIT files commonly
          store this field for pressure and sigma coordinates.
        - **flag=3 without ``HGTS``**: returns the stored 1D ``height``
          non-dim coord ``(level,)`` — terrain-following levels are heights AGL.
        - **flag=1/2/4 without ``HGTS``**: hypsometric integration from the
          surface to each level using ``PRSS`` (hPa) and ``TEMP`` (K).

    Notes
    -----
    The hypsometric fallback uses the temperature at the lowest level for
    the surface-to-level-0 layer, and the mean temperature of bounding levels
    for all subsequent layers.
    """
    vaxis = ds.arl.vertical_axis

    # HGTS present → use it directly regardless of flag
    if "HGTS" in ds:
        return ds["HGTS"]

    if vaxis.flag == 3:
        if "height" not in ds.coords:
            raise ValueError(
                "Dataset has no 'height' coordinate and no 'HGTS' variable."
            )
        return ds.coords["height"]

    if vaxis.flag not in (1, 2, 4):
        raise NotImplementedError(
            f"z_agl() is not supported for flag={vaxis.flag} ({vaxis.coord_system})."
        )

    for name in ("PRSS", "TEMP"):
        if name not in ds:
            raise ValueError(
                f"z_agl() requires '{name}' in dataset (or 'HGTS' as an alternative)."
            )

    p_levels = pressure(ds)

    # For flag=2, ensure pressure levels are ordered high-to-low (surface→top)
    if vaxis.flag == 2:
        p_vals = p_levels.values
        if len(p_vals) > 1 and p_vals[0] < p_vals[-1]:
            ds = ds.isel(level=slice(None, None, -1))
            p_levels = pressure(ds)

    return _hypsometric_z_agl(p_levels, ds["PRSS"], ds["TEMP"])


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
