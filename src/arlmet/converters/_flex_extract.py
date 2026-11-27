"""Convert the ERA5 files flex_extract writes for FLEXPART."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from arlmet._header import Header
from arlmet.vertical import R_D

from ._era5 import MODEL_LEVEL_FIELDS, _first_repeated, _hybrid_axis, _stored_levels
from ._pressure import (
    GRAVITY,
    Field,
    pressure_level_dataset,
    scalar_coord,
    stack_times,
    standardize,
)

__all__ = ["from_flex_extract"]

#: flex_extract's model-level fields (cfgrib names, ``hybrid``) and their ARL fields.
#: ``etadot`` is not converted: HYSPLIT's ``WWND`` is a pressure velocity.
LEVEL_FIELDS: dict[str, Field] = {
    name: field for name, field in MODEL_LEVEL_FIELDS.items() if name != "w"
}

#: flex_extract's surface fields and their ARL fields. flex_extract has turned
#: ERA5's hourly totals into rates (W m-2, N m-2, mm h-1) at the valid time,
#: whatever their GRIB labels say; ECMWF's fluxes are positive downward,
#: HYSPLIT's heat flux positive upward. ``ustar`` and ``tp`` are made here.
SURFACE_FIELDS: dict[str, Field] = {
    "sp": Field("PRSS", 0.01, "hPa"),
    "msl": Field("MSLP", 0.01, "hPa"),
    "z": Field("SHGT", 1.0 / GRAVITY, "m"),
    "t2m": Field("T02M", 1.0, "K"),
    "d2m": Field("DP2M", 1.0, "K"),
    "u10": Field("U10M", 1.0, "m s-1"),
    "v10": Field("V10M", 1.0, "m s-1"),
    "tcc": Field("TCLD", 100.0, "%"),
    "sshf": Field("SHTF", -1.0, "W m-2"),
    "ssr": Field("DSWF", 1.0, "W m-2"),
    "ustar": Field("USTR", 1.0, "m s-1"),
    "tp": Field("TPP1", 0.001, "m"),
}

#: The surface fields read from the files: those above, and what ``ustar``
#: (the surface stress) and ``tp`` (the two precipitation rates) are made of.
_READ = (
    *(n for n in SURFACE_FIELDS if n not in ("ustar", "tp")),
    "ewss",
    "nsss",
    "lsp",
    "cp",
)


def _longitudes(da: xr.DataArray) -> xr.DataArray:
    """
    Return *da* with its longitudes from 0 to 360, or -180 to 180 across 0.

    flex_extract's files hold the same grid on both conventions; this puts
    every field on one, so that they line up. A grid that goes all the way
    round is put on 0 to 360, its columns turned to start at 0.
    """
    lon = np.asarray(da["longitude"].values, dtype=float) % 360.0
    if (np.diff(lon) <= 0).any():
        # Equal steps that, with one more, make a whole turn.
        if np.isclose((lon.max() - lon.min()) * lon.size / (lon.size - 1), 360.0):
            return da.assign_coords(longitude=lon).sortby("longitude")
        lon = np.where(lon >= 180.0, lon - 360.0, lon)
    return da.assign_coords(longitude=lon)


def _storable(model_levels: Iterable[float]) -> list[float]:
    """
    Return the model levels an ARL file can hold, from the ground up.

    They stop below the first level whose pressure offset rounds to the same
    whole hPa as the level below it (above about model level 24, see
    :func:`arlmet.converters.from_era5`), and at :attr:`Header.MAX_LEVEL`.
    """
    levels = sorted((float(n) for n in model_levels), reverse=True)
    repeated = _first_repeated(_stored_levels(levels))
    return levels[:repeated][: Header.MAX_LEVEL]


def from_flex_extract(datasets: Iterable[xr.Dataset]) -> xr.Dataset:
    """
    Convert the ERA5 files flex_extract writes for FLEXPART to a Dataset for :func:`arlmet.write_dataset`.

    `flex_extract <https://www.flexpart.eu/flex_extract/>`_ retrieves ERA5
    for FLEXPART and writes one GRIB file per hour (``EAyymmddhh``), with the
    model levels, the surface fields and the precipitation on ERA5's own lat/lon
    grid. This takes the Datasets :func:`cfgrib.open_datasets` returns for one
    or several of those files and gives an ARL hybrid Dataset (vertical flag 4)
    on model levels, as :func:`~arlmet.converters.from_era5` writes ERA5 on
    model levels. Each file's valid time is one time step.

    The files are not ERA5 as the Climate Data Store serves it, so this is not
    :func:`~arlmet.converters.from_era5`:

    - flex_extract has turned ERA5's hourly totals into rates at the valid
      time (W m-2, N m-2, mm h-1), but keeps their GRIB names, units and
      ``stepType=accum``; they are read as the rates they are.
    - Some fields are on longitudes from 0 to 360 and others from -180 to 180,
      on the same grid; all are put on one (0 to 360 for a global grid).
    - The files hold all 137 model levels, more than an ARL file can: the
      lowest 99 are kept (model levels 137 to 39, up to about 25 hPa), as a
      record's header numbers its level in two characters. Fewer are kept when
      the files hold fewer, and none above about model level 24, whose
      pressure offsets round to the same whole hPa as the level below.

    ===================  ======  ==============================================
    flex_extract         ARL     conversion
    ===================  ======  ==============================================
    ``t``                TEMP    K
    ``u``, ``v``         UWND,   m s-1
                         VWND
    ``q``                SPHU    kg kg-1
    ``sp``               PRSS    Pa to hPa
    ``msl``              MSLP    Pa to hPa
    ``z``                SHGT    surface geopotential / 9.80665 (m)
    ``t2m``, ``d2m``     T02M,   K
                         DP2M
    ``u10``, ``v10``     U10M,   m s-1
                         V10M
    ``tcc``              TCLD    fraction to %
    ``sshf``             SHTF    W m-2, positive upward
    ``ssr``              DSWF    W m-2, net (not downward) shortwave
    ``ewss``, ``nsss``   USTR    ``sqrt(sqrt(ewss² + nsss²) / ρ)`` (m s-1),
                                 ``ρ = sp / (R_d t2m)``
    ``lsp`` + ``cp``     TPP1    mm h-1 at the valid time to m in an hour
    ===================  ======  ==============================================

    Not written, because the files do not hold them: ``PBLH``, ``LHTF`` and
    ``WWND``. flex_extract's vertical velocity is ``etadot`` (s-1), the rate of
    change of the hybrid coordinate, not a pressure velocity; without
    ``WWND`` HYSPLIT finds the vertical velocity from the divergence of the
    horizontal wind, and its mixed layer from the temperature profile.

    Parameters
    ----------
    datasets : iterable of xarray.Dataset
        The Datasets of one or more flex_extract files, as
        :func:`cfgrib.open_datasets` returns them (each with a scalar
        ``valid_time``; the precipitation's with a ``step`` dimension, of which
        the first is the valid time's), one list after another. Datasets and
        variables not in the table (the invariant fields, ``etadot``, the cloud
        water) are left out.

    Returns
    -------
    xarray.Dataset
        The fields in :func:`arlmet.open_dataset`'s layout, with source
        ``"ERA5"`` and each variable's ARL ``units`` in its attrs.

    Raises
    ------
    ValueError
        If no Dataset holds model-level fields, a Dataset has no scalar valid
        time, the time steps have different model levels, the grids differ,
        or a time step has no surface pressure ``sp``.

    Notes
    -----
    FLEXPART finds the friction velocity from the surface stress as above
    (``calcpar``), with the air density at 2 m. The precipitation is the rate
    at the valid time, which flex_extract interpolates from ERA5's hourly
    totals (its ``RRINT`` option); written as ``TPP1`` it stands for the hour
    before.

    Each field is read once and is then in memory: flex_extract's files are
    already cropped to the domain they were retrieved for.

    Examples
    --------
    >>> import cfgrib  # doctest: +SKIP
    >>> import arlmet
    >>> files = ["EA23053118", "EA23053119", "EA23053120"]
    >>> datasets = [
    ...     ds for f in files for ds in cfgrib.open_datasets(f)
    ... ]  # doctest: +SKIP
    >>> ds = arlmet.converters.from_flex_extract(datasets)  # doctest: +SKIP
    >>> arlmet.write_dataset(ds, "era5.arl")  # doctest: +SKIP
    """
    upper: dict[tuple[pd.Timestamp, str], xr.DataArray] = {}
    surface: dict[tuple[pd.Timestamp, str], xr.DataArray] = {}
    for ds in datasets:
        if not isinstance(ds, xr.Dataset):
            raise TypeError(
                f"from_flex_extract() takes xarray.Datasets, got {type(ds).__name__}."
            )
        if "hybrid" in ds.dims:
            target, names = upper, [n for n in LEVEL_FIELDS if n in ds.data_vars]
        else:
            target, names = surface, [n for n in _READ if n in ds.data_vars]
        if not names:
            continue
        if "step" in ds.dims:
            # flex_extract's precipitation: the valid time, then two times
            # between it and the next hour.
            ds = ds.isel(step=0)
        valid_time = scalar_coord(ds, "valid_time")
        if valid_time is None:
            raise ValueError(
                "Each Dataset needs a scalar 'valid_time', as cfgrib.open_datasets() "
                f"gives one per GRIB file; got one with dims {dict(ds.sizes)}."
            )
        time = pd.Timestamp(valid_time)
        for name in names:
            # Read once: cfgrib reads a whole message on every access.
            target[(time, name)] = _longitudes(ds[name].reset_coords(drop=True)).load()

    times = sorted({time for time, _ in upper})
    if not times:
        raise ValueError(
            "None of the Datasets holds flex_extract's model-level fields "
            f"({', '.join(LEVEL_FIELDS)} on 'hybrid' levels)."
        )
    missing = [time for time in times if (time, "sp") not in surface]
    if missing:
        raise ValueError(
            f"No Dataset holds the surface pressure 'sp' at {missing[0]}, which "
            "ERA5 on model levels needs: HYSPLIT finds the levels' pressures from it."
        )

    upper_ds = standardize(
        stack_times(upper, times),
        list(LEVEL_FIELDS),
        upper=True,
        label="the model-level fields",
        level_dims=("hybrid",),
        level_kind="model-level",
    )
    upper_ds = upper_ds.sel(level=_storable(upper_ds["level"].values))

    surface_ds = stack_times(surface, times)
    if {"ewss", "nsss", "sp", "t2m"} <= set(surface_ds.data_vars):
        stress = np.sqrt(surface_ds["ewss"] ** 2 + surface_ds["nsss"] ** 2)
        density = surface_ds["sp"] / (R_D * surface_ds["t2m"])
        surface_ds["ustar"] = np.sqrt(stress / density)
    if {"lsp", "cp"} <= set(surface_ds.data_vars):
        surface_ds["tp"] = surface_ds["lsp"] + surface_ds["cp"]
    surface_ds = standardize(
        surface_ds, list(SURFACE_FIELDS), upper=False, label="the surface fields"
    )
    return pressure_level_dataset(
        upper_ds,
        surface_ds,
        LEVEL_FIELDS,
        SURFACE_FIELDS,
        source="ERA5",
        labels=("the model-level fields", "the surface fields"),
        vertical_axis=_hybrid_axis,
    )
