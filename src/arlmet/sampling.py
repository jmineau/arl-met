"""Point-sampling helpers for ARL meteorology files."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from arlmet._time import ensure_timestamp
from arlmet.grid import Grid, GridWindow

if TYPE_CHECKING:
    from arlmet.file import File
    from arlmet.record import DataRecord
    from arlmet.recordset import RecordSet


SURFACE_VARIABLES = {"PRSS", "SHGT"}


@dataclass(frozen=True)
class HorizontalSamplePlan:
    """
    Pre-computed bilinear interpolation weights for a set of (lon, lat) points.

    Attributes
    ----------
    method : 'linear' or 'nearest'
    window : minimal GridWindow bounding all valid points, or None if all outside
    inside : boolean mask — True where the point falls within the grid
    x0, x1, y0, y1 : integer grid-cell corners for each point
    wx, wy : fractional weights toward x1/y1 (zero for nearest-neighbor)
    """

    method: str
    window: GridWindow | None
    inside: np.ndarray
    x0: np.ndarray
    x1: np.ndarray
    y0: np.ndarray
    y1: np.ndarray
    wx: np.ndarray
    wy: np.ndarray


def _normalize_points(
    points: pd.DataFrame | Mapping[str, Any],
    *,
    require_time: bool,
    default_time: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Coerce *points* to a DataFrame with float lon/lat/z and Timestamp time columns."""
    df = points if isinstance(points, pd.DataFrame) else pd.DataFrame(points)

    missing = [name for name in ("lon", "lat") if name not in df.columns]
    if missing:
        raise ValueError(
            "Point sampling requires 'lon' and 'lat' columns, "
            f"missing {', '.join(repr(name) for name in missing)}."
        )

    normalized = pd.DataFrame(index=df.index.copy())
    normalized["lon"] = np.asarray(df["lon"], dtype=float)
    normalized["lat"] = np.asarray(df["lat"], dtype=float)

    if "time" in df.columns:
        normalized["time"] = pd.to_datetime(df["time"])
    elif default_time is not None:
        normalized["time"] = pd.Timestamp(default_time)
    elif require_time:
        raise ValueError(
            "Point sampling requires a 'time' column unless a single file time is implied."
        )

    if "z" in df.columns:
        normalized["z"] = np.asarray(df["z"], dtype=float)
    else:
        raise ValueError(
            "Point sampling requires a 'z' column for native, pressure, agl, or msl queries."
        )

    return normalized


def _normalize_variables(variables: str | Iterable[str]) -> tuple[str, ...]:
    """Coerce *variables* to a non-empty tuple of strings."""
    if isinstance(variables, str):
        names = (variables,)
    else:
        names = tuple(str(name) for name in variables)

    if not names:
        raise ValueError("variables must include at least one variable name.")
    return names


def _record_levels(recordset: RecordSet, variable: str) -> OrderedDict[int, DataRecord]:
    """Return {level_index: DataRecord} for *variable*, sorted by level index."""
    records = OrderedDict(
        sorted(
            (record.level, record)
            for record in recordset.records
            if record.variable == variable
        )
    )
    if not records and variable not in {"pressure"}:
        raise KeyError(
            f"Variable '{variable}' is not available at time {recordset.time}."
        )
    return records


def _surface_record(recordset: RecordSet, variable: str) -> DataRecord | None:
    """Return the level-0 record for *variable*, or None if absent."""
    levels = _record_levels(recordset, variable)
    if not levels:
        return None
    return next(iter(levels.values()))


def _build_horizontal_plan(
    grid: Grid,
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    method: str,
) -> HorizontalSamplePlan:
    """
    Compute grid-space fractional indices and bilinear weights for (lon, lat) points.

    The returned GridWindow is the tightest bounding box over all valid points so
    callers can read only the necessary subset of a record.
    """
    if method not in {"linear", "nearest"}:
        raise ValueError("method must be 'linear' or 'nearest'.")

    x, y = grid.fractional_indices(lon, lat)
    inside = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0.0)
        & (x <= grid.nx - 1)
        & (y >= 0.0)
        & (y <= grid.ny - 1)
    )

    x_safe = np.clip(x, 0.0, max(grid.nx - 1, 0))
    y_safe = np.clip(y, 0.0, max(grid.ny - 1, 0))
    x0 = np.floor(x_safe).astype(int)
    y0 = np.floor(y_safe).astype(int)
    x1 = np.clip(x0 + 1, 0, grid.nx - 1)
    y1 = np.clip(y0 + 1, 0, grid.ny - 1)

    if method == "nearest":
        x0 = x1 = np.rint(x_safe).astype(int)
        y0 = y1 = np.rint(y_safe).astype(int)
        wx = np.zeros_like(x_safe, dtype=float)
        wy = np.zeros_like(y_safe, dtype=float)
    else:
        wx = x_safe - x0
        wy = y_safe - y0

    if inside.any():
        window = GridWindow(
            x_start=int(min(x0[inside].min(), x1[inside].min())),
            x_stop=int(max(x0[inside].max(), x1[inside].max())) + 1,
            y_start=int(min(y0[inside].min(), y1[inside].min())),
            y_stop=int(max(y0[inside].max(), y1[inside].max())) + 1,
        )
    else:
        window = None

    return HorizontalSamplePlan(
        method=method,
        window=window,
        inside=inside,
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
        wx=wx,
        wy=wy,
    )


def _sample_field(field: np.ndarray, plan: HorizontalSamplePlan) -> np.ndarray:
    """Bilinearly interpolate a pre-read windowed field to each point in *plan*."""
    result = np.full(plan.inside.shape, np.nan, dtype=np.float32)
    if plan.window is None or not plan.inside.any():
        return result

    # Indices are relative to the full grid; subtract window origin before indexing.
    x0 = plan.x0 - plan.window.x_start
    x1 = plan.x1 - plan.window.x_start
    y0 = plan.y0 - plan.window.y_start
    y1 = plan.y1 - plan.window.y_start

    valid = plan.inside
    if plan.method == "nearest":
        result[valid] = field[y0[valid], x0[valid]]
        return result

    v00 = field[y0[valid], x0[valid]]
    v10 = field[y0[valid], x1[valid]]
    v01 = field[y1[valid], x0[valid]]
    v11 = field[y1[valid], x1[valid]]
    wx = plan.wx[valid].astype(np.float32, copy=False)
    wy = plan.wy[valid].astype(np.float32, copy=False)

    result[valid] = (
        (1.0 - wx) * (1.0 - wy) * v00
        + wx * (1.0 - wy) * v10
        + (1.0 - wx) * wy * v01
        + wx * wy * v11
    )
    return result


def _sample_record(record: DataRecord, plan: HorizontalSamplePlan) -> np.ndarray:
    """Read *record* from disk (windowed) and interpolate to each point in *plan*."""
    if plan.window is None:
        return np.full(plan.inside.shape, np.nan, dtype=np.float32)
    field = record.read(window=plan.window)
    return _sample_field(np.asarray(field, dtype=np.float32), plan)


def _interp_profile(
    values: np.ndarray, coords: np.ndarray, target: float
) -> np.float32:
    """1-D linear interpolation of *values* at *target* along *coords*. Returns NaN outside range."""
    mask = np.isfinite(values) & np.isfinite(coords)
    if mask.sum() < 2:
        return np.float32(np.nan)

    profile_values = np.asarray(values[mask], dtype=float)
    profile_coords = np.asarray(coords[mask], dtype=float)
    order = np.argsort(profile_coords)
    profile_coords = profile_coords[order]
    profile_values = profile_values[order]
    unique_coords, unique_index = np.unique(profile_coords, return_index=True)
    unique_values = profile_values[unique_index]
    return np.float32(
        np.interp(
            float(target),
            unique_coords,
            unique_values,
            left=np.nan,
            right=np.nan,
        )
    )


def _interp_profiles(
    values: np.ndarray,
    coords: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """
    Vectorised _interp_profile over n_points rows.

    Parameters
    ----------
    values : (n_points, n_levels)
    coords : (n_levels,) shared across all points, or (n_points, n_levels) per-point
    targets : (n_points,)
    """
    out = np.full(values.shape[0], np.nan, dtype=np.float32)
    shared_coords = coords.ndim == 1
    for i in range(values.shape[0]):
        profile_coords = coords if shared_coords else coords[i]
        out[i] = _interp_profile(values[i], profile_coords, targets[i])
    return out


def _sample_hgts_profiles(
    recordset: RecordSet,
    plan: HorizontalSamplePlan,
    levels: Sequence[int],
) -> np.ndarray | None:
    """Sample HGTS (geopotential height, m MSL) at *levels*. Returns (n_points, n_levels) or None if HGTS absent."""
    hgts_by_level = {r.level: r for r in recordset.records if r.variable == "HGTS"}
    if not hgts_by_level:
        return None
    n_points = len(plan.inside)
    hgts = np.full((n_points, len(levels)), np.nan, dtype=np.float32)
    for pos, level in enumerate(levels):
        if level in hgts_by_level:
            hgts[:, pos] = _sample_record(hgts_by_level[level], plan)
    return hgts


def _variable_profiles(
    recordset: RecordSet,
    variable: str,
    plan: HorizontalSamplePlan,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Sample *variable* at all its levels. Returns ((n_points, n_levels), level_indices)."""
    records = _record_levels(recordset, variable)
    levels = tuple(records.keys())
    samples = np.full((len(plan.inside), len(levels)), np.nan, dtype=np.float32)
    for pos, level in enumerate(levels):
        samples[:, pos] = _sample_record(records[level], plan)
    return samples, levels


def _sample_variable(
    recordset: RecordSet,
    variable: str,
    targets: np.ndarray,
    *,
    z_kind: str,
    plan: HorizontalSamplePlan,
    surface_pressure: np.ndarray | None,
    terrain: np.ndarray | None,
) -> np.ndarray:
    """
    Interpolate *variable* to each point's target height.

    Parameters
    ----------
    variable :
        ARL field name, or ``'pressure'`` for the virtual pressure variable.
    targets :
        Target z values in the coordinate system specified by *z_kind*.
    z_kind :
        ``'native'`` — target is a level index (fractional);
        ``'pressure'`` — target is pressure in hPa;
        ``'agl'`` — target is metres above ground level;
        ``'msl'`` — target is metres above mean sea level.
    surface_pressure :
        (n_points,) PRSS in hPa. Required for sigma/hybrid (flag 1/4) when
        z_kind='pressure' or variable='pressure'.
    terrain :
        (n_points,) SHGT in metres. Required for AGL sampling with non-terrain
        vertical axes, and for MSL sampling with terrain-following (flag=3) axes.

    Returns
    -------
    np.ndarray of shape (n_points,) float32, NaN for out-of-range or off-grid points.
    """
    axis = recordset.vertical_axis

    # --- virtual "pressure" variable ---
    if variable == "pressure":
        if z_kind == "pressure":
            return targets.astype(np.float32, copy=False)

        level_values = np.asarray(axis.levels, dtype=float)
        n = len(targets)

        if z_kind == "native":
            return _interp_profiles(
                np.broadcast_to(level_values[None, :], (n, len(level_values))),
                np.arange(len(axis.levels), dtype=float),
                targets,
            )

        # z_kind in {"agl", "msl"}: need HGTS to map pressure → height
        if axis.flag == 3:
            # Terrain-following files have no pressure coordinate.
            return np.full(n, np.nan, dtype=np.float32)

        hgts = _sample_hgts_profiles(recordset, plan, list(range(len(axis.levels))))
        if hgts is None:
            raise ValueError(
                "HGTS records are required to compute AGL/MSL heights. "
                "Ensure the ARL file contains HGTS at each level."
            )
        if axis.flag == 2:
            pressure_values = np.broadcast_to(
                level_values[None, :], (n, len(level_values))
            )
        else:  # flag=1 (sigma) or flag=4 (hybrid)
            if surface_pressure is None:
                raise ValueError(
                    "surface_pressure (PRSS) is required for sigma/hybrid axes."
                )
            pressure_values = axis.sigma_to_pressure(
                surface_pressure, list(range(len(axis.levels)))
            )

        if z_kind == "msl":
            height_coords = hgts
        else:
            if terrain is None:
                raise ValueError("terrain (SHGT) is required to compute AGL heights.")
            height_coords = np.maximum(hgts - terrain[:, None], 0.0)

        return _interp_profiles(pressure_values, height_coords, targets)

    # --- data variable ---
    samples, levels = _variable_profiles(recordset, variable, plan)
    if len(levels) == 1 or variable in SURFACE_VARIABLES:
        # Single-level or surface field — return directly without vertical interpolation.
        return samples[:, 0]

    if z_kind == "native":
        return _interp_profiles(samples, np.asarray(levels, dtype=float), targets)

    if z_kind == "pressure":
        if axis.flag == 2:
            coords = np.asarray(axis.levels, dtype=float)[list(levels)]
        elif axis.flag in {1, 4}:
            if surface_pressure is None:
                raise ValueError(
                    "surface_pressure (PRSS) is required for sigma/hybrid pressure sampling."
                )
            coords = axis.sigma_to_pressure(surface_pressure, levels)
        elif axis.flag == 3:
            raise ValueError(
                "z_kind='pressure' is not supported for terrain-following (flag=3) vertical axes."
            )
        else:
            raise NotImplementedError(
                f"z_kind='pressure' not implemented for flag={axis.flag}."
            )
        return _interp_profiles(samples, coords, targets)

    # z_kind in {"agl", "msl"}
    if axis.flag == 3:
        # Terrain-following: stored level heights are already AGL metres.
        coords = np.broadcast_to(
            np.asarray(axis.levels, dtype=float)[list(levels)],
            (len(targets), len(levels)),
        ).copy()
        if z_kind == "msl":
            if terrain is None:
                raise ValueError("terrain (SHGT) is required to sample z_kind='msl'.")
            coords = coords + terrain[:, None]
    else:
        # Flags 1, 2, 4: use HGTS (geopotential height, m MSL) from the file.
        # Matches HYSPLIT PRFPRS: ZTOP = Z(KZ) - ZSFC.
        hgts = _sample_hgts_profiles(recordset, plan, levels)
        if hgts is None:
            raise ValueError(
                "HGTS records are required to compute AGL/MSL heights. "
                "Ensure the ARL file contains HGTS at each level."
            )
        if z_kind == "msl":
            coords = hgts
        else:
            if terrain is None:
                raise ValueError("terrain (SHGT) is required to compute AGL heights.")
            coords = np.maximum(hgts - terrain[:, None], 0.0)

    return _interp_profiles(samples, coords, targets)


def sample_points_from_file(
    file: File,
    points: pd.DataFrame | Mapping[str, Any],
    variables: str | Iterable[str],
    *,
    time: pd.Timestamp | str | None = None,
    z_kind: str = "pressure",
    method: str = "linear",
) -> pd.DataFrame:
    """
    Sample meteorological variables at arbitrary (lon, lat, z, time) points from one file.

    Parameters
    ----------
    file :
        Open ARL :class:`~arlmet.file.File` in read mode.
    points :
        DataFrame or dict with columns ``lon``, ``lat``, ``z``, and optionally ``time``.
        If ``time`` is absent and *file* has a single time, that time is used for all points.
    variables :
        One or more ARL field names (e.g. ``'TEMP'``, ``'UWND'``), or ``'pressure'`` for
        the virtual pressure variable.
    time :
        Override or supply a single timestamp when *points* has no ``time`` column.
    z_kind :
        Vertical coordinate system for *z* values:

        - ``'native'`` — fractional level index
        - ``'pressure'`` — hPa
        - ``'agl'`` — metres above ground level (requires HGTS and SHGT in file)
        - ``'msl'`` — metres above mean sea level (requires HGTS in file)
    method :
        Horizontal interpolation: ``'linear'`` (bilinear) or ``'nearest'``.

    Returns
    -------
    pd.DataFrame
        Copy of *points* with one column added per requested variable.
    """
    variable_names = _normalize_variables(variables)
    require_time = time is None and len(file.times) != 1
    default_time = (
        ensure_timestamp(time)
        if time is not None
        else (file.times[0] if len(file.times) == 1 else None)
    )
    normalized = _normalize_points(
        points,
        require_time=require_time,
        default_time=default_time,
    )

    if z_kind not in {"native", "pressure", "agl", "msl"}:
        raise ValueError("z_kind must be one of 'native', 'pressure', 'agl', or 'msl'.")

    axis = file.vertical_axis
    # PRSS only needed for sigma/hybrid when computing per-point pressure values.
    need_surface_pressure = axis.flag in {1, 4} and (
        z_kind == "pressure" or "pressure" in variable_names
    )
    # AGL (flags 1/2/4) needs terrain for HGTS - SHGT; MSL (flag=3) needs it for AGL + SHGT.
    need_terrain = (z_kind == "agl" and axis.flag in {1, 2, 4}) or (
        z_kind == "msl" and axis.flag == 3
    )

    result = normalized.copy()
    for variable in variable_names:
        result[variable] = np.nan

    for sample_time, index in result.groupby("time").groups.items():
        recordset = file[ensure_timestamp(sample_time)]
        subset = result.loc[index]
        plan = _build_horizontal_plan(
            recordset.grid,
            subset["lon"].to_numpy(dtype=float),
            subset["lat"].to_numpy(dtype=float),
            method=method,
        )

        surface_pressure = terrain = None
        if need_surface_pressure:
            surface_record = _surface_record(recordset, "PRSS")
            if surface_record is None:
                raise ValueError(
                    f"Surface pressure field PRSS is required but not available at time {recordset.time}."
                )
            surface_pressure = _sample_record(surface_record, plan)
        if need_terrain:
            terrain_record = _surface_record(recordset, "SHGT")
            if terrain_record is None:
                raise ValueError(
                    f"Terrain field SHGT is required at time {recordset.time}."
                )
            terrain = _sample_record(terrain_record, plan)

        targets = subset["z"].to_numpy(dtype=float)
        for variable in variable_names:
            sampled = _sample_variable(
                recordset,
                variable,
                targets,
                z_kind=z_kind,
                plan=plan,
                surface_pressure=surface_pressure,
                terrain=terrain,
            )
            result.loc[index, variable] = sampled

    return result


def _normalize_sources(source: File | Sequence[File]) -> tuple[File, ...]:
    """Normalize one File or a sequence of Files to a tuple."""
    from arlmet.file import File

    if isinstance(source, File):
        return (source,)  # single File
    return tuple(source)


def sample_points(
    source: File | Sequence[File],
    points: pd.DataFrame | Mapping[str, Any],
    variables: str | Iterable[str],
    *,
    time: pd.Timestamp | str | None = None,
    z_kind: str = "pressure",
    method: str = "linear",
) -> pd.DataFrame:
    """
    Sample meteorological variables at arbitrary (lon, lat, z, time) points.

    Accepts one file or a sequence of files spanning different time periods.
    For single-file sampling prefer :func:`sample_points_from_file` to avoid
    the overhead of building a time-to-file map.

    Parameters
    ----------
    source :
        A single :class:`~arlmet.file.File` or a sequence of files. Each
        timestamp must appear in at most one file.
    points :
        DataFrame or dict with columns ``lon``, ``lat``, ``z``, and ``time``.
        ``time`` may be omitted when *source* is a single-time file.
    variables :
        One or more ARL field names, or ``'pressure'`` for the virtual pressure variable.
    time :
        Override or supply a single timestamp when *points* has no ``time`` column.
    z_kind :
        Vertical coordinate for *z*: ``'native'``, ``'pressure'``, ``'agl'``, or ``'msl'``.
        See :func:`sample_points_from_file` for details.
    method :
        Horizontal interpolation: ``'linear'`` (bilinear) or ``'nearest'``.

    Returns
    -------
    pd.DataFrame
        Copy of *points* with one column added per requested variable, index preserved.

    Examples
    --------
    >>> import pandas as pd
    >>> import arlmet
    >>> points = pd.DataFrame(
    ...     {"lon": [-111.9], "lat": [40.7], "z": [850.0], "time": ["2024-07-18 00:00"]}
    ... )
    >>> with arlmet.File("met.arl") as met:
    ...     arlmet.sample_points(met, points, ["UWND", "VWND"])
    """
    files = _normalize_sources(source)
    if len(files) == 1:
        return sample_points_from_file(
            files[0],
            points,
            variables,
            time=time,
            z_kind=z_kind,
            method=method,
        )

    normalized = _normalize_points(
        points,
        require_time=True,
        default_time=ensure_timestamp(time) if time is not None else None,
    )
    time_map: dict[pd.Timestamp, File] = {}
    for file in files:
        for sample_time in file.times:
            if sample_time in time_map:
                raise ValueError(
                    f"Multiple sources contain meteorology for time {sample_time}."
                )
            time_map[ensure_timestamp(sample_time)] = file

    missing_times = sorted(set(normalized["time"]) - set(time_map))
    if missing_times:
        raise ValueError(
            "No source contains the requested point times: "
            + ", ".join(str(ensure_timestamp(t)) for t in missing_times)
        )

    pieces: list[pd.DataFrame] = []
    for sample_time, index in normalized.groupby("time").groups.items():
        piece = sample_points_from_file(
            time_map[ensure_timestamp(sample_time)],
            normalized.loc[index],
            variables,
            time=ensure_timestamp(sample_time),
            z_kind=z_kind,
            method=method,
        )
        pieces.append(piece)

    return pd.concat(pieces).reindex(normalized.index)
