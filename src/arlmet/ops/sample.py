"""Point-sampling helpers for ARL meteorology files."""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

from arlmet._time import ensure_timestamp
from arlmet.grid import Grid, GridWindow
from arlmet.vertical import (
    HybridAxis,
    PressureAxis,
    SigmaAxis,
    TerrainAxis,
    hypsometric_z_agl,
)

if TYPE_CHECKING:
    from arlmet.file import File
    from arlmet.record import DataRecord
    from arlmet.recordset import RecordSet


SURFACE_VARIABLES = {"PRSS", "SHGT"}

#: Wind component pairs that are stored grid-relative on projected grids.
WIND_PAIRS: tuple[tuple[str, str], ...] = (("UWND", "VWND"), ("U10M", "V10M"))

# One sampling input: an open File or a path to an ARL file.
FileLike = Union["File", str, "os.PathLike[str]"]

#: Vertical coordinate systems accepted for the ``z`` of sample points.
ZKind = Literal["native", "pressure", "agl", "msl"]
#: Horizontal interpolation methods for point sampling.
SampleMethod = Literal["linear", "nearest"]

_Z_KINDS = frozenset({"native", "pressure", "agl", "msl"})
_METHODS = frozenset({"linear", "nearest"})


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

    method: SampleMethod
    window: GridWindow | None
    inside: npt.NDArray[Any]
    x0: npt.NDArray[Any]
    x1: npt.NDArray[Any]
    y0: npt.NDArray[Any]
    y1: npt.NDArray[Any]
    wx: npt.NDArray[Any]
    wy: npt.NDArray[Any]


def _normalize_points(
    points: pd.DataFrame | Mapping[str, Any],
    variable_names: Sequence[str],
    *,
    time: pd.Timestamp | str | None,
    implied_time: pd.Timestamp | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate *points* and split it into the result frame and sampling coordinates.

    Returns ``(result, coords)``. *result* is a copy of *points* as a DataFrame,
    with every column and the index preserved, that the sampled variables are
    added to. *coords* has a positional ``RangeIndex`` aligned row-for-row with
    *result* and holds float ``lon``/``lat``/``z`` and Timestamp ``time``
    columns.

    The point time comes from the ``time`` column, else from *time*, else from
    *implied_time* (the single valid time of the inputs). Giving both a
    ``time`` column and *time* is ambiguous and raises ``ValueError``.
    """
    df = points if isinstance(points, pd.DataFrame) else pd.DataFrame(points)

    missing = [name for name in ("lon", "lat", "z") if name not in df.columns]
    if missing:
        raise ValueError(
            "Point sampling requires 'lon', 'lat', and 'z' columns, "
            f"missing {', '.join(repr(name) for name in missing)}."
        )

    clashes = [name for name in variable_names if name in df.columns]
    if clashes:
        raise ValueError(
            "Requested variables collide with existing columns of points: "
            f"{', '.join(repr(name) for name in clashes)}. Rename or drop those "
            "columns before sampling."
        )

    has_time_column = "time" in df.columns
    if has_time_column and time is not None:
        raise ValueError(
            "points has a 'time' column and time= was also given; pass only one."
        )

    coords = pd.DataFrame(
        {
            "lon": np.asarray(df["lon"], dtype=float),
            "lat": np.asarray(df["lat"], dtype=float),
            "z": np.asarray(df["z"], dtype=float),
        }
    )
    if has_time_column:
        coords["time"] = pd.to_datetime(df["time"]).to_numpy()
    elif time is not None:
        coords["time"] = ensure_timestamp(time)
    elif implied_time is not None:
        coords["time"] = implied_time
    else:
        raise ValueError(
            "Point sampling needs a time: add a 'time' column to points or pass "
            "time=, unless the input file(s) contain exactly one valid time."
        )
    if np.any(pd.isna(coords["time"])):
        raise ValueError("points has missing (NaT) values in its 'time' column.")

    return df.copy(), coords


def _normalize_variables(variables: str | Iterable[str]) -> tuple[str, ...]:
    """Coerce *variables* to a non-empty tuple of strings."""
    if isinstance(variables, str):
        names = (variables,)
    else:
        names = tuple(str(name) for name in variables)

    if not names:
        raise ValueError("variables must include at least one variable name.")
    return names


def _check_wind_pairs(variable_names: Sequence[str]) -> list[tuple[str, str]]:
    """Return the wind pairs present in *variable_names*; raise if a component lacks its partner."""
    requested = set(variable_names)
    pairs: list[tuple[str, str]] = []
    for u_name, v_name in WIND_PAIRS:
        has_u = u_name in requested
        has_v = v_name in requested
        if has_u != has_v:
            missing = v_name if has_u else u_name
            present = u_name if has_u else v_name
            raise ValueError(
                f"earth_relative=True needs both wind components: '{present}' "
                f"was requested without '{missing}'."
            )
        if has_u:
            pairs.append((u_name, v_name))
    return pairs


def _rotate_winds(
    sampled: dict[str, npt.NDArray[Any]],
    grid: Grid,
    lon: npt.NDArray[Any],
    lat: npt.NDArray[Any],
    pairs: Sequence[tuple[str, str]],
) -> None:
    """Rotate each sampled wind pair in *sampled* from grid- to earth-relative, in place."""
    if grid.is_latlon or not pairs:
        return
    for u_name, v_name in pairs:
        u_earth, v_earth = grid.rotate_winds(
            sampled[u_name].astype(float), sampled[v_name].astype(float), lon, lat
        )
        sampled[u_name] = np.asarray(u_earth, dtype=np.float32)
        sampled[v_name] = np.asarray(v_earth, dtype=np.float32)


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
    lon: npt.NDArray[Any],
    lat: npt.NDArray[Any],
    *,
    method: SampleMethod,
) -> HorizontalSamplePlan:
    """
    Compute grid-space fractional indices and bilinear weights for (lon, lat) points.

    The returned GridWindow is the tightest bounding box over all valid points so
    callers can read only the necessary subset of a record.
    """
    if method not in {"linear", "nearest"}:
        raise ValueError("method must be 'linear' or 'nearest'.")

    x, y = grid.fractional_indices(lon, lat)
    # On a global lat/lon grid, x in [nx - 1, nx) lies between the last and
    # first columns and interpolates across the seam.
    wraps = grid.wraps_lon
    x_max = float(grid.nx) if wraps else float(grid.nx - 1)
    inside = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0.0)
        & ((x < x_max) if wraps else (x <= x_max))
        & (y >= 0.0)
        & (y <= grid.ny - 1)
    )

    x_safe = np.clip(np.nan_to_num(x), 0.0, max(x_max, 0.0))
    y_safe = np.clip(np.nan_to_num(y), 0.0, max(grid.ny - 1, 0))
    x0 = np.minimum(np.floor(x_safe).astype(int), max(grid.nx - 1, 0))
    y0 = np.floor(y_safe).astype(int)
    x1 = (x0 + 1) % grid.nx if wraps else np.clip(x0 + 1, 0, grid.nx - 1)
    y1 = np.clip(y0 + 1, 0, grid.ny - 1)

    if method == "nearest":
        x_near = np.rint(x_safe).astype(int)
        x0 = x1 = (x_near % grid.nx) if wraps else x_near
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


def _sample_field(
    field: npt.NDArray[Any], plan: HorizontalSamplePlan
) -> npt.NDArray[Any]:
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


def _sample_record(record: DataRecord, plan: HorizontalSamplePlan) -> npt.NDArray[Any]:
    """Read *record* from disk (windowed) and interpolate to each point in *plan*."""
    if plan.window is None:
        return np.full(plan.inside.shape, np.nan, dtype=np.float32)
    field = record.read(window=plan.window)
    return _sample_field(np.asarray(field, dtype=np.float32), plan)


def _interp_profile(
    values: npt.NDArray[Any], coords: npt.NDArray[Any], target: float
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
    values: npt.NDArray[Any],
    coords: npt.NDArray[Any],
    targets: npt.NDArray[Any],
) -> npt.NDArray[Any]:
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
) -> npt.NDArray[Any] | None:
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
) -> tuple[npt.NDArray[Any], tuple[int, ...]]:
    """Sample *variable* at all its levels. Returns ((n_points, n_levels), level_indices)."""
    records = _record_levels(recordset, variable)
    levels = tuple(records.keys())
    samples = np.full((len(plan.inside), len(levels)), np.nan, dtype=np.float32)
    for pos, level in enumerate(levels):
        samples[:, pos] = _sample_record(records[level], plan)
    return samples, levels


def _vertical_coords(
    recordset: RecordSet,
    levels: Sequence[int],
    *,
    z_kind: ZKind,
    plan: HorizontalSamplePlan,
    surface_pressure: npt.NDArray[Any] | None,
    terrain: npt.NDArray[Any] | None,
) -> npt.NDArray[Any]:
    """
    Per-point vertical coordinate at each *level*, expressed in the *z_kind* system.

    Returns an ``(n_points, n_levels)`` array giving the coordinate that a target
    *z* (in the same *z_kind* units) is interpolated against.

    Dispatches to the vertical axis subclass methods, matching HYSPLIT:

    - flag=2 (pressure): stored levels for pressure; HGTS for heights.
    - flag=1/4 (sigma/hybrid): sigma-to-pressure conversion; hypsometric for heights.
    - flag=3 (terrain): stored level heights are AGL; no pressure coordinate.
    """
    axis = recordset.vertical_axis
    level_list = list(levels)
    n_points = len(plan.inside)

    def _broadcast(values_1d: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Repeat per-level values for every point: ``(n_points, n_levels)``."""
        return np.broadcast_to(values_1d[None, :], (n_points, len(level_list))).copy()

    if z_kind == "native":
        return _broadcast(np.asarray(level_list, dtype=float))

    if z_kind == "pressure":
        if isinstance(axis, PressureAxis):
            return _broadcast(axis.to_pressure()[level_list])
        if isinstance(axis, (SigmaAxis, HybridAxis)):
            if surface_pressure is None:
                raise ValueError(
                    "surface_pressure (PRSS) is required for sigma/hybrid pressure sampling."
                )
            return axis.to_pressure(surface_pressure=surface_pressure)[:, level_list]
        if isinstance(axis, TerrainAxis):
            raise ValueError(
                "z_kind='pressure' is not supported for terrain-following (flag=3) vertical axes."
            )
        raise NotImplementedError(
            f"z_kind='pressure' not implemented for {type(axis).__name__}."
        )

    # z_kind in {"agl", "msl"}
    if isinstance(axis, TerrainAxis):
        coords = _broadcast(axis.to_height_agl()[level_list])
        if z_kind == "msl":
            if terrain is None:
                raise ValueError("terrain (SHGT) is required to sample z_kind='msl'.")
            coords = coords + terrain[:, None]
        return coords

    if isinstance(axis, PressureAxis):
        # HYSPLIT PRFPRS: heights come from HGTS (geopotential height MSL).
        hgts = _sample_hgts_profiles(recordset, plan, level_list)
        if hgts is None:
            raise ValueError(
                "Pressure-level (flag=2) files require HGTS records for AGL/MSL sampling."
            )
        if z_kind == "msl":
            return hgts
        return np.maximum(
            hgts - _resolve_terrain(recordset, plan, terrain)[:, None], 0.0
        )

    if isinstance(axis, (SigmaAxis, HybridAxis)):
        # HYSPLIT PRFSIG/PRFECM: hypsometric integration from PRSS + TEMP.
        assert surface_pressure is not None  # guaranteed by need_surface_pressure
        agl = _hypsometric_agl_profiles(
            recordset, plan, level_list, axis=axis, surface_pressure=surface_pressure
        )
        if z_kind == "agl":
            return agl
        return agl + _resolve_terrain(recordset, plan, terrain)[:, None]

    raise NotImplementedError(
        f"z_kind='{z_kind}' not implemented for {type(axis).__name__}."
    )


def _resolve_terrain(
    recordset: RecordSet,
    plan: HorizontalSamplePlan,
    terrain: npt.NDArray[Any] | None,
) -> npt.NDArray[Any]:
    """Return per-point terrain height (SHGT, m); sample it if not precomputed."""
    if terrain is not None:
        return terrain
    shgt_records = [r for r in recordset.records if r.variable == "SHGT"]
    if not shgt_records:
        raise ValueError("terrain (SHGT) is required to compute AGL/MSL heights.")
    return _sample_record(min(shgt_records, key=lambda r: r.level), plan)


def _hypsometric_agl_profiles(
    recordset: RecordSet,
    plan: HorizontalSamplePlan,
    levels: Sequence[int],
    *,
    axis: SigmaAxis | HybridAxis,
    surface_pressure: npt.NDArray[Any],
) -> npt.NDArray[Any]:
    """
    Per-point AGL heights ``(n_points, n_levels)`` via the hypsometric equation.

    Used for sigma/hybrid files (flags 1/4). Requires pre-sampled surface
    pressure and TEMP records in the file.
    """
    level_list = list(levels)
    n_points = len(plan.inside)

    temp_records = {r.level: r for r in recordset.records if r.variable == "TEMP"}
    if not temp_records:
        raise ValueError(
            f"{axis.coord_system} (flag={axis.flag}) files require TEMP "
            "for AGL/MSL sampling."
        )
    temp = np.full((n_points, len(level_list)), np.nan, dtype=np.float32)
    for pos, level in enumerate(level_list):
        if level in temp_records:
            temp[:, pos] = _sample_record(temp_records[level], plan)

    p = axis.to_pressure(surface_pressure=surface_pressure)[:, level_list]
    return np.asarray(
        hypsometric_z_agl(p, surface_pressure, temp, level_axis=1),
        dtype=np.float32,
    )


def _sample_variable(
    recordset: RecordSet,
    variable: str,
    targets: npt.NDArray[Any],
    *,
    z_kind: ZKind,
    plan: HorizontalSamplePlan,
    surface_pressure: npt.NDArray[Any] | None,
    terrain: npt.NDArray[Any] | None,
) -> npt.NDArray[Any]:
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
            # Pressure sampled in pressure space is the requested target itself.
            return targets.astype(np.float32, copy=False)
        if isinstance(axis, TerrainAxis):
            # Terrain-following files have no pressure coordinate.
            return np.full(len(targets), np.nan, dtype=np.float32)

        # Treat pressure as a virtual field whose per-level values are the
        # pressures themselves, interpolated over the requested z system.
        all_levels = tuple(range(len(axis.levels)))
        values = _vertical_coords(
            recordset,
            all_levels,
            z_kind="pressure",
            plan=plan,
            surface_pressure=surface_pressure,
            terrain=terrain,
        )
        coords = _vertical_coords(
            recordset,
            all_levels,
            z_kind=z_kind,
            plan=plan,
            surface_pressure=surface_pressure,
            terrain=terrain,
        )
        return _interp_profiles(values, coords, targets)

    # --- data variable ---
    samples, levels = _variable_profiles(recordset, variable, plan)
    if len(levels) == 1 or variable in SURFACE_VARIABLES:
        # Single-level or surface field — return directly without vertical interpolation.
        return samples[:, 0]

    coords = _vertical_coords(
        recordset,
        levels,
        z_kind=z_kind,
        plan=plan,
        surface_pressure=surface_pressure,
        terrain=terrain,
    )
    return _interp_profiles(samples, coords, targets)


def _sample_points_from_file(
    file: File,
    time: pd.Timestamp,
    coords: pd.DataFrame,
    variable_names: Sequence[str],
    *,
    z_kind: ZKind,
    method: SampleMethod,
    wind_pairs: Sequence[tuple[str, str]],
) -> dict[str, npt.NDArray[Any]]:
    """
    Sample variables at points that all share one valid time in one file.

    Parameters
    ----------
    file :
        Open ARL :class:`~arlmet.file.File` in read mode that contains *time*.
    time :
        Valid time of every point in *coords*.
    coords :
        Normalized points (see :func:`_normalize_points`): float ``lon``,
        ``lat``, and ``z`` columns.
    variable_names :
        ARL field names, or ``'pressure'`` for the virtual pressure variable.
    z_kind :
        Vertical coordinate system of *z*. The method is fixed by the file's
        vertical axis, matching HYSPLIT, with no fallback between methods:

        - ``'native'``: fractional level index, for every axis.
        - ``'pressure'``: hPa. Sigma/hybrid (flag 1/4) convert from PRSS;
          pressure (flag 2) uses the stored levels; terrain-following
          (flag 3) raises ``ValueError``.
        - ``'agl'``: metres above ground. Sigma/hybrid integrate
          hypsometrically from PRSS and TEMP; pressure uses HGTS - SHGT
          (HGTS is required); terrain-following uses the stored levels.
        - ``'msl'``: metres above mean sea level. Sigma/hybrid add SHGT to the
          hypsometric AGL height; pressure uses HGTS (required);
          terrain-following adds SHGT to the stored levels.
    method :
        Horizontal interpolation: ``'linear'`` (bilinear) or ``'nearest'``.
    wind_pairs :
        Wind component pairs to rotate from grid- to earth-relative (see
        :func:`_check_wind_pairs`); empty to leave winds grid-relative.

    Returns
    -------
    dict[str, np.ndarray]
        One array per variable, aligned with the rows of *coords*.
    """
    recordset = file[time]
    axis = file.vertical_axis
    # Sigma/hybrid need PRSS for pressure conversion and hypsometric heights.
    need_surface_pressure = (
        isinstance(axis, (SigmaAxis, HybridAxis)) and z_kind != "native"
    )
    # flag=2 AGL needs terrain for HGTS - SHGT; flag=3 MSL needs terrain;
    # flag=1/4 MSL needs terrain for hypsometric AGL + SHGT.
    need_terrain = (z_kind == "agl" and isinstance(axis, PressureAxis)) or (
        z_kind == "msl" and isinstance(axis, (TerrainAxis, SigmaAxis, HybridAxis))
    )

    lon = coords["lon"].to_numpy(dtype=float)
    lat = coords["lat"].to_numpy(dtype=float)
    plan = _build_horizontal_plan(recordset.grid, lon, lat, method=method)

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

    targets = coords["z"].to_numpy(dtype=float)
    sampled = {
        variable: _sample_variable(
            recordset,
            variable,
            targets,
            z_kind=z_kind,
            plan=plan,
            surface_pressure=surface_pressure,
            terrain=terrain,
        )
        for variable in variable_names
    }
    _rotate_winds(sampled, file.grid, lon, lat, wind_pairs)
    return sampled


def _open_files(
    files: FileLike | Sequence[FileLike],
) -> tuple[tuple[File, ...], list[File]]:
    """
    Normalize *files* to a tuple of open read-mode Files.

    Accepts a single open :class:`~arlmet.file.File` or path, or a sequence of
    them. Paths are opened here; already-open Files are passed through. Returns
    ``(files, opened)`` where *opened* lists only the Files this function
    created and is therefore responsible for closing.
    """
    from arlmet.file import File

    if isinstance(files, (File, str, os.PathLike)):
        items: list[FileLike] = [files]
    else:
        items = list(files)
    if not items:
        raise ValueError("sample_points requires at least one input file.")

    handles: list[File] = []
    opened: list[File] = []
    try:
        for item in items:
            if isinstance(item, File):
                handles.append(item)
            else:
                handle = File(item)  # read mode
                handles.append(handle)
                opened.append(handle)
    except Exception:
        for handle in opened:
            handle.close()
        raise

    return tuple(handles), opened


def _map_times(files: Sequence[File]) -> dict[pd.Timestamp, File]:
    """Map each valid time to the one input File that contains it."""
    time_map: dict[pd.Timestamp, File] = {}
    for file in files:
        for file_time in file.times:
            file_time = ensure_timestamp(file_time)
            if file_time in time_map:
                raise ValueError(
                    f"Multiple input files contain meteorology for time {file_time}."
                )
            time_map[file_time] = file
    return time_map


def sample_points(
    files: FileLike | Sequence[FileLike],
    points: pd.DataFrame | Mapping[str, Any],
    variables: str | Iterable[str],
    *,
    time: pd.Timestamp | str | None = None,
    z_kind: ZKind = "pressure",
    method: SampleMethod = "linear",
    earth_relative: bool = False,
) -> pd.DataFrame:
    """
    Sample meteorological variables at arbitrary (lon, lat, z, time) points.

    Accepts a single ARL file or a sequence of files spanning different time
    periods. Each may be an open :class:`~arlmet.file.File` or a path to an
    ARL file; paths are opened (read mode) and closed automatically, while
    already-open Files are left open for the caller to manage. Each point is
    sampled from the file that contains its valid time.

    Parameters
    ----------
    files : File, path-like, or sequence of File or path-like
        A single open :class:`~arlmet.file.File` or path, or a sequence of
        Files and/or paths. Each valid time must appear in at most one file.
    points : pandas.DataFrame or mapping
        Table-like object with ``lon``, ``lat`` (degrees), and ``z`` columns,
        and optionally a ``time`` column. Any other columns are carried
        through to the result unchanged.
    variables : str or iterable of str
        One or more ARL field names (e.g. ``"TEMP"``, ``"UWND"``), or
        ``"pressure"`` for the virtual pressure variable. A name must not
        collide with an existing column of ``points``.
    time : pandas.Timestamp or str, optional
        One valid time for every point, used when ``points`` has no ``time``
        column. Passing both raises ``ValueError``. When neither is given, the
        input file(s) must contain exactly one valid time, which is used.
    z_kind : {"pressure", "native", "agl", "msl"}, default "pressure"
        Vertical coordinate system of the ``z`` values. See
        :meth:`arlmet.File.sample_points` for the method each vertical
        coordinate system uses and the fields it requires.
    method : {"linear", "nearest"}, default "linear"
        Horizontal interpolation: bilinear or nearest grid point.
    earth_relative : bool, default False
        Rotate sampled wind pairs (``UWND``/``VWND``, ``U10M``/``V10M``) from
        the grid axes to east/north, using the meridian convergence of each
        file's grid. Both components of a pair must be requested. No effect on
        lat/lon grids.

    Returns
    -------
    pandas.DataFrame
        Copy of ``points`` (all columns and the index preserved) with one
        added column per requested variable. Points outside the grid or the
        vertical range are NaN.

    Raises
    ------
    ValueError
        If a required column is missing, ``time`` is given alongside a
        ``time`` column, no time is given and the inputs hold more than one
        time, a point time is in none of the files, a time is in more than one
        file, a variable name collides with a column of ``points``, ``z_kind``
        or ``method`` is invalid, or a field that ``z_kind`` requires is
        missing.

    Examples
    --------
    >>> import pandas as pd
    >>> import arlmet
    >>> points = pd.DataFrame(
    ...     {"lon": [-111.9], "lat": [40.7], "z": [850.0], "time": ["2024-07-18 00:00"]}
    ... )
    >>> arlmet.sample_points("met.arl", points, ["UWND", "VWND"])

    Sample across files spanning different times by passing their paths:

    >>> arlmet.sample_points(["met_00.arl", "met_06.arl"], points, ["UWND", "VWND"])
    """
    if z_kind not in _Z_KINDS:
        raise ValueError("z_kind must be one of 'native', 'pressure', 'agl', or 'msl'.")
    if method not in _METHODS:
        raise ValueError("method must be 'linear' or 'nearest'.")
    variable_names = _normalize_variables(variables)
    wind_pairs = _check_wind_pairs(variable_names) if earth_relative else []

    handles, opened = _open_files(files)
    try:
        time_map = _map_times(handles)
        implied_time = next(iter(time_map)) if len(time_map) == 1 else None
        result, coords = _normalize_points(
            points, variable_names, time=time, implied_time=implied_time
        )

        codes, point_times = pd.factorize(coords["time"])
        missing_times = [t for t in point_times if ensure_timestamp(t) not in time_map]
        if missing_times:
            available = sorted(time_map)
            coverage = f"{available[0]} to {available[-1]}" if available else "no times"
            raise ValueError(
                "No input file contains the requested point times: "
                + ", ".join(str(ensure_timestamp(t)) for t in missing_times)
                + f". The input file(s) cover {coverage}."
            )

        columns = {
            name: np.full(len(coords), np.nan, dtype=float) for name in variable_names
        }
        for code, point_time in enumerate(point_times):
            point_time = ensure_timestamp(point_time)
            rows = np.flatnonzero(codes == code)
            sampled = _sample_points_from_file(
                time_map[point_time],
                point_time,
                coords.iloc[rows],
                variable_names,
                z_kind=z_kind,
                method=method,
                wind_pairs=wind_pairs,
            )
            for name, values in sampled.items():
                columns[name][rows] = values

        for name, values in columns.items():
            result[name] = values
        return result
    finally:
        for handle in opened:
            handle.close()
