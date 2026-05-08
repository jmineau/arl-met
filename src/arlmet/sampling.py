"""Point-sampling helpers for ARL meteorology files."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from arlmet.grid import Grid, GridWindow
from arlmet.surface import Surface
from arlmet.vertical import VerticalAxis

if TYPE_CHECKING:
    from arlmet.file import File
    from arlmet.record import DataRecord
    from arlmet.recordset import RecordSet


SURFACE_VARIABLES = {"PRSS", "SHGT"}


@dataclass(frozen=True)
class HorizontalSamplePlan:
    method: str
    window: GridWindow | None
    inside: np.ndarray
    x: np.ndarray
    y: np.ndarray
    x0: np.ndarray
    x1: np.ndarray
    y0: np.ndarray
    y1: np.ndarray
    wx: np.ndarray
    wy: np.ndarray


def _coerce_points(points: Any) -> pd.DataFrame:
    if isinstance(points, pd.DataFrame):
        return points.copy()
    return pd.DataFrame(points)


def _normalize_points(
    points: Any,
    *,
    require_time: bool,
    require_z: bool,
    default_time: pd.Timestamp | None = None,
) -> pd.DataFrame:
    df = _coerce_points(points)

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
    elif require_z:
        raise ValueError(
            "Point sampling requires a 'z' column for native, pressure, agl, or msl queries."
        )

    return normalized


def _normalize_variables(variables: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(variables, str):
        names = (variables,)
    else:
        names = tuple(str(name) for name in variables)

    if not names:
        raise ValueError("variables must include at least one variable name.")
    return names


def _record_levels(recordset: RecordSet, variable: str) -> OrderedDict[int, DataRecord]:
    records = OrderedDict(
        sorted(
            (
                record.level,
                record,
            )
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
        x=x,
        y=y,
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
        wx=wx,
        wy=wy,
    )


def _sample_field(field: np.ndarray, plan: HorizontalSamplePlan) -> np.ndarray:
    result = np.full(plan.inside.shape, np.nan, dtype=np.float32)
    if plan.window is None or not plan.inside.any():
        return result

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
    if plan.window is None:
        return np.full(plan.inside.shape, np.nan, dtype=np.float32)
    field = record.read(window=plan.window)
    return _sample_field(np.asarray(field, dtype=np.float32), plan)


def _delta_z_for_pressure_levels(pressure_levels: np.ndarray) -> np.ndarray:
    pressure_levels = np.asarray(pressure_levels, dtype=float)
    delta_z = np.zeros_like(pressure_levels, dtype=float)

    mask1 = pressure_levels >= 100.0
    if np.any(mask1):
        idx1 = np.clip(
            (pressure_levels[mask1] // 100).astype(int),
            0,
            len(VerticalAxis.Z_PHI1) - 1,
        )
        delta_z[mask1] = VerticalAxis.Z_PHI1[idx1]

    mask2 = ~mask1
    if np.any(mask2):
        idx2 = np.clip(
            (pressure_levels[mask2] // 10).astype(int),
            0,
            len(VerticalAxis.Z_PHI2) - 1,
        )
        delta_z[mask2] = VerticalAxis.Z_PHI2[idx2]

    return delta_z


def _interp_profile(values: np.ndarray, coords: np.ndarray, target: float) -> np.float32:
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
    out = np.full(values.shape[0], np.nan, dtype=np.float32)
    shared_coords = coords.ndim == 1
    for i in range(values.shape[0]):
        profile_coords = coords if shared_coords else coords[i]
        out[i] = _interp_profile(values[i], profile_coords, targets[i])
    return out


def _sample_surface_support(
    recordset: RecordSet,
    plan: HorizontalSamplePlan,
    *,
    require_terrain: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    surface_record = _surface_record(recordset, "PRSS")
    if surface_record is None:
        surface_pressure = np.full(plan.inside.shape, Surface.DEFAULT_PRESSURE, dtype=np.float32)
    else:
        surface_pressure = _sample_record(surface_record, plan)

    terrain = None
    if require_terrain:
        terrain_record = _surface_record(recordset, "SHGT")
        if terrain_record is None:
            raise ValueError(
                f"Terrain field SHGT is required for z_kind='msl' at time {recordset.time}."
            )
        terrain = _sample_record(terrain_record, plan)

    return surface_pressure, terrain


def _pressure_profile_for_levels(axis: VerticalAxis, levels: Sequence[int]) -> np.ndarray:
    pressure = axis.pressure()
    if pressure is None:
        raise NotImplementedError(
            "Point sampling with derived vertical coordinates requires a pressure-like vertical axis."
        )
    return np.asarray(pressure, dtype=float)[list(levels)]


def _variable_profiles(
    recordset: RecordSet,
    variable: str,
    plan: HorizontalSamplePlan,
) -> tuple[np.ndarray, tuple[int, ...]]:
    records = _record_levels(recordset, variable)
    levels = tuple(records.keys())
    samples = np.full((len(plan.inside), len(levels)), np.nan, dtype=np.float32)
    for pos, level in enumerate(levels):
        samples[:, pos] = _sample_record(records[level], plan)
    return samples, levels


def _sample_special_pressure(
    axis: VerticalAxis,
    targets: np.ndarray,
    *,
    z_kind: str,
    surface_pressure: np.ndarray | None,
    terrain: np.ndarray | None,
) -> np.ndarray:
    if z_kind == "pressure":
        return targets.astype(np.float32, copy=False)

    pressure_levels = np.asarray(axis.heights, dtype=float)
    if z_kind == "native":
        native_levels = np.arange(len(axis.heights), dtype=float)
        return _interp_profiles(
            np.broadcast_to(pressure_levels, (len(targets), len(pressure_levels))),
            native_levels,
            targets,
        )

    if axis.flag != 2:
        raise NotImplementedError(
            f"sample_points(z_kind={z_kind!r}) currently only supports pressure-level vertical axes."
        )

    delta_z = _delta_z_for_pressure_levels(pressure_levels)
    coords = np.broadcast_to(pressure_levels, (len(targets), len(pressure_levels))).copy()
    if z_kind == "agl":
        if surface_pressure is None:
            raise ValueError("surface_pressure is required for z_kind='agl'.")
        vertical_coords = np.maximum(
            (surface_pressure[:, None] - pressure_levels[None, :]) * delta_z[None, :],
            0.0,
        )
    elif z_kind == "msl":
        if surface_pressure is None or terrain is None:
            raise ValueError("surface_pressure and terrain are required for z_kind='msl'.")
        vertical_coords = np.maximum(
            (surface_pressure[:, None] - pressure_levels[None, :]) * delta_z[None, :],
            0.0,
        ) + terrain[:, None]
    else:
        raise ValueError(f"Unsupported z_kind {z_kind!r}.")

    return _interp_profiles(coords, vertical_coords, targets)


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
    axis = recordset.vertical_axis

    if variable == "pressure":
        return _sample_special_pressure(
            axis,
            targets,
            z_kind=z_kind,
            surface_pressure=surface_pressure,
            terrain=terrain,
        )

    samples, levels = _variable_profiles(recordset, variable, plan)
    if len(levels) == 1 or variable in SURFACE_VARIABLES:
        return samples[:, 0]

    if z_kind == "native":
        coords = np.asarray(levels, dtype=float)
        return _interp_profiles(samples, coords, targets)

    if z_kind == "pressure":
        coords = _pressure_profile_for_levels(axis, levels)
        return _interp_profiles(samples, coords, targets)

    if z_kind not in {"agl", "msl"}:
        raise ValueError("z_kind must be one of 'native', 'pressure', 'agl', or 'msl'.")
    if axis.flag != 2:
        raise NotImplementedError(
            f"sample_points(z_kind={z_kind!r}) currently only supports pressure-level vertical axes."
        )
    if surface_pressure is None:
        raise ValueError(
            f"surface pressure is required to sample z_kind={z_kind!r}."
        )

    pressure_levels = np.asarray(axis.heights, dtype=float)[list(levels)]
    delta_z = _delta_z_for_pressure_levels(pressure_levels)
    coords = np.maximum(
        (surface_pressure[:, None] - pressure_levels[None, :]) * delta_z[None, :],
        0.0,
    )
    if z_kind == "msl":
        if terrain is None:
            raise ValueError("terrain is required to sample z_kind='msl'.")
        coords = coords + terrain[:, None]

    return _interp_profiles(samples, coords, targets)


def terrain_from_file(file: File, time: pd.Timestamp | str | None = None) -> np.ndarray:
    if time is None:
        if len(file.times) != 1:
            raise ValueError(
                "terrain() requires an explicit time when the file contains multiple times."
            )
        time = file.times[0]

    recordset = file[pd.Timestamp(time)]
    record = _surface_record(recordset, "SHGT")
    if record is None:
        raise ValueError(f"Terrain field SHGT is not available at time {recordset.time}.")
    return np.asarray(record.read(), dtype=np.float32)


def sample_points_from_file(
    file: File,
    points: Any,
    variables: str | Iterable[str],
    *,
    time: pd.Timestamp | str | None = None,
    z_kind: str = "pressure",
    method: str = "linear",
) -> pd.DataFrame:
    variable_names = _normalize_variables(variables)
    require_time = time is None and len(file.times) != 1
    default_time = pd.Timestamp(time) if time is not None else (file.times[0] if len(file.times) == 1 else None)
    normalized = _normalize_points(
        points,
        require_time=require_time,
        require_z=True,
        default_time=default_time,
    )

    if z_kind not in {"native", "pressure", "agl", "msl"}:
        raise ValueError("z_kind must be one of 'native', 'pressure', 'agl', or 'msl'.")

    result = normalized.copy()
    for variable in variable_names:
        result[variable] = np.nan

    for sample_time, index in result.groupby("time").groups.items():
        recordset = file[pd.Timestamp(sample_time)]
        subset = result.loc[index]
        plan = _build_horizontal_plan(
            recordset.grid,
            subset["lon"].to_numpy(dtype=float),
            subset["lat"].to_numpy(dtype=float),
            method=method,
        )

        need_surface_pressure = z_kind in {"agl", "msl"} or "pressure" in variable_names
        need_terrain = z_kind == "msl"
        surface_pressure = terrain = None
        if need_surface_pressure or need_terrain:
            surface_pressure, terrain = _sample_surface_support(
                recordset,
                plan,
                require_terrain=need_terrain,
            )

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
    # File is a Mapping; a plain sequence of Files is not — use that to distinguish.
    if isinstance(source, Mapping):
        return (source,)  # single File
    return tuple(source)


def terrain(
    source: File | Sequence[File],
    *,
    time: pd.Timestamp | str | None = None,
) -> np.ndarray:
    files = _normalize_sources(source)
    if len(files) == 1:
        return terrain_from_file(files[0], time=time)

    if time is None:
        raise ValueError(
            "terrain() requires time=... when multiple sources are provided."
        )
    target_time = pd.Timestamp(time)
    matches = [file for file in files if target_time in file.times]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one source containing time {target_time}, found {len(matches)}."
        )
    return terrain_from_file(matches[0], time=target_time)


def sample_points(
    source: File | Sequence[File],
    points: Any,
    variables: str | Iterable[str],
    *,
    time: pd.Timestamp | str | None = None,
    z_kind: str = "pressure",
    method: str = "linear",
) -> pd.DataFrame:
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
        require_z=True,
        default_time=pd.Timestamp(time) if time is not None else None,
    )
    time_map: dict[pd.Timestamp, File] = {}
    for file in files:
        for sample_time in file.times:
            if sample_time in time_map:
                raise ValueError(
                    f"Multiple sources contain meteorology for time {sample_time}."
                )
            time_map[sample_time] = file

    missing_times = sorted(set(normalized["time"]) - set(time_map))
    if missing_times:
        raise ValueError(
            "No source contains the requested point times: "
            + ", ".join(str(pd.Timestamp(t)) for t in missing_times)
        )

    pieces: list[pd.DataFrame] = []
    for sample_time, index in normalized.groupby("time").groups.items():
        piece = sample_points_from_file(
            time_map[pd.Timestamp(sample_time)],
            normalized.loc[index],
            variables,
            time=pd.Timestamp(sample_time),
            z_kind=z_kind,
            method=method,
        )
        pieces.append(piece)

    return pd.concat(pieces).reindex(normalized.index)
