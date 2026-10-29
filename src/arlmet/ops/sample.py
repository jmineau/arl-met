"""Point-sampling helpers for ARL meteorology files."""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
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
    WrfAxis,
    _hysplit_z_agl,
)

if TYPE_CHECKING:
    from arlmet.file import File
    from arlmet.record import DataRecord
    from arlmet.recordset import RecordSet

__all__ = ["sample_points"]

SURFACE_VARIABLES = {"PRSS", "SHGT"}

#: On sigma, pressure-level, hybrid and WRF files, the surface field that gives
#: an upper-level variable its value at the ground, as HYSPLIT's ``prfsig``,
#: ``prfprs`` and ``prfecm`` start their columns. ``TEMP`` takes the column's
#: ground temperature (``T02M``, ``TMPS``, else the first level's brought down
#: adiabatically), vertical motion is zero, and any other variable keeps its
#: first level's value.
_GROUND_FIELDS: dict[str, str] = {
    "PRES": "PRSS",
    "HGTS": "SHGT",
    "UWND": "U10M",
    "VWND": "V10M",
}
_ZERO_AT_GROUND = frozenset({"WWND", "DZDT"})
#: Variables holding pressure, interpolated as ln p along a :class:`_Column`, as
#: HYSPLIT's ``prfsig``, ``prfprs`` and ``prfecm`` do (``prfter`` is linear).
_PRESSURE_VARIABLES = frozenset({"PRES", "pressure"})

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


def _interp_column(
    values: npt.NDArray[Any], coords: npt.NDArray[Any], target: float
) -> np.float32:
    """
    Interpolate *values* at the height *target* along a column, in level order.

    *coords* are the column's heights from the ground (index 0) up. As
    HYSPLIT's ``prfsig``/``prfprs``/``prfecm`` walk it, *target* falls in the
    first layer, in level order, whose top is at or above it, and is
    interpolated between that layer's bottom and top. Where heights increase
    this is plain linear interpolation; where a level lies below the ground (a
    sigma file whose first ``PRES`` exceeds ``PRSS``) the ground's value is
    passed over and the column runs from that level up. NaN below the ground
    or above the top.
    """
    mask = np.isfinite(values) & np.isfinite(coords)
    if mask.sum() < 2:
        return np.float32(np.nan)
    v = np.asarray(values[mask], dtype=float)
    c = np.asarray(coords[mask], dtype=float)
    target = float(target)
    if not target >= c[0]:
        return np.float32(np.nan)
    above = np.flatnonzero(c[1:] >= target)
    if above.size == 0:
        return np.float32(np.nan)
    k = int(above[0]) + 1
    lo, hi = c[k - 1], c[k]
    if hi == lo:
        return np.float32(v[k])
    return np.float32(v[k - 1] + (target - lo) / (hi - lo) * (v[k] - v[k - 1]))


def _interp_profiles(
    values: npt.NDArray[Any],
    coords: npt.NDArray[Any],
    targets: npt.NDArray[Any],
    *,
    column: bool = False,
) -> npt.NDArray[Any]:
    """
    Vectorised _interp_profile over n_points rows.

    Parameters
    ----------
    values : (n_points, n_levels)
    coords : (n_levels,) shared across all points, or (n_points, n_levels) per-point
    targets : (n_points,)
    column : heights along a column from the ground up, walked in level order
        (:func:`_interp_column`) instead of sorted
    """
    out = np.full(values.shape[0], np.nan, dtype=np.float32)
    shared_coords = coords.ndim == 1
    interp = _interp_column if column else _interp_profile
    for i in range(values.shape[0]):
        profile_coords = coords if shared_coords else coords[i]
        out[i] = interp(values[i], profile_coords, targets[i])
    return out


def _level_profiles(
    recordset: RecordSet,
    variable: str,
    plan: HorizontalSamplePlan,
    levels: Sequence[int],
) -> npt.NDArray[Any] | None:
    """
    Sample *variable* at *levels*: ``(n_points, n_levels)``, NaN where a level lacks it.

    Returns None when the time step has no *variable* record at all.
    """
    by_level = {r.level: r for r in recordset.records if r.variable == variable}
    if not by_level:
        return None
    n_points = len(plan.inside)
    out = np.full((n_points, len(levels)), np.nan, dtype=np.float32)
    for pos, level in enumerate(levels):
        if level in by_level:
            out[:, pos] = _sample_record(by_level[level], plan)
    return out


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
    terrain: npt.NDArray[Any] | None,
    column: _Column | None,
) -> npt.NDArray[Any]:
    """
    Per-point vertical coordinate at each *level*, expressed in the *z_kind* system.

    Returns an ``(n_points, n_levels)`` array giving the coordinate that a target
    *z* (in the same *z_kind* units) is interpolated against.

    Dispatches to the vertical axis subclass methods, matching HYSPLIT:

    - flag=2 (pressure): stored levels for pressure; for heights the *column*
      (the ground at level 0, then HGTS, levels at or below the ground left out).
    - flag=1/4/5 (sigma/hybrid/WRF): the *column* (PRSS at level 0, the ground;
      PRES on sigma files that have it and on WRF files, else the axis's
      pressure; HYSPLIT's heights).
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
            if column is not None:  # sampling at heights: PRSS at the ground
                return column.pressure[:, level_list]
            return _broadcast(axis.to_pressure()[level_list])
        if isinstance(axis, (SigmaAxis, HybridAxis, WrfAxis)):
            assert column is not None
            return column.pressure[:, level_list]
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

    if isinstance(axis, (PressureAxis, SigmaAxis, HybridAxis, WrfAxis)):
        # HYSPLIT PRFPRS: HGTS above the ground. PRFSIG/PRFECM: hypsometric from
        # the ground with virtual temperature; a WRF file without hybrid
        # coefficients goes to PRFSIG too.
        assert column is not None
        agl = column.z_agl[:, level_list]
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


class _Column:
    """
    The vertical column of one sigma, pressure-level, hybrid or WRF time step.

    Built at the sample points as HYSPLIT's ``prfsig`` (sigma, flag 1, and WRF,
    flag 5, without WRF's hybrid coefficients), ``prfprs`` (pressure, flag 2)
    and ``prfecm`` (hybrid, flag 4) build theirs, and shared by every variable
    sampled at that time. Level 0 is the ground:
    0 m above it, at the surface pressure ``PRSS``. On a pressure-level file a
    level at or below the ground has no height (NaN), so a profile runs from
    the ground to the first level above it, as ``prfprs`` skips such levels.
    Each field is read only when first needed, and every array is
    ``(n_points, n_levels)`` over all the axis's levels, or ``(n_points,)`` for
    the ground.
    """

    def __init__(
        self,
        recordset: RecordSet,
        plan: HorizontalSamplePlan,
        *,
        terrain: npt.NDArray[Any] | None = None,
    ) -> None:
        self.recordset = recordset
        self.plan = plan
        self.axis = recordset.vertical_axis
        self.upper = list(range(1, len(self.axis.levels)))
        self.terrain = terrain

    def _surface(self, variable: str) -> npt.NDArray[Any] | None:
        """Sample the surface field *variable*, or None if the time step lacks it."""
        profile = _level_profiles(self.recordset, variable, self.plan, [0])
        return None if profile is None else profile[:, 0]

    def _upper(self, variable: str) -> npt.NDArray[Any] | None:
        """Sample *variable* at every level above ground, or None if absent."""
        return _level_profiles(self.recordset, variable, self.plan, self.upper)

    @cached_property
    def surface_pressure(self) -> npt.NDArray[Any]:
        """``PRSS`` [hPa] at each point."""
        prss = self._surface("PRSS")
        if prss is None:
            raise ValueError(
                "Surface pressure field PRSS is required but not available at "
                f"time {self.recordset.time}."
            )
        return prss

    @cached_property
    def pressure(self) -> npt.NDArray[Any]:
        """
        Pressure [hPa] of each level.

        ``PRSS`` at the ground, then the ``PRES`` field when a sigma time step
        has one (HYSPLIT's ``ZFLG``), else the nominal pressure of the axis. A
        WRF time step needs ``PRES``: its eta levels give pressure only with
        WRF's hybrid coefficients, which the file does not hold.
        """
        sp = self.surface_pressure
        if isinstance(self.axis, PressureAxis):
            p = np.tile(self.axis.to_pressure(), (len(sp), 1))
            p[:, 0] = sp
            return p
        if isinstance(self.axis, WrfAxis):
            pres = self._upper("PRES")
            if pres is None:
                raise ValueError(
                    "WRF (flag=5) files require PRES records for pressure and "
                    "AGL/MSL sampling."
                )
            return np.column_stack([sp, pres])
        assert isinstance(self.axis, (SigmaAxis, HybridAxis))
        p = self.axis.to_pressure(surface_pressure=sp)
        pres = self._upper("PRES") if isinstance(self.axis, SigmaAxis) else None
        if pres is not None:
            p[:, 1:] = pres
        p[:, 0] = sp
        return p

    @cached_property
    def z_agl(self) -> npt.NDArray[Any]:
        """Height above ground [m] of each level, 0 at the ground (HYSPLIT's heights)."""
        if isinstance(self.axis, PressureAxis):
            return self._hgts_agl
        z = np.zeros_like(self.pressure)
        if not self.upper:
            return z
        temp = self._temperature
        p = self.pressure
        z[:, 1:] = _hysplit_z_agl(
            p[:, 1:],
            p[:, 0],
            temp,
            first_sigma=self._first_sigma,
            surface_temperature=self._surface_temperature,
            specific_humidity=self._upper("SPHU"),
            relative_humidity=self._upper("RELH"),
        )
        return z

    @cached_property
    def _hgts_agl(self) -> npt.NDArray[Any]:
        """
        ``HGTS - SHGT`` [m] of each level of a pressure-level file, 0 at the ground.

        A level at or below the ground is NaN: HYSPLIT's ``prfprs`` puts no
        output level there and keeps the ground as the bottom of the profile.
        """
        hgts = self._upper("HGTS")
        if hgts is None:
            raise ValueError(
                "Pressure-level (flag=2) files require HGTS records for AGL/MSL sampling."
            )
        terrain = _resolve_terrain(self.recordset, self.plan, self.terrain)
        agl = hgts - terrain[:, None]
        z = np.zeros((len(terrain), len(self.axis.levels)), dtype=np.float32)
        z[:, 1:] = np.where(agl > 0.0, agl, np.nan)
        return z

    @cached_property
    def ground_temperature(self) -> npt.NDArray[Any]:
        """Temperature [K] at the ground, as HYSPLIT's ``PRFSFCT`` finds it."""
        if self._surface_temperature is not None:
            return self._surface_temperature
        return self._temperature[:, 0] * self._first_sigma**-0.286

    @cached_property
    def _temperature(self) -> npt.NDArray[Any]:
        """``TEMP`` [K] at every level above ground."""
        temp = self._upper("TEMP")
        if temp is None:
            raise ValueError(
                f"{self.axis.coord_system} (flag={self.axis.flag}) files require "
                "TEMP for AGL/MSL sampling."
            )
        return temp

    @cached_property
    def _surface_temperature(self) -> npt.NDArray[Any] | None:
        """``T02M``, else ``TMPS``, else None, as HYSPLIT's ``metini`` sets ``TFLG``."""
        t02m = self._surface("T02M")
        return t02m if t02m is not None else self._surface("TMPS")

    @cached_property
    def _first_sigma(self) -> npt.NDArray[Any]:
        """
        Sigma of the first level above ground.

        It brings that level's temperature down to the ground: the stored sigma
        (``PSG(1)``) on sigma files and the stored eta on WRF files, its nominal pressure over ``PRSS``
        (``ESG(1)``) on hybrid files, and level 1's pressure over ``PRSS``
        (``PSG(1)/P0``) on pressure-level files, where that level can be below
        the ground.
        """
        if isinstance(self.axis, (SigmaAxis, WrfAxis)):
            return np.full(len(self.plan.inside), self.axis.levels[1])
        if isinstance(self.axis, PressureAxis):
            return self.axis.levels[1] / self.surface_pressure
        sp = self.surface_pressure
        return self.axis.to_pressure(surface_pressure=sp)[:, 1] / sp

    def ground_values(
        self, variable: str, first_level: npt.NDArray[Any]
    ) -> npt.NDArray[Any]:
        """Value of upper-level *variable* at the ground, given its first level's value."""
        if variable == "TEMP":
            return self.ground_temperature
        if variable in _ZERO_AT_GROUND:
            return np.zeros_like(first_level)
        surface = _GROUND_FIELDS.get(variable)
        values = None if surface is None else self._surface(surface)
        return first_level if values is None else values


def _sample_variable(
    recordset: RecordSet,
    variable: str,
    targets: npt.NDArray[Any],
    *,
    z_kind: ZKind,
    plan: HorizontalSamplePlan,
    terrain: npt.NDArray[Any] | None,
    column: _Column | None,
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
        ``'agl'`` — target is meters above ground level;
        ``'msl'`` — target is meters above mean sea level.
    terrain :
        (n_points,) SHGT in meters. Required for AGL sampling with non-terrain
        vertical axes, and for MSL sampling with terrain-following (flag=3) axes.
    column :
        The column of this time step on sigma, hybrid and WRF files (flag
        1/4/5), and on pressure-level files (flag 2) sampled at heights; else
        None. Level
        0 is then the ground, and a variable without a record there takes its
        ground value (:meth:`_Column.ground_values`).

    Returns
    -------
    np.ndarray of shape (n_points,) float32, NaN for out-of-range or off-grid points.
    Along a *column* pressure (``PRES`` or the virtual pressure) is
    interpolated as ln p, as HYSPLIT's ``prfsig``, ``prfprs`` and ``prfecm``
    do; otherwise (terrain-following files, whose ``prfter`` is linear, and
    the native levels of pressure-level files, whose level 0 holds no
    pressure) linearly.
    """
    axis = recordset.vertical_axis
    log_p = column is not None and variable in _PRESSURE_VARIABLES
    # At heights a column is walked in level order, as HYSPLIT does (#106).
    walk = column is not None and z_kind in ("agl", "msl")

    def _interp(values: npt.NDArray[Any], coords: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Interpolate *values* to the targets, as ln p for pressure."""
        if not log_p:
            return _interp_profiles(values, coords, targets, column=walk)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_values = np.log(values)
        return np.exp(_interp_profiles(log_values, coords, targets, column=walk))

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
            terrain=terrain,
            column=column,
        )
        coords = _vertical_coords(
            recordset,
            all_levels,
            z_kind=z_kind,
            plan=plan,
            terrain=terrain,
            column=column,
        )
        return _interp(values, coords)

    # --- data variable ---
    samples, levels = _variable_profiles(recordset, variable, plan)
    if len(levels) == 1 or variable in SURFACE_VARIABLES:
        # Single-level or surface field — return directly without vertical interpolation.
        return samples[:, 0]

    if column is not None and levels[0] != 0:
        # Level 0 is the ground: start the profile there, as HYSPLIT does.
        ground = column.ground_values(variable, samples[:, 0])
        samples = np.column_stack([ground, samples])
        levels = (0, *levels)

    coords = _vertical_coords(
        recordset,
        levels,
        z_kind=z_kind,
        plan=plan,
        terrain=terrain,
        column=column,
    )
    return _interp(samples, coords)


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
        - ``'pressure'``: hPa. Sigma (flag 1) uses PRES when the file has it,
          else converts from PRSS, as hybrid (flag 4) does; pressure (flag 2)
          uses the stored levels; terrain-following (flag 3) raises
          ``ValueError``; WRF (flag 5) uses PRES.
        - ``'agl'``: meters above ground. Sigma/hybrid/WRF integrate
          hypsometrically from the ground (PRSS, T02M) through those
          pressures with the virtual temperature (TEMP and SPHU or RELH), as
          HYSPLIT's ``prfsig``/``prfecm`` do (WRF through PRES); pressure uses
          HGTS - SHGT (both required) from the ground
          up, leaving out the levels at or below the ground, as HYSPLIT's
          ``prfprs`` does; terrain-following uses the stored levels.
        - ``'msl'``: meters above mean sea level. Sigma/hybrid/WRF add SHGT to
          the hypsometric AGL height; pressure uses HGTS, from the ground
          (SHGT) up (both required); terrain-following adds SHGT to the stored
          levels.
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
    heights = z_kind in ("agl", "msl")
    # flag=2 heights need terrain for the ground and HGTS - SHGT; flag=3 MSL
    # needs terrain; flag=1/4/5 MSL needs terrain for hypsometric AGL + SHGT.
    need_terrain = (heights and isinstance(axis, PressureAxis)) or (
        z_kind == "msl"
        and isinstance(axis, (TerrainAxis, SigmaAxis, HybridAxis, WrfAxis))
    )

    lon = coords["lon"].to_numpy(dtype=float)
    lat = coords["lat"].to_numpy(dtype=float)
    plan = _build_horizontal_plan(recordset.grid, lon, lat, method=method)

    terrain = None
    if need_terrain:
        terrain_record = _surface_record(recordset, "SHGT")
        if terrain_record is None:
            raise ValueError(
                f"Terrain field SHGT is required at time {recordset.time}."
            )
        terrain = _sample_record(terrain_record, plan)

    # Sigma, hybrid and WRF files always start their column at the ground;
    # pressure files do when sampled at heights (at pressures the stored levels
    # are used).
    column = (
        _Column(recordset, plan, terrain=terrain)
        if isinstance(axis, (SigmaAxis, HybridAxis, WrfAxis))
        or (heights and isinstance(axis, PressureAxis))
        else None
    )
    targets = coords["z"].to_numpy(dtype=float)
    sampled = {
        variable: _sample_variable(
            recordset,
            variable,
            targets,
            z_kind=z_kind,
            plan=plan,
            terrain=terrain,
            column=column,
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
    >>> arlmet.sample_points("met.arl", points, ["UWND", "VWND"]).round(1)
         lon   lat      z              time  UWND  VWND
    0 -111.9  40.7  850.0  2024-07-18 00:00   6.2  -3.8

    Sample across files spanning different times by passing their paths:

    >>> arlmet.sample_points(
    ...     ["met_00.arl", "met_06.arl"], points, ["UWND", "VWND"]
    ... ).round(1)
         lon   lat      z              time  UWND  VWND
    0 -111.9  40.7  850.0  2024-07-18 00:00   6.2  -3.8
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
