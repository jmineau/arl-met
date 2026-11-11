"""
A variable's profile along each column, and reading it at a target height, pressure or level.

At heights a column is walked from the ground up in level order, as HYSPLIT
does (:func:`_interp_column`, #106); at pressures and native levels the
levels are sorted by their coordinate (:func:`_interp_profile`). Pressure is
interpolated as ln p along a column, as HYSPLIT's ``prfsig``, ``prfprs`` and
``prfecm`` do (``prfter`` is linear).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt

from arlmet.vertical import TerrainAxis

from .fields import levels_of, sample_levels
from .horizontal import Plan

if TYPE_CHECKING:
    from arlmet._recordset import RecordSet
    from arlmet.vertical import VerticalAxis

    from .column import Column

__all__ = [
    "PRESSURE_VARIABLES",
    "ColumnZKind",
    "Profile",
    "interp_profiles",
    "sample_variable",
    "variable_profile",
]

#: The vertical coordinates a column is interpolated in: ``"msl"`` targets are
#: turned into heights above the ground at the point first.
ColumnZKind = Literal["native", "pressure", "agl"]

#: Fields that are the same at every height.
_SURFACE_VARIABLES = frozenset({"PRSS", "SHGT"})
#: Variables holding pressure, interpolated as ln p along a column.
PRESSURE_VARIABLES = frozenset({"PRES", "pressure"})

#: Rows interpolated at once by :func:`interp_profiles`, to bound its memory.
_INTERP_CHUNK = 65536


def _interp_profile(
    values: npt.NDArray[Any], coords: npt.NDArray[Any], targets: npt.NDArray[Any]
) -> npt.NDArray[Any]:
    """
    Interpolate each row of *values* linearly at its target, along that row's sorted *coords*.

    Per row, the levels with a finite value and coordinate are taken in
    increasing coordinate (of levels at one coordinate, the first in level
    order), as :func:`numpy.interp` takes them. NaN outside their range or
    where fewer than two levels have a value. *values* and *coords* are ``(rows,
    levels)``, *targets* ``(rows,)``.
    """
    v = np.asarray(values, dtype=float)
    t = np.asarray(targets, dtype=float)
    valid = np.isfinite(v) & np.isfinite(coords)
    c = np.where(valid, coords, np.nan).astype(float)
    order = np.argsort(c, axis=1, kind="stable")  # NaN last
    c, v = np.take_along_axis(c, order, axis=1), np.take_along_axis(v, order, axis=1)
    repeat = np.zeros(c.shape, dtype=bool)
    repeat[:, 1:] = c[:, 1:] == c[:, :-1]
    if repeat.any():  # keep the first of levels at one coordinate
        c = np.where(repeat, np.nan, c)
        order = np.argsort(c, axis=1, kind="stable")
        c, v = (
            np.take_along_axis(c, order, axis=1),
            np.take_along_axis(v, order, axis=1),
        )
    rows = np.arange(len(t))
    n = np.isfinite(c).sum(axis=1)  # distinct coordinates
    k = (c <= t[:, None]).sum(axis=1) - 1  # the last level at or below the target
    k1 = np.minimum(k + 1, c.shape[1] - 1)
    lo, hi = c[rows, k], c[rows, k1]
    v_lo, v_hi = v[rows, k], v[rows, k1]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (v_hi - v_lo) / (hi - lo) * (t - lo) + v_lo  # numpy.interp's formula
    at_top = (k == n - 1) & (lo == t)
    out = np.where(at_top, v_lo, out)
    out[(valid.sum(axis=1) < 2) | (k < 0) | ((k >= n - 1) & ~at_top)] = np.nan
    return out


def _interp_column(
    values: npt.NDArray[Any], coords: npt.NDArray[Any], targets: npt.NDArray[Any]
) -> npt.NDArray[Any]:
    """
    Interpolate each row of *values* at its target height, walking the row's column in level order.

    *coords* are each column's heights from the ground (level 0) up. As
    HYSPLIT's ``prfsig``/``prfprs``/``prfecm`` walk it, a target falls in the
    first layer, in level order and between levels with a finite value and
    height, whose top is at or above it, and is interpolated between that
    layer's bottom and top. Where heights increase this is plain linear
    interpolation; where a level lies below the ground (a sigma file whose
    first ``PRES`` exceeds ``PRSS``) the ground's value is passed over and the
    column runs from that level up. NaN below the first such level or above
    the top. *values* and *coords* are ``(rows, levels)``, *targets* ``(rows,)``.
    """
    v = np.asarray(values, dtype=float)
    c = np.asarray(coords, dtype=float)
    t = np.asarray(targets, dtype=float)
    valid = np.isfinite(v) & np.isfinite(c)
    rows = np.arange(len(t))
    level = np.arange(c.shape[1])
    first = np.argmax(valid, axis=1)
    # the level below each level, among those with a value
    below = np.maximum.accumulate(np.where(valid, level, -1), axis=1)
    below = np.concatenate([np.full((len(t), 1), -1), below[:, :-1]], axis=1)
    with np.errstate(invalid="ignore"):
        top = valid & (level > first[:, None]) & (c >= t[:, None])
        found = top.any(axis=1) & (t >= c[rows, first])
    k = np.argmax(top, axis=1)
    k0 = np.maximum(below[rows, k], 0)
    lo, hi = c[rows, k0], c[rows, k]
    v_lo, v_hi = v[rows, k0], v[rows, k]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(hi == lo, v_hi, v_lo + (t - lo) / (hi - lo) * (v_hi - v_lo))
    out[~found] = np.nan
    return out


def interp_profiles(
    values: npt.NDArray[Any],
    coords: npt.NDArray[Any],
    targets: npt.NDArray[Any],
    *,
    column: bool = False,
) -> npt.NDArray[Any]:
    """
    Interpolate each row of *values* at its target: ``(n_points,)`` float32.

    *values* are ``(n_points, n_levels)``, *coords* the same or ``(n_levels,)``
    for every point, *targets* ``(n_points,)``. *column*: heights along a
    column from the ground up, walked in level order (:func:`_interp_column`)
    instead of sorted (:func:`_interp_profile`).
    """
    interp = _interp_column if column else _interp_profile
    coords = np.broadcast_to(coords, values.shape)
    out = np.empty(values.shape[0], dtype=np.float32)
    for start in range(0, values.shape[0], _INTERP_CHUNK):
        rows = slice(start, start + _INTERP_CHUNK)
        out[rows] = interp(values[rows], coords[rows], targets[rows])
    return out


@dataclass(frozen=True)
class Profile:
    """
    One variable's profile at each row of a plan, to be read at any target.

    *values* and *coords* are ``(rows, levels)``. Pressure values are held as
    ln p (*log_p*) and read back as p; pressure coordinates are held as ln p
    too (*log_coords*), and the targets, in hPa, are turned into ln p to read
    them. *walk*: a column from the ground up, walked in level order. *fixed*:
    a field with one value per row (a surface or single-level field), the same
    at every target.
    """

    values: npt.NDArray[Any] | None = None
    coords: npt.NDArray[Any] | None = None
    log_p: bool = False
    log_coords: bool = False
    walk: bool = False
    fixed: npt.NDArray[Any] | None = None

    def at(self, targets: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Return the value at each row's target: ``(rows,)`` float32, NaN outside the profile."""
        if self.fixed is not None:
            return self.fixed
        assert self.values is not None and self.coords is not None
        if self.log_coords:
            with np.errstate(divide="ignore", invalid="ignore"):
                targets = np.log(np.asarray(targets, dtype=float))
        out = interp_profiles(self.values, self.coords, targets, column=self.walk)
        return np.exp(out) if self.log_p else out

    def top(self) -> npt.NDArray[Any]:
        """Return each row's highest coordinate with a value (NaN where none)."""
        assert self.values is not None and self.coords is not None
        coords = np.broadcast_to(self.coords, self.values.shape)
        valid = np.isfinite(self.values) & np.isfinite(coords)
        top = np.where(valid, coords, -np.inf).max(axis=1)
        return np.where(np.isfinite(top), top, np.nan)


def _vertical_coords(
    axis: VerticalAxis,
    levels: Sequence[int],
    *,
    z_kind: ColumnZKind,
    column: Column | None,
    n_points: int,
) -> npt.NDArray[Any]:
    """
    Each point's coordinate of each of *levels* in *z_kind*: ``(n_points, n_levels)``.

    Along a *column*, its pressures or heights (level 0 the ground). Without
    one the stored levels: those of a pressure-level file at native levels, and
    of a terrain-following file off HYSPLIT's internal levels, which has no
    pressure coordinate.
    """
    levels = list(levels)
    if column is not None and z_kind != "native":
        return (column.pressure if z_kind == "pressure" else column.z_agl)[:, levels]
    if z_kind == "native":
        per_level = np.asarray(levels, dtype=float)
    elif z_kind == "agl":
        per_level = axis.to_height_agl()[levels]
    elif isinstance(axis, TerrainAxis):
        raise ValueError(
            "z_kind='pressure' is not supported for terrain-following (flag=3) vertical axes."
        )
    else:
        per_level = axis.to_pressure()[levels]
    return np.broadcast_to(per_level, (n_points, len(levels)))


def variable_profile(
    recordset: RecordSet,
    variable: str,
    *,
    z_kind: ColumnZKind,
    plan: Plan,
    column: Column | None,
) -> Profile:
    """
    Build *variable*'s profile along the column of each row of *plan*.

    *variable* is an ARL field name or ``'pressure'``, the pressure itself
    (a terrain-following file's from its ``PRES`` field: NaN without one or
    off the internal levels). *z_kind* is the coordinate of the targets:
    ``'native'`` a (fractional) level index, ``'pressure'`` hPa, ``'agl'``
    meters above the ground. Along a *column* level 0 is the ground, and a
    variable without a record there takes its ground value
    (:meth:`~.column.Column.ground_values`). At pressures a level at or below
    the ground (its pressure at or above ``PRSS``) is left out, as
    pressure-level files' are at heights, so a target below the ground is NaN.
    """
    axis = recordset.vertical_axis
    n_points = plan.n_points

    if variable == "pressure":
        if isinstance(axis, TerrainAxis) and (
            column is None or not levels_of(recordset, "PRES")
        ):
            # Terrain-following files have no pressure coordinate of their own.
            return Profile(fixed=np.full(n_points, np.nan, dtype=np.float32))
        levels: tuple[int, ...] = tuple(range(len(axis.levels)))
        values = _vertical_coords(
            axis, levels, z_kind="pressure", column=column, n_points=n_points
        )
    else:
        samples = sample_levels(recordset, variable, plan)
        if samples is None:
            raise KeyError(
                f"Variable '{variable}' is not available at time {recordset.time}."
            )
        levels = levels_of(recordset, variable)
        if len(levels) == 1 or variable in _SURFACE_VARIABLES:
            return Profile(fixed=samples[:, 0])
        if column is not None and levels[0] != 0:
            # Level 0 is the ground: start the profile there, as HYSPLIT does.
            ground = column.ground_values(variable, samples[:, 0])
            samples = np.column_stack([ground, samples])
            levels = (0, *levels)
        values = samples

    coords = _vertical_coords(
        axis, levels, z_kind=z_kind, column=column, n_points=n_points
    )
    if z_kind == "pressure" and column is not None and variable != "pressure":
        # A level at or below the ground (on a pressure-level file, values the
        # model extrapolated) is left out, as prfprs leaves it out at heights.
        below = (np.asarray(levels) > 0) & (coords >= column.surface_pressure[:, None])
        coords = np.where(below, np.nan, coords)

    log_p = (
        column is not None
        and variable in PRESSURE_VARIABLES
        and not isinstance(axis, TerrainAxis)
    )
    log_coords = z_kind == "pressure"
    with np.errstate(divide="ignore", invalid="ignore"):
        return Profile(
            values=np.log(values) if log_p else values,
            coords=np.log(coords) if log_coords else coords,
            log_p=log_p,
            log_coords=log_coords,
            walk=column is not None and z_kind == "agl",
        )


def sample_variable(
    recordset: RecordSet,
    variable: str,
    targets: npt.NDArray[Any],
    *,
    z_kind: ColumnZKind,
    plan: Plan,
    column: Column | None,
) -> npt.NDArray[Any]:
    """Interpolate *variable* to each target along each row's column: ``(n_points,)`` float32."""
    if variable == "pressure" and z_kind == "pressure":
        # Pressure at a pressure is the target itself, NaN below the ground.
        if column is None:
            return targets.astype(np.float32, copy=False)
        above_ground = targets <= column.surface_pressure
        return np.where(above_ground, targets, np.nan).astype(np.float32)
    profile = variable_profile(
        recordset, variable, z_kind=z_kind, plan=plan, column=column
    )
    return profile.at(targets)
