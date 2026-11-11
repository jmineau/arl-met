"""
Where each sample point is on the grid, and how grid-point values are blended to it.

A :class:`Plan` holds, for every point, the grid cell around it and its
bilinear weights (or the nearest grid point), and the smallest window of the
grid that covers every point, so a record is decoded only there.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from arlmet._grid import Grid, GridWindow

__all__ = [
    "WIND_PAIRS",
    "WRF_STAGGER",
    "Plan",
    "SampleMethod",
    "blend",
    "build_plan",
    "node_plan",
    "rotate_winds",
    "sample_field",
]

#: Horizontal interpolation methods for point sampling.
SampleMethod = Literal["linear", "nearest"]

#: Wind component pairs that are stored grid-relative on projected grids.
WIND_PAIRS: tuple[tuple[str, str], ...] = (("UWND", "VWND"), ("U10M", "V10M"))

#: On WRF files (source ``AWRF``), where ``UWND`` and ``VWND`` are stored, in
#: grid cells from the mass points (cell centers): WRF's staggered grid, whose
#: first ``nx`` U points and ``ny`` V points ``arw2arl`` writes. Column ``i`` of
#: ``UWND`` is the west face of cell ``i`` and row ``j`` of ``VWND`` its south
#: face, so HYSPLIT reads them at array positions ``x + 0.5`` and ``y + 0.5``
#: (``metini.f``'s ``SUX``, ``SVY``; ``adviec.f``, ``advmet.f``).
WRF_STAGGER: dict[str, tuple[float, float]] = {
    "UWND": (0.5, 0.0),
    "VWND": (0.0, 0.5),
}

#: How far outside the grid, in cells, a point is still on its edge. Mapping a
#: projected grid's own node lon/lat back to grid indices leaves about 1e-12
#: cells of rounding error, which puts half the edge nodes just off the grid
#: (#134); 1e-6 cells is a few millimeters on a 3 km grid.
_EDGE_TOLERANCE = 1e-6

#: A grid point whose weight is below this does not count: a point given by a
#: grid point's own longitude and latitude comes back from the projection with
#: weights of about 1e-12 toward its neighbors, not 0.
_NO_WEIGHT = 1e-6


@dataclass(frozen=True)
class Plan:
    """
    Where a set of (lon, lat) points is on the grid.

    Attributes
    ----------
    method : 'linear' or 'nearest'
    window : the smallest GridWindow holding every inside point, or None if none is
    inside : True where the point is on the grid
    x0, x1, y0, y1 : the grid cell around each point (all equal for nearest)
    wx, wy : fractional weights toward x1/y1 (zero for nearest)
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

    @property
    def n_points(self) -> int:
        """The number of points, inside the grid or not."""
        return len(self.inside)

    @cached_property
    def _nearest(self) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
        """For ``"nearest"``: the inside points' rows and their row and column in the window."""
        assert self.window is not None
        rows = np.flatnonzero(self.inside)
        return (
            rows,
            self.y0[rows] - self.window.y_start,
            self.x0[rows] - self.window.x_start,
        )


def build_plan(
    grid: Grid,
    lon: npt.NDArray[Any],
    lat: npt.NDArray[Any],
    *,
    method: SampleMethod,
    stagger: tuple[float, float] = (0.0, 0.0),
) -> Plan:
    """
    Place (lon, lat) points on *grid*.

    *stagger* is ``(dx, dy)`` in grid cells for a field stored on a staggered
    grid (:data:`WRF_STAGGER`): it is read at array position ``(x + dx, y +
    dy)``, as HYSPLIT's ``adv3nt`` reads it. A point is inside only where both
    its own position and that one are on the grid. With ``"nearest"`` the point
    moves to the nearest cell center first and the field is interpolated
    there, as HYSPLIT would at that center: the mean of the cell's two faces.
    """
    x, y = grid.fractional_indices(lon, lat)
    # On a global lat/lon grid, x in [nx - 1, nx) lies between the last and
    # first columns and interpolates across the seam.
    wraps = grid.wraps_lon
    x_max = float(grid.nx) if wraps else float(grid.nx - 1)
    y_max = float(grid.ny - 1)

    def snap(v: npt.NDArray[Any], hi: float) -> npt.NDArray[Any]:
        """Move values within ``_EDGE_TOLERANCE`` outside ``[0, hi]`` onto it."""
        v = np.where((v < 0.0) & (v >= -_EDGE_TOLERANCE), 0.0, v)
        return np.where((v > hi) & (v <= hi + _EDGE_TOLERANCE), hi, v)

    def onto_edges(
        x: npt.NDArray[Any], y: npt.NDArray[Any]
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """Move positions within ``_EDGE_TOLERANCE`` outside the grid onto its edge."""
        if wraps:
            # x = nx on a global grid (a longitude a rounding error west of
            # the first column; np.mod can return nx too) is column 0.
            x = np.mod(x, grid.nx)
            x = np.where(x >= grid.nx, 0.0, x)
        else:
            if grid.is_latlon:
                # Longitude is measured east of the first column modulo 360, so
                # one a rounding error west of it is a full turn east (#141).
                turn = 360.0 / abs(grid.projection.tangent_lon)
                x = np.where(np.abs(x - turn) <= _EDGE_TOLERANCE, x - turn, x)
            x = snap(x, x_max)
        return x, snap(y, y_max)

    def on_grid(x: npt.NDArray[Any], y: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Return True where array position ``(x, y)`` can be interpolated."""
        return (
            np.isfinite(x)
            & np.isfinite(y)
            & (x >= 0.0)
            & ((x < x_max) if wraps else (x <= x_max))
            & (y >= 0.0)
            & (y <= y_max)
        )

    x, y = onto_edges(x, y)
    inside = on_grid(x, y)
    if stagger != (0.0, 0.0):
        if method == "nearest":
            x, y = np.rint(x), np.rint(y)
            method = "linear"
        x, y = onto_edges(x + stagger[0], y + stagger[1])
        inside &= on_grid(x, y)

    x = np.clip(np.nan_to_num(x), 0.0, max(x_max, 0.0))
    y = np.clip(np.nan_to_num(y), 0.0, max(y_max, 0.0))
    if method == "nearest":
        x0 = x1 = np.rint(x).astype(int) % grid.nx if wraps else np.rint(x).astype(int)
        y0 = y1 = np.rint(y).astype(int)
        wx = wy = np.zeros_like(x)
    else:
        x0 = np.minimum(np.floor(x).astype(int), max(grid.nx - 1, 0))
        y0 = np.floor(y).astype(int)
        x1 = (x0 + 1) % grid.nx if wraps else np.clip(x0 + 1, 0, grid.nx - 1)
        y1 = np.clip(y0 + 1, 0, grid.ny - 1)
        wx, wy = x - x0, y - y0

    window = None
    if inside.any():
        both = np.tile(inside, 2)
        xs = np.concatenate([x0, x1])[both]
        ys = np.concatenate([y0, y1])[both]
        window = GridWindow(
            x_start=int(xs.min()),
            x_stop=int(xs.max()) + 1,
            y_start=int(ys.min()),
            y_stop=int(ys.max()) + 1,
        )
    return Plan(method, window, inside, x0, x1, y0, y1, wx, wy)


def sample_field(field: npt.NDArray[Any], plan: Plan) -> npt.NDArray[Any]:
    """Interpolate a field read in ``plan.window`` to each point: ``(n_points,)`` float32, NaN outside."""
    result = np.full(plan.n_points, np.nan, dtype=np.float32)
    if plan.window is None:
        return result
    if plan.method == "nearest":
        rows, iy, ix = plan._nearest
        result[rows] = field[iy, ix]
        return result

    # Indices are relative to the full grid; subtract the window's origin.
    ok = plan.inside
    x0 = plan.x0[ok] - plan.window.x_start
    x1 = plan.x1[ok] - plan.window.x_start
    y0 = plan.y0[ok] - plan.window.y_start
    y1 = plan.y1[ok] - plan.window.y_start
    wx = plan.wx[ok].astype(np.float32, copy=False)
    wy = plan.wy[ok].astype(np.float32, copy=False)
    result[ok] = (
        (1.0 - wx) * (1.0 - wy) * field[y0, x0]
        + wx * (1.0 - wy) * field[y0, x1]
        + (1.0 - wx) * wy * field[y1, x0]
        + wx * wy * field[y1, x1]
    )
    return result


def node_plan(
    plan: Plan, *, each_level_first: bool = False
) -> tuple[Plan, npt.NDArray[Any]]:
    """
    Return the grid points that each point of *plan* is blended from, and their weights.

    Returns ``(nodes, weights)``: *nodes* samples, without interpolating, the
    four grid points around each point (``(x0, y0)``, ``(x1, y0)``, ``(x0,
    y1)``, ``(x1, y1)``, in that order, each a block of ``n_points`` rows), and
    *weights* ``(4, n_points)`` are their bilinear weights. Each grid point's
    profile is then built and interpolated to the point's height before the
    values are blended (:func:`blend`), as HYSPLIT's ``prfcom`` builds a
    profile at each grid point and ``adv3nt`` blends them. With ``"nearest"``,
    or with *each_level_first*, *plan* is its own one grid point, weight 1:
    each level is then blended first and the blended column interpolated.
    """
    if plan.method == "nearest" or each_level_first:
        return plan, np.ones((1, plan.n_points))
    x = np.concatenate([plan.x0, plan.x1, plan.x0, plan.x1])
    y = np.concatenate([plan.y0, plan.y0, plan.y1, plan.y1])
    wx, wy = plan.wx, plan.wy
    weights = np.stack([(1 - wx) * (1 - wy), wx * (1 - wy), (1 - wx) * wy, wx * wy])
    zeros = np.zeros(len(x))
    nodes = Plan(
        "nearest", plan.window, np.tile(plan.inside, 4), x, x, y, y, zeros, zeros
    )
    return nodes, weights


def blend(values: npt.NDArray[Any], weights: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """
    Blend the values at each point's grid points (:func:`node_plan`) with their weights.

    Only the grid points that have a value count, and their weights are
    rescaled to sum to 1: a neighbor whose top is below a height does not make
    the point NaN. The point is NaN only where none of its grid points (with a
    weight of at least ``_NO_WEIGHT``) has a value.
    """
    v = np.asarray(values, dtype=float).reshape(weights.shape)
    w = np.where(np.isfinite(v) & (weights >= _NO_WEIGHT), weights, 0.0)
    total = w.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        blended = np.where(w > 0, w * v, 0.0).sum(axis=0) / total
    return np.where(total > 0, blended, np.nan).astype(np.float32)


def rotate_winds(
    sampled: dict[str, npt.NDArray[Any]],
    grid: Grid,
    lon: npt.NDArray[Any],
    lat: npt.NDArray[Any],
    pairs: Sequence[tuple[str, str]],
) -> None:
    """Rotate each sampled wind pair in *sampled* from grid- to earth-relative, in place."""
    if grid.is_latlon:
        return
    for u_name, v_name in pairs:
        u, v = grid.rotate_winds(
            sampled[u_name].astype(float), sampled[v_name].astype(float), lon, lat
        )
        sampled[u_name] = np.asarray(u, dtype=np.float32)
        sampled[v_name] = np.asarray(v, dtype=np.float32)
