"""
Sampling on HYSPLIT's internal levels, where a HYSPLIT run reads the met at heights.

``hymodelc`` does not read the file's own levels at a particle: ``metlvl.f``
picks one of six quadratic grids for the run, the levels follow the terrain as
sigma surfaces, each grid point's column is filled on them (``prfsig``,
``prfprs``, ``prfter``, ``prfecm``, then ``prfend`` above the file's top), and
the particle's value is linear between the two levels around it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from arlmet.vertical import (
    HybridAxis,
    PressureAxis,
    SigmaAxis,
    TerrainAxis,
    VerticalAxis,
    WrfAxis,
)

from .column import GRAV, RDRY, SFCP
from .horizontal import Plan, blend
from .profile import PRESSURE_VARIABLES, interp_profiles, variable_profile

if TYPE_CHECKING:
    from arlmet._file import File

    from .column import Column
    from .fields import TimeStep

__all__ = ["InternalLevels", "internal_levels", "sample_on_internal_levels"]

#: The quadratic internal grids HYSPLIT's ``metlvl.f`` chooses from, finest
#: last: level ``k`` (from 1) is ``a k**2 + b k + c`` meters above the ground
#: at zero terrain.
_HYSPLIT_GRIDS = (
    (30.0, -25.0, 5.0),
    (20.0, -15.0, 5.0),
    (15.0, -5.0, 0.0),
    (10.0, -10.0, 10.0),
    (5.0, 5.0, 0.0),
    (2.5, 2.5, 0.0),
)
#: ``metlvl.f``'s meters per hPa by 100 hPa (100 to 900) and by 10 hPa (10 to 90).
_ZPH1 = (17.98, 14.73, 13.09, 11.98, 11.15, 10.52, 10.04, 9.75, 9.88)
_ZPH2 = (31.37, 27.02, 24.59, 22.92, 21.65, 20.66, 19.83, 19.13, 18.51)

#: Variables HYSPLIT carries above a file's top on its internal levels at the
#: value of the last internal level below the top (``prfsig``'s tail, ``prfend``).
_HELD_ABOVE_TOP = frozenset({"TEMP", "UWND", "VWND"})


@dataclass(frozen=True, eq=False)
class InternalLevels:
    """
    HYSPLIT's internal vertical levels for a run, as ``hymodelc`` sets them up.

    Made by :func:`internal_levels`. These are the heights at which
    :func:`sample_points` reads the met at heights with a ``model_top``: it
    fills each grid point's column on them and reads a point linearly between
    the two levels around it. The levels follow the terrain as sigma surfaces
    that go flat at ``zmdl``: above ground at ``zsfc`` meters each level is
    ``(zmdl - zsfc) / zmdl`` times as high as above ground at 0 m
    (:meth:`scale`, :meth:`heights_at`).

    Attributes
    ----------
    zmdl : float
        ``ZMDL``, the height at which the sigma surfaces go flat [m]:
        ``max(model_top, 25000)`` (HYSPLIT's ``runset.f``).
    heights : numpy.ndarray
        The levels' heights above the ground where the terrain is 0 m [m],
        ``a k**2 + b k + c`` for ``k = 1, 2, ...``, from the grid
        ``metlvl.f`` picks. Read-only.
    zmdlt : float
        ``ZMDLT``, the height that scales a terrain-following (flag 3) file's
        stored levels [m]: 20000, or ``zmdl`` on a file whose source is
        ``RAMS`` (``prfter``).

    See Also
    --------
    internal_levels : Builds the levels for a file and a model top.
    sample_points : Samples on these levels at heights.
    """

    zmdl: float
    heights: npt.NDArray[np.float64]
    zmdlt: float = 20000.0

    def __post_init__(self) -> None:
        heights = np.array(self.heights, dtype=float)
        heights.setflags(write=False)
        object.__setattr__(self, "heights", heights)

    @classmethod
    def _for_axis(
        cls, axis: VerticalAxis, model_top: float, *, source: str = ""
    ) -> InternalLevels:
        """
        Return the internal grid HYSPLIT's ``metlvl.f`` sets up for a run on *axis*.

        ``metlvl`` counts the file's levels up to *model_top* (each level's
        pressure turned into a rough height with a fixed table; a
        terrain-following file's stored height as it is) and takes the first
        of six quadratic grids that has more levels than that up to
        *model_top*. Computed in single precision, as HYSPLIT does. A WRF
        (flag 5) file counts as sigma: without WRF's hybrid coefficients
        HYSPLIT reads it as flag 1. ``ZMDLT`` is 20000 m, ``runset.f``'s
        default, and ``ZMDL`` on a file whose *source* is ``RAMS`` (``prfter``).
        """
        f32 = np.float32
        zdata = f32(model_top)
        offset = f32(axis.offset)
        levels = np.asarray(axis.levels, dtype=np.float32)
        n = 0
        zlevel = f32(0.0)
        while zlevel <= zdata and n < len(levels):
            n += 1
            height = levels[n - 1]
            if isinstance(axis, TerrainAxis):
                zlevel = height
                continue
            if isinstance(axis, (SigmaAxis, WrfAxis)):
                plevel = offset + (f32(SFCP) - offset) * height
            elif isinstance(axis, PressureAxis):
                plevel = height if n > 1 else f32(SFCP)
            else:
                assert isinstance(axis, HybridAxis)
                floor = f32(int(height))
                plevel = f32(SFCP) * (height - floor) + floor if n > 1 else f32(SFCP)
            if plevel >= 100.0:
                delz = f32(_ZPH1[min(9, int(plevel / f32(100.0))) - 1])
            else:
                delz = f32(_ZPH2[max(1, min(9, int(plevel / f32(10.0)))) - 1])
            zlevel = (f32(SFCP) - plevel) * delz
        needed = min(n + 1, len(levels))

        count = 0
        a = b = c = f32(0.0)
        for grid in _HYSPLIT_GRIDS:
            a, b, c = (f32(v) for v in grid)
            dist = b * b - f32(4.0) * a * (c - zdata)
            count = int((-b + np.sqrt(dist)) / (f32(2.0) * a)) + 1 if dist >= 0 else 1
            if count > needed:
                break
        k = np.arange(1, count + 1, dtype=float)
        heights = float(a) * k * k + float(b) * k + float(c)
        zmdl = max(abs(float(model_top)), 25000.0)
        return cls(
            zmdl=zmdl, heights=heights, zmdlt=zmdl if source == "RAMS" else 20000.0
        )

    def scale(self, terrain: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """
        Return how much higher the levels are above *terrain* than above 0 m.

        Parameters
        ----------
        terrain : array_like
            Ground heights above sea level [m], such as a file's ``SHGT``.

        Returns
        -------
        numpy.ndarray
            ``(zmdl - terrain) / zmdl``, the shape of *terrain*.

        See Also
        --------
        heights_at : The levels' heights above that ground.

        Examples
        --------
        >>> import arlmet
        >>> levels = arlmet.internal_levels("met.arl")
        >>> levels.scale([0.0, 1250.0])
        array([1.  , 0.95])
        """
        return (self.zmdl - np.asarray(terrain, dtype=float)) / self.zmdl

    def heights_at(self, terrain: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """
        Return the levels' heights above the ground at each *terrain*.

        Parameters
        ----------
        terrain : array_like
            Ground heights above sea level [m], such as a file's ``SHGT`` at
            the grid points.

        Returns
        -------
        numpy.ndarray
            Heights above the ground [m], ``heights * scale(terrain)``, of shape
            ``terrain.shape + heights.shape``: the last axis is the level.

        See Also
        --------
        scale : The factor applied to ``heights``.

        Examples
        --------
        >>> import arlmet
        >>> levels = arlmet.internal_levels("met.arl")
        >>> levels.heights_at([0.0, 1250.0])[:, :3]
        array([[ 10.  ,  75.  , 200.  ],
               [  9.5 ,  71.25, 190.  ]])
        """
        return np.multiply.outer(self.scale(terrain), self.heights)

    def _position(
        self, z: npt.NDArray[Any], terrain: npt.NDArray[Any]
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """
        Return the internal layer of each height *z* above its *terrain*, and the fraction up it.

        Returns ``(k, f)``: the value is the one at level ``k`` (0-based)
        plus ``f`` of the way to level ``k + 1``, linear in sigma, as
        ``hymodelc`` reads a particle's ``PRES``, ``TEMP`` and ``DENS`` and
        ``zindex.f`` places it for ``adv3nt``. The particle's sigma is ``1 -
        z / (ZMDL - zsfc)``. Below the first level the first level's value
        (``f = 0``), above the last the last's (``f = 1``). ``f`` is NaN for a
        NaN or negative *z*.
        """
        h = self.heights
        z = np.asarray(z, dtype=float)
        with np.errstate(invalid="ignore"):
            internal = z / self.scale(terrain)  # the height at 0 m terrain
            k = np.clip(
                np.searchsorted(h, internal, side="right") - 1, 0, max(len(h) - 2, 0)
            )
            if len(h) < 2:
                f = np.zeros_like(internal)
            else:
                f = np.clip((internal - h[k]) / (h[k + 1] - h[k]), 0.0, 1.0)
            f = np.where(np.isfinite(z) & (z >= 0.0), f, np.nan)
        return k, f


def internal_levels(
    file: File | str | os.PathLike[str], *, model_top: float = 25000.0
) -> InternalLevels:
    """
    Return HYSPLIT's internal vertical levels for a run on *file* with *model_top*.

    HYSPLIT does not read the met at a particle's own height: it fills its own
    internal levels at every grid point and reads the particle between two of
    them. For the run's top of model domain and the file's vertical grid,
    HYSPLIT's ``metlvl.f`` picks one of six grids of heights ``a k**2 + b k +
    c``; these are the levels :func:`sample_points` uses at heights with the
    same ``model_top``.

    Parameters
    ----------
    file : File or path-like
        An open ARL file, or the path to one. Only its vertical axis and
        source are read; a path is opened and closed here.
    model_top : float, default 25000
        The run's top of model domain [m] (the ``CONTROL`` line after the
        vertical motion option; PYSTILT's ``z_top``), as in
        :func:`sample_points`.

    Returns
    -------
    InternalLevels

    Raises
    ------
    ValueError
        If *model_top* is not a positive number.

    See Also
    --------
    InternalLevels : The returned levels.
    sample_points : Samples on these levels at heights.

    Examples
    --------
    >>> import arlmet
    >>> levels = arlmet.internal_levels("met.arl")
    >>> levels.zmdl
    25000.0
    >>> levels.heights[:4]
    array([ 10.,  75., 200., 385.])
    """
    from arlmet._file import File  # lazy: _file imports this module

    if not (np.isfinite(model_top) and model_top > 0):
        raise ValueError(
            f"model_top must be a positive number of meters, got {model_top!r}."
        )
    if isinstance(file, File):
        return InternalLevels._for_axis(
            file.vertical_axis, model_top, source=file.source
        )
    with File(file) as met:
        return InternalLevels._for_axis(met.vertical_axis, model_top, source=met.source)


def _pressure_above_top(
    p_top: npt.NDArray[Any],
    t_top: npt.NDArray[Any],
    tv_top: npt.NDArray[Any],
    k_top: npt.NDArray[Any],
    k: npt.NDArray[Any],
    heights: npt.NDArray[Any],
) -> npt.NDArray[Any]:
    """
    Pressure at internal level *k*, above level *k_top*, the last below a column's top.

    HYSPLIT's ``prfend`` (and ``prfsig``'s tail) steps it up level by level
    hydrostatically with the level below's density: ``p_k = p_{k-1} (1 - g dz
    / (R T))``, where ``T`` is *tv_top*, the virtual temperature at level
    *k_top*, for the first step and the temperature *t_top* (dry) after.
    *heights* ``(rows, levels)`` are the levels' heights above each row's
    ground; every other array is ``(rows,)``.
    """
    rows = np.arange(len(k))
    dz = np.diff(heights, axis=1)  # dz[:, i] from level i to level i + 1
    step = np.log1p(-GRAV * dz / (RDRY * t_top[:, None]))
    cum = np.concatenate([np.zeros((len(k), 1)), np.cumsum(step, axis=1)], axis=1)
    # the first step uses tv_top; the steps after it, from k_top + 1 to k, t_top
    first = np.log1p(-GRAV * dz[rows, k_top] / (RDRY * tv_top))
    log_p = np.log(p_top) + first + cum[rows, k] - cum[rows, k_top + 1]
    return np.exp(log_p)


def sample_on_internal_levels(
    recordset: TimeStep,
    variable: str,
    k: npt.NDArray[Any],
    f: npt.NDArray[Any],
    levels: InternalLevels,
    nodes: Plan,
    weights: npt.NDArray[Any],
    column: Column,
) -> npt.NDArray[Any]:
    """
    Sample *variable* at each point as HYSPLIT does on its internal levels.

    At each grid point around a point (*nodes*, *weights*) the profile is read
    at the internal levels ``k`` and ``k + 1`` (:meth:`InternalLevels._position`)
    above that grid point's own ground, and the value is ``f`` of the way
    between them; the grid points are then blended. Above a column's top
    ``TEMP``, ``UWND`` and ``VWND`` keep the value of the last internal level
    below it and pressure falls hydrostatically from there
    (:func:`_pressure_above_top`), as HYSPLIT fills its internal levels;
    another variable is NaN there.
    """
    kk, ff = np.tile(k, len(weights)), np.tile(f, len(weights))
    h = levels.heights
    scale = levels.scale(column.terrain)
    k_lo, k_hi = kk, np.minimum(kk + 1, len(h) - 1)
    profile = variable_profile(
        recordset, variable, z_kind="agl", plan=nodes, column=column
    )
    lo = profile.at(h[k_lo] * scale).astype(float)
    hi = profile.at(h[k_hi] * scale).astype(float)

    held = variable in _HELD_ABOVE_TOP
    if (held or variable in PRESSURE_VARIABLES) and profile.fixed is None:
        top = profile.top()
        with np.errstate(invalid="ignore"):
            # the last internal level at or below each column's top
            k_top = np.searchsorted(h, top / scale, side="right") - 1
            rows = np.flatnonzero((h[k_hi] * scale > top) & (k_top >= 0))
        if len(rows):
            k_top = np.maximum(k_top, 0)
            h_top = h[k_top] * scale
            base = profile.at(h_top)[rows].astype(float)
            if held:
                extended = [base, base]
            else:
                t_top = variable_profile(
                    recordset, "TEMP", z_kind="agl", plan=nodes, column=column
                ).at(h_top)[rows]
                tv_top = interp_profiles(
                    column.virtual_temperature, column.z_agl, h_top, column=True
                )[rows]
                extended = [
                    _pressure_above_top(
                        base,
                        t_top.astype(float),
                        tv_top.astype(float),
                        k_top[rows],
                        np.maximum(k_at[rows], k_top[rows] + 1),
                        h[None, :] * scale[rows, None],
                    )
                    for k_at in (k_lo, k_hi)
                ]
            for values, k_at, above in zip(
                (lo, hi), (k_lo, k_hi), extended, strict=True
            ):
                past = k_at[rows] > k_top[rows]
                values[rows] = np.where(past, above, values[rows])

    with np.errstate(invalid="ignore"):
        at_nodes = np.where(ff == 0.0, lo, lo + ff * (hi - lo))
    return blend(at_nodes, weights)
