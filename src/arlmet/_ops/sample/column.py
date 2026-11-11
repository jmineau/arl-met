"""
The vertical column of one time step at each point of a :class:`~.horizontal.Plan`.

One class per HYSPLIT routine that builds a column at each grid point:
``prfsig`` (sigma, flag 1, and WRF, flag 5, without WRF's hybrid
coefficients), ``prfprs`` (pressure, flag 2), ``prfter`` (terrain-following,
flag 3) and ``prfecm`` (hybrid, flag 4). Level 0 is the ground: 0 m above it,
at the surface pressure ``PRSS``. Every variable sampled at a time step shares
its column; each field is read only when first needed, and every array is
``(n_points, n_levels)`` over all the axis's levels, or ``(n_points,)`` for
the ground.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
from typing_extensions import override

from arlmet.vertical import (
    HybridAxis,
    PressureAxis,
    SigmaAxis,
    TerrainAxis,
    WrfAxis,
    _hysplit_virtual_temperature,
    _hysplit_z_agl,
)

from .fields import levels_of, sample_levels
from .horizontal import Plan

if TYPE_CHECKING:
    from arlmet._recordset import RecordSet

    from .internal import InternalLevels

__all__ = ["GRAV", "RDRY", "SFCP", "Column", "column_for"]

#: HYSPLIT's ``prfmod`` constants.
GRAV = 9.80616  # m/s2
RDRY = 287.04  # J/(kg K)
#: HYSPLIT's 1013 hPa: ``metlvl.f``'s assumed surface pressure and ``prfmod``'s
#: mean sea-level pressure.
SFCP = 1013.0

#: The surface field that gives an upper-level variable its value at the
#: ground, as HYSPLIT's ``prfsig``, ``prfprs``, ``prfecm`` and ``prfter`` start
#: their columns. ``TEMP`` takes the column's ground temperature, vertical
#: motion is zero, and any other variable keeps its first level's value.
_GROUND_FIELDS: dict[str, str] = {
    "PRES": "PRSS",
    "HGTS": "SHGT",
    "UWND": "U10M",
    "VWND": "V10M",
}
_ZERO_AT_GROUND = frozenset({"WWND", "DZDT"})


def terrain_at(recordset: RecordSet, plan: Plan) -> npt.NDArray[Any]:
    """Sample the terrain height ``SHGT`` [m] at each point of *plan*."""
    held = levels_of(recordset, "SHGT")
    if not held:
        raise ValueError(f"Terrain field SHGT is required at time {recordset.time}.")
    terrain = sample_levels(recordset, "SHGT", plan, held[:1])
    assert terrain is not None
    return terrain[:, 0]


class Column:
    """
    The column of a sigma, hybrid or WRF time step: the base the others change.

    Heights come from the hypsometric equation, integrated up from the ground
    with the virtual temperature, as ``prfsig`` and ``prfecm`` find them.
    """

    def __init__(
        self, recordset: RecordSet, plan: Plan, *, levels: InternalLevels | None = None
    ) -> None:
        self.recordset = recordset
        self.plan = plan
        self.levels = levels
        self.axis = recordset.vertical_axis
        self.upper = list(range(1, len(self.axis.levels)))

    def _surface(self, variable: str) -> npt.NDArray[Any] | None:
        """Sample the surface field *variable*, or None if the time step lacks it."""
        values = sample_levels(self.recordset, variable, self.plan, [0])
        return None if values is None else values[:, 0]

    def _upper(self, variable: str) -> npt.NDArray[Any] | None:
        """Sample *variable* at every level above ground, or None if absent."""
        return sample_levels(self.recordset, variable, self.plan, self.upper)

    def _upper_pres(self, flag_name: str) -> npt.NDArray[Any]:
        """``PRSS`` at the ground, then the ``PRES`` field, which this kind of file must hold."""
        pres = self._upper("PRES")
        if pres is None:
            raise ValueError(
                f"{flag_name} files require PRES records for pressure and AGL/MSL sampling."
            )
        return np.column_stack([self.surface_pressure, pres])

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
        """Pressure [hPa] of each level: ``PRSS`` at the ground, then the axis's."""
        sp = self.surface_pressure
        p = self.axis.to_pressure(surface_pressure=sp)
        p[:, 0] = sp
        return p

    @cached_property
    def z_agl(self) -> npt.NDArray[Any]:
        """Height above ground [m] of each level, 0 at the ground (HYSPLIT's heights)."""
        z = np.zeros_like(self.pressure)
        if self.upper:
            p = self.pressure
            z[:, 1:] = _hysplit_z_agl(
                p[:, 1:],
                p[:, 0],
                self._temperature,
                first_sigma=self._first_sigma,
                surface_temperature=self._surface_temperature,
                specific_humidity=self._upper("SPHU"),
                relative_humidity=self._upper("RELH"),
            )
        return z

    @cached_property
    def terrain(self) -> npt.NDArray[Any]:
        """
        Terrain height [m] at each point: ``SHGT``.

        A time step without ``SHGT`` gets HYSPLIT's estimate from the surface
        pressure and the first level's temperature (``prfmod``'s ``PRFA``:
        ``ln(1013 / PRSS) R T1 / g``, 0 where ``PRSS`` is above 1013 hPa), which
        ``prfsig``, ``prfter`` and ``prfecm`` write back into HYSPLIT's terrain
        array.
        """
        if (0, "SHGT") in self.recordset:
            return terrain_at(self.recordset, self.plan)
        p0 = self.surface_pressure.astype(float)
        t1 = self._temperature[:, 0].astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            estimate = np.log(SFCP / p0) * RDRY * t1 / GRAV
        return np.where(p0 <= SFCP, estimate, 0.0)

    @property
    def _level_pressure(self) -> npt.NDArray[Any]:
        """What ``CALCTV`` takes as each upper level's pressure."""
        return self.pressure[:, 1:]

    @cached_property
    def virtual_temperature(self) -> npt.NDArray[Any]:
        """
        Virtual temperature [K] of each level, the ground's at level 0.

        HYSPLIT's ``CALCTV`` at each level's pressure, and at the ground's
        temperature and pressure for the ground, as ``prfsig``, ``prfprs`` and
        ``prfecm`` find the air's density from it.
        """
        tv, tv_ground = _hysplit_virtual_temperature(
            self._level_pressure,
            self.surface_pressure,
            self._temperature,
            self.ground_temperature,
            specific_humidity=self._upper("SPHU"),
            relative_humidity=self._upper("RELH"),
        )
        return np.column_stack([tv_ground, tv])

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
                "TEMP (or THET and PRES) for AGL/MSL sampling."
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
        Sigma of the first level above ground, which brings its temperature down to the ground.

        Here the hybrid file's: level 1's nominal pressure over ``PRSS`` (``ESG(1)``).
        """
        return self.pressure[:, 1] / self.surface_pressure

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


class _SigmaColumn(Column):
    """``prfsig``: pressure from the ``PRES`` field where the file has one (``ZFLG``), else sigma times ``PRSS``."""

    @override
    @cached_property
    def pressure(self) -> npt.NDArray[Any]:
        """Pressure [hPa] of each level: ``PRSS``, then ``PRES`` or the nominal sigma pressure."""
        sp = self.surface_pressure
        p = self.axis.to_pressure(surface_pressure=sp)
        pres = self._upper("PRES")
        if pres is not None:
            p[:, 1:] = pres
        p[:, 0] = sp
        return p

    @override
    @cached_property
    def _first_sigma(self) -> npt.NDArray[Any]:
        """The stored sigma of the first level (``PSG(1)``; on WRF files the stored eta)."""
        return np.full(self.plan.n_points, self.axis.levels[1])


class _WrfColumn(_SigmaColumn):
    """``prfsig`` on a WRF file: its eta levels give pressure only with WRF's coefficients, so ``PRES``."""

    @override
    @cached_property
    def pressure(self) -> npt.NDArray[Any]:
        """Pressure [hPa] of each level: ``PRSS``, then ``PRES``."""
        return self._upper_pres("WRF (flag=5)")


class _PressureColumn(Column):
    """
    ``prfprs``: heights from ``HGTS - SHGT``.

    A level at or below the ground has no height (NaN), so a profile runs from
    the ground to the first level above it, as ``prfprs`` skips such levels.
    Given HYSPLIT's internal *levels*, ``prfprs`` also skips a level between
    the ground and the first internal level above it, and so does this.
    """

    @override
    @cached_property
    def pressure(self) -> npt.NDArray[Any]:
        """Pressure [hPa] of each level: ``PRSS``, then the stored levels."""
        sp = self.surface_pressure
        p = np.tile(self.axis.to_pressure(), (len(sp), 1))
        p[:, 0] = sp
        return p

    @override
    @cached_property
    def z_agl(self) -> npt.NDArray[Any]:
        """``HGTS - SHGT`` [m] of each level, 0 at the ground, NaN at or below it."""
        hgts = self._upper("HGTS")
        if hgts is None:
            raise ValueError(
                "Pressure-level (flag=2) files require HGTS records for AGL/MSL sampling."
            )
        terrain = self.terrain
        agl = hgts - terrain[:, None]
        if self.levels is None:
            above = agl > 0.0
        else:
            # prfprs moves the bottom of its profile up only after filling its
            # first output level (IF(KL.GT.1)).
            first = self.levels.heights[0] * self.levels.scale(terrain)
            above = agl >= first[:, None]
        z = np.zeros((len(terrain), len(self.axis.levels)), dtype=np.float32)
        z[:, 1:] = np.where(above, agl, np.nan)
        return z

    @override
    @cached_property
    def terrain(self) -> npt.NDArray[Any]:
        """Terrain height [m] at each point: ``SHGT``, which a pressure-level file must hold."""
        return terrain_at(self.recordset, self.plan)

    @override
    @cached_property
    def _first_sigma(self) -> npt.NDArray[Any]:
        """Level 1's pressure over ``PRSS`` (``PSG(1)/P0``); that level can be below the ground."""
        return self.axis.levels[1] / self.surface_pressure


class _TerrainColumn(Column):
    """
    ``prfter``: the stored levels are ``Z*``, heights over flat ground.

    It exists only on HYSPLIT's internal *levels*, which say how the levels
    scale with the terrain.
    """

    @property
    def _zmdlt(self) -> float:
        """``ZMDLT``, which scales the stored levels with the terrain."""
        assert self.levels is not None  # column_for builds this only on internal levels
        return self.levels.zmdlt

    @override
    @cached_property
    def surface_pressure(self) -> npt.NDArray[Any]:
        """
        ``PRSS`` [hPa] at each point.

        A time step without ``PRSS`` gets HYSPLIT's estimate from ``SHGT`` and
        the first level's temperature (``prfmod``'s ``PRFA``: ``1013
        exp(-zsfc g / (R T1))``).
        """
        if levels_of(self.recordset, "PRSS") or (0, "SHGT") not in self.recordset:
            return super().surface_pressure
        t1 = self._temperature[:, 0].astype(float)
        return SFCP * np.exp(-self.terrain * GRAV / RDRY / t1)

    @override
    @cached_property
    def pressure(self) -> npt.NDArray[Any]:
        """Pressure [hPa] of each level: ``PRSS``, then ``PRES``."""
        return self._upper_pres("Terrain-following (flag=3)")

    @override
    @cached_property
    def z_agl(self) -> npt.NDArray[Any]:
        """``Z* (1 - zsfc / ZMDLT)`` [m] at each grid point, as ``prfter`` scales the stored levels."""
        terrain = np.asarray(self.terrain, dtype=float)
        z = np.outer(1.0 - terrain / self._zmdlt, self.axis.levels)
        z[:, 0] = 0.0
        return z

    @override
    @property
    def _level_pressure(self) -> npt.NDArray[Any]:
        """
        Each level's stored height: ``prfter`` passes it where ``CALCTV`` takes the pressure.

        That only matters with ``RELH``; this does the same, to give its value.
        """
        return np.tile(self.axis.levels[1:], (self.plan.n_points, 1))

    @override
    @cached_property
    def _first_sigma(self) -> npt.NDArray[Any]:
        """``prfter``'s ``1 - Z*(1) / ZMDLT``."""
        sigma = 1.0 - self.axis.levels[1] / self._zmdlt
        return np.full(self.plan.n_points, sigma)

    @override
    def ground_values(
        self, variable: str, first_level: npt.NDArray[Any]
    ) -> npt.NDArray[Any]:
        """As :meth:`Column.ground_values`, but ``PRES`` at the ground is ``PRSS`` or its estimate."""
        if variable == "PRES":
            return self.surface_pressure
        return super().ground_values(variable, first_level)


_COLUMNS: dict[type, type[Column]] = {
    SigmaAxis: _SigmaColumn,
    WrfAxis: _WrfColumn,
    PressureAxis: _PressureColumn,
    TerrainAxis: _TerrainColumn,
    HybridAxis: Column,
}


def column_for(
    recordset: RecordSet,
    plan: Plan,
    *,
    z_kind: Literal["native", "pressure", "agl"],
    levels: InternalLevels | None,
) -> Column | None:
    """
    Return the column of *recordset* at *plan*'s points, or None where the stored levels are used.

    Sigma, hybrid and WRF files always start their column at the ground;
    pressure-level files do at heights and at pressures (not at native
    levels), and terrain-following files on the internal *levels* only.
    """
    axis = recordset.vertical_axis
    if isinstance(axis, PressureAxis) and z_kind == "native":
        return None
    if isinstance(axis, TerrainAxis) and levels is None:
        return None
    return _COLUMNS[type(axis)](recordset, plan, levels=levels)
