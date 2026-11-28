"""
Reading a field as HYSPLIT's ``metinp.f`` uses it, at each point of a :class:`~.horizontal.Plan`.

That is the stored field, except for two conversions HYSPLIT makes at each
grid point before any interpolation: ``TEMP`` from potential temperature, and
the winds of a mass-coupled WRF file.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import numpy.typing as npt

from .horizontal import Plan, sample_field

if TYPE_CHECKING:
    import pandas as pd

    from arlmet._grid import Grid, GridWindow
    from arlmet.vertical import VerticalAxis

__all__ = [
    "Field",
    "TimeStep",
    "levels_of",
    "mass_coupled",
    "sample_levels",
    "temperature_from_theta",
]


class Field(Protocol):
    """One stored 2D field: a :class:`~arlmet.DataRecord`, or a slice of a Dataset."""

    def read(self, window: GridWindow | None = None) -> npt.NDArray[np.float32]:
        """Return the field, or the part of it in *window*, as float32."""
        ...


class TimeStep(Protocol):
    """
    The fields of one valid time, as the samplers read them.

    A :class:`~arlmet.RecordSet` of an ARL file, or one time of a Dataset in
    :func:`arlmet.open_dataset`'s layout (``arlmet._xarray._sampling``). A
    field is stored at a level when ``(level, name) in`` the time step.
    """

    @property
    def time(self) -> pd.Timestamp:
        """The valid time."""
        ...

    @property
    def source(self) -> str:
        """The ARL source ID, such as ``"AWRF"``."""
        ...

    @property
    def grid(self) -> Grid:
        """The horizontal grid."""
        ...

    @property
    def vertical_axis(self) -> VerticalAxis:
        """The vertical axis; its levels are numbered from 0, the ground."""
        ...

    def __contains__(self, key: object, /) -> bool:
        """Return whether the field ``(level, name)`` is stored."""
        ...

    def __getitem__(self, key: tuple[int, str], /) -> Field:
        """Return the stored field ``(level, name)``."""
        ...


#: On a mass-coupled WRF file (``AWRF`` with ``MUU0`` and ``MUV0``: ``arw2arl``'s
#: time-averaged fluxes, HYSPLIT's ``WRF = 3``), each upper-level wind is stored
#: coupled to the column mass. HYSPLIT's ``metinp.f`` decouples it at each grid
#: point: the field times the surface map factor named here, divided by the
#: surface column mass named here (or by 100), ``UWND * MSFU / MUU0``.
_DECOUPLE: dict[str, tuple[str, str | float]] = {
    "UWND": ("MSFU", "MUU0"),
    "VWND": ("MSFV", "MUV0"),
    "WWND": ("MSFT", 100.0),
}


def mass_coupled(recordset: TimeStep) -> bool:
    """
    Whether the time step's winds are WRF's mass-coupled fluxes (HYSPLIT's ``WRF = 3``).

    As HYSPLIT's ``metini.f`` decides: an ``AWRF`` file with ``MUU0`` and
    ``MUV0`` at the surface.
    """
    return recordset.source == "AWRF" and all(
        (0, name) in recordset for name in ("MUU0", "MUV0")
    )


def temperature_from_theta(thet: Any, pres: Any) -> Any:
    """``TEMP`` of a file that stores ``THET``, as HYSPLIT's ``metinp.f`` finds it: ``THET * (PRES / 1000) ** 0.286``."""
    return thet * (pres / np.float32(1000.0)) ** np.float32(0.286)


def levels_of(recordset: TimeStep, variable: str) -> tuple[int, ...]:
    """
    Levels at which the time step holds *variable* as HYSPLIT reads it, in order.

    ``TEMP`` on a file without it but with potential temperature (``THET``) is
    at the levels that hold both ``THET`` and ``PRES`` (:func:`_read_field`).
    """
    every = range(len(recordset.vertical_axis.levels))
    levels = [level for level in every if (level, variable) in recordset]
    if not levels and variable == "TEMP":
        levels = [
            level
            for level in every
            if (level, "THET") in recordset and (level, "PRES") in recordset
        ]
    return tuple(levels)


def _read_field(
    recordset: TimeStep, variable: str, level: int, window: GridWindow
) -> npt.NDArray[Any]:
    """
    Read *variable* at *level* in *window* as HYSPLIT's ``metinp.f`` uses it.

    - ``TEMP`` on a file that stores ``THET`` instead (lines 579 to 584):
      ``THET * (PRES / 1000) ** 0.286``.
    - The upper-level winds of a mass-coupled WRF file (:func:`mass_coupled`;
      lines 360 to 386): ``UWND * MSFU / MUU0``, ``VWND * MSFV / MUV0`` and
      ``WWND * MSFT / 100``, its ``DIFW`` included (HYSPLIT scales the two
      before adding them; arlmet adds, then scales).
    """

    def read(name: str, at: int) -> npt.NDArray[Any]:
        """Read the stored field *name* at level *at* in the window."""
        return np.asarray(recordset[(at, name)].read(window=window), dtype=np.float32)

    if variable == "TEMP" and (level, "TEMP") not in recordset:
        return temperature_from_theta(read("THET", level), read("PRES", level))
    field = read(variable, level)
    if level > 0 and variable in _DECOUPLE and mass_coupled(recordset):
        missing = [n for n in ("MSFU", "MSFV", "MSFT") if (0, n) not in recordset]
        if missing:  # HYSPLIT stops too
            raise ValueError(
                "A mass-coupled WRF file (MUU0 and MUV0 at the surface) needs the "
                "map factors MSFU, MSFV and MSFT to decouple its winds; missing "
                f"{', '.join(missing)} at time {recordset.time}."
            )
        factor, mass = _DECOUPLE[variable]
        divisor = read(mass, 0) if isinstance(mass, str) else np.float32(mass)
        field = field * read(factor, 0) / divisor
    return field


def sample_levels(
    recordset: TimeStep,
    variable: str,
    plan: Plan,
    levels: Sequence[int] | None = None,
) -> npt.NDArray[Any] | None:
    """
    Sample *variable* at *levels* at each point: ``(n_points, n_levels)`` float32.

    *levels* defaults to every level that holds *variable*; a level that does
    not hold it is NaN. None when the time step does not hold *variable* at all.
    """
    held = levels_of(recordset, variable)
    if not held:
        return None
    levels = held if levels is None else levels
    # Filled one level at a time, so held as (n_levels, n_points) and returned transposed.
    out = np.full((len(levels), plan.n_points), np.nan, dtype=np.float32)
    if plan.window is not None:
        for pos, level in enumerate(levels):
            if level in held:
                field = _read_field(recordset, variable, level, plan.window)
                out[pos] = sample_field(field, plan)
    return out.T
