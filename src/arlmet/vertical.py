"""
Vertical coordinate helpers for ARL meteorology grids.

This module keeps vertical metadata separate from the horizontal grid model in
``arlmet.grid`` and provides lightweight helpers for deriving level
coordinates.
"""

from abc import ABC, abstractmethod
from dataclasses import FrozenInstanceError
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
from typing_extensions import override

__all__ = [
    "VerticalAxis",
    "SigmaAxis",
    "PressureAxis",
    "TerrainAxis",
    "HybridAxis",
    "hypsometric_z_agl",
]

R_D = 287.05  # dry air gas constant [J/(kg·K)]
G = 9.80665  # standard gravity [m/s²]


def hypsometric_z_agl(
    pressure: npt.ArrayLike,
    surface_pressure: npt.ArrayLike,
    temperature: npt.ArrayLike,
    *,
    level_axis: int = -1,
) -> npt.NDArray[Any]:
    """
    Height above ground level (m) at each level via the hypsometric equation.

    Pure NumPy helper shared by the xarray vertical helpers and point sampling,
    so vertical calculations are not tied to xarray.

    Parameters
    ----------
    pressure : array-like
        Pressure at each level [hPa], ordered from high to low pressure
        (surface to top) along ``level_axis``. Either the same shape as
        *temperature*, or 1-D ``(nlev,)`` to broadcast across the other axes.
    surface_pressure : array-like
        Surface pressure [hPa], broadcastable to *temperature* with the level
        axis removed.
    temperature : array-like
        Temperature [K] at each level.
    level_axis : int, default -1
        Axis of *temperature* that indexes vertical levels.

    Returns
    -------
    numpy.ndarray
        Heights AGL [m], same shape as *temperature*. The first level is
        integrated from ``surface_pressure`` to the first level using that
        level's temperature; each layer above uses the mean temperature of its
        bounding levels.
    """
    temp_vals = np.asarray(temperature, dtype=float)
    prss_vals = np.asarray(surface_pressure, dtype=float)
    level_ax = level_axis % temp_vals.ndim
    nlev = temp_vals.shape[level_ax]

    # Broadcast 1-D pressure to match temperature along level_ax.
    p_vals = np.asarray(pressure, dtype=float)
    if p_vals.ndim == 1:
        expand_axes = [i for i in range(temp_vals.ndim) if i != level_ax]
        for ax in sorted(expand_axes):
            p_vals = np.expand_dims(p_vals, ax)
        p_vals = np.broadcast_to(p_vals, temp_vals.shape)

    def _take(arr: npt.NDArray[Any], i: int) -> npt.NDArray[Any]:
        """Select level ``i`` of *arr* along the level axis."""
        idx: list[int | slice] = [slice(None)] * arr.ndim
        idx[level_ax] = i
        return arr[tuple(idx)]

    def _take_range(
        arr: npt.NDArray[Any], start: int | None, stop: int | None
    ) -> npt.NDArray[Any]:
        """Slice levels ``start:stop`` of *arr* along the level axis."""
        idx: list[int | slice | None] = [slice(None)] * arr.ndim
        idx[level_ax] = slice(start, stop)
        return arr[tuple(idx)]

    # Layer 0: from surface pressure to p[0], using T[0] as representative.
    dz0 = (R_D / G) * _take(temp_vals, 0) * np.log(prss_vals / _take(p_vals, 0))
    dz0_exp = np.expand_dims(dz0, level_ax)

    if nlev > 1:
        t_mean = (
            _take_range(temp_vals, None, -1) + _take_range(temp_vals, 1, None)
        ) / 2.0
        dz_layers = (
            (R_D / G)
            * t_mean
            * np.log(_take_range(p_vals, None, -1) / _take_range(p_vals, 1, None))
        )
        dz_all = np.concatenate([dz0_exp, dz_layers], axis=level_ax)
    else:
        dz_all = dz0_exp

    return np.cumsum(dz_all, axis=level_ax)


def _require(name: str, value: npt.ArrayLike | None, axis: "VerticalAxis") -> None:
    """Raise a clear ValueError when an input *axis* needs was not given."""
    if value is None:
        raise ValueError(
            f"{type(axis).__name__} (flag={axis.flag}) requires {name}= for this "
            "conversion."
        )


class VerticalAxis(ABC):
    """
    Abstract base class for ARL vertical coordinate axes.

    Use :meth:`from_flag` to construct from a raw ARL flag integer, or
    instantiate a subclass directly (e.g. ``PressureAxis(levels=[...])``).

    Vertical axes are immutable value objects: ``levels`` is a read-only
    NumPy array, attributes cannot be reassigned, and equal axes hash
    equally. Build a new axis to change one.

    Parameters
    ----------
    levels : array-like of float
        Native level values stored in the file (1-D).
    offset : float, default 0.0
        Pressure offset used by sigma and hybrid coordinate conversions.

    Attributes
    ----------
    flag : int
        ARL vertical coordinate flag (1=sigma, 2=pressure, 3=terrain,
        4=hybrid).
    coord_system : str
        Human-readable coordinate system name.
    levels : numpy.ndarray
        Read-only ``float64`` array of native level values.
    offset : float
        Pressure offset stored in the index record.
    """

    flag: ClassVar[int]
    coord_system: ClassVar[str]

    levels: npt.NDArray[np.float64]
    offset: float

    def __init__(
        self,
        levels: npt.ArrayLike,
        *,
        offset: float = 0.0,
    ):
        arr = np.array(levels, dtype=float)
        if arr.ndim != 1:
            raise ValueError(f"levels must be 1-D, got shape {arr.shape}.")
        arr.setflags(write=False)
        object.__setattr__(self, "levels", arr)
        object.__setattr__(self, "offset", float(offset))

    @override
    def __setattr__(self, name: str, value: object) -> None:
        raise FrozenInstanceError(f"cannot assign to field {name!r}")

    @override
    def __delattr__(self, name: str) -> None:
        raise FrozenInstanceError(f"cannot delete field {name!r}")

    @classmethod
    def from_flag(
        cls,
        flag: int,
        levels: npt.ArrayLike,
        *,
        offset: float = 0.0,
    ) -> "VerticalAxis":
        """Construct the appropriate subclass from an ARL vertical flag."""
        subclass = _FLAG_MAP.get(flag)
        if subclass is None:
            raise ValueError(
                f"Unsupported vertical flag {flag}. "
                f"Supported flags: {sorted(_FLAG_MAP)}."
            )
        return subclass(levels=levels, offset=offset)

    @abstractmethod
    def to_pressure(
        self, *, surface_pressure: npt.ArrayLike | None = None
    ) -> npt.NDArray[np.float64]:
        """
        Compute pressure [hPa] at each level.

        Parameters
        ----------
        surface_pressure : array-like, optional
            Surface pressure [hPa] (``PRSS``). Required for sigma and hybrid
            axes; ignored by pressure axes.

        Returns
        -------
        numpy.ndarray
            ``(nlev,)`` for pressure axes, or ``surface_pressure.shape +
            (nlev,)`` for sigma and hybrid axes.

        Raises
        ------
        ValueError
            If a required input is missing, or the axis has no pressure
            coordinate (terrain-following).
        """
        ...

    @abstractmethod
    def to_height_agl(
        self,
        *,
        surface_pressure: npt.ArrayLike | None = None,
        temperature: npt.ArrayLike | None = None,
        hgts: npt.ArrayLike | None = None,
        terrain: npt.ArrayLike | None = None,
    ) -> npt.NDArray[np.float64]:
        """
        Compute height above ground level [m] at each level.

        Each axis uses only the inputs its coordinate system needs (matching
        HYSPLIT's ``prfcom``) and ignores the rest.

        Parameters
        ----------
        surface_pressure : array-like, optional
            Surface pressure [hPa] (``PRSS``). Required for sigma and hybrid.
        temperature : array-like, optional
            Temperature [K] at each level (``TEMP``), levels on the last
            axis. Required for sigma and hybrid.
        hgts : array-like, optional
            Geopotential height [m MSL] at each level (``HGTS``). Required
            for pressure axes.
        terrain : array-like, optional
            Terrain height [m] (``SHGT``), broadcastable to ``hgts``.
            Required for pressure axes.

        Returns
        -------
        numpy.ndarray
            Heights AGL [m].

        Raises
        ------
        ValueError
            If an input this axis needs is missing.
        """
        ...

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VerticalAxis):
            return False
        return (
            self.flag == other.flag
            and self.offset == other.offset
            and np.array_equal(self.levels, other.levels)
        )

    @override
    def __hash__(self) -> int:
        return hash((self.flag, self.offset, tuple(self.levels.tolist())))

    def __len__(self) -> int:
        return len(self.levels)

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}(n={len(self.levels)})"


class SigmaAxis(VerticalAxis):
    """Flag=1. Sigma coordinate — heights via hypsometric integration."""

    flag = 1
    coord_system = "sigma"

    @override
    def to_pressure(
        self, *, surface_pressure: npt.ArrayLike | None = None
    ) -> npt.NDArray[np.float64]:
        """Pressure [hPa] from ``surface_pressure``: offset + (sp - offset) * sigma."""
        _require("surface_pressure", surface_pressure, self)
        sp = np.asarray(surface_pressure, dtype=float)
        return self.offset + (sp[..., None] - self.offset) * self.levels

    @override
    def to_height_agl(
        self,
        *,
        surface_pressure: npt.ArrayLike | None = None,
        temperature: npt.ArrayLike | None = None,
        hgts: npt.ArrayLike | None = None,
        terrain: npt.ArrayLike | None = None,
    ) -> npt.NDArray[np.float64]:
        """Height AGL [m] by hypsometric integration of ``surface_pressure`` and ``temperature``."""
        return _hypsometric_height(self, surface_pressure, temperature)


class PressureAxis(VerticalAxis):
    """Flag=2. Stored levels are pressures. Heights come from HGTS."""

    flag = 2
    coord_system = "pressure"

    @override
    def to_pressure(
        self, *, surface_pressure: npt.ArrayLike | None = None
    ) -> npt.NDArray[np.float64]:
        """Pressure [hPa]: a writable copy of the stored levels. No inputs needed."""
        return self.levels.copy()

    @override
    def to_height_agl(
        self,
        *,
        surface_pressure: npt.ArrayLike | None = None,
        temperature: npt.ArrayLike | None = None,
        hgts: npt.ArrayLike | None = None,
        terrain: npt.ArrayLike | None = None,
    ) -> npt.NDArray[np.float64]:
        """Height AGL [m] as ``hgts`` (geopotential height, HGTS) minus ``terrain``."""
        _require("hgts", hgts, self)
        _require("terrain", terrain, self)
        return np.asarray(hgts, dtype=float) - np.asarray(terrain, dtype=float)


class TerrainAxis(VerticalAxis):
    """Flag=3. Terrain-following — stored levels are heights AGL."""

    flag = 3
    coord_system = "terrain"

    @override
    def to_pressure(
        self, *, surface_pressure: npt.ArrayLike | None = None
    ) -> npt.NDArray[np.float64]:
        """Always raises ValueError: terrain-following files have no pressure."""
        raise ValueError(
            "Terrain-following (flag=3) files have no pressure coordinate."
        )

    @override
    def to_height_agl(
        self,
        *,
        surface_pressure: npt.ArrayLike | None = None,
        temperature: npt.ArrayLike | None = None,
        hgts: npt.ArrayLike | None = None,
        terrain: npt.ArrayLike | None = None,
    ) -> npt.NDArray[np.float64]:
        """Height AGL [m]: a writable copy of the stored levels. No inputs needed."""
        return self.levels.copy()


class HybridAxis(VerticalAxis):
    """Flag=4. ECMWF hybrid sigma-pressure — pressure then hypsometric."""

    flag = 4
    coord_system = "hybrid"

    @override
    def to_pressure(
        self, *, surface_pressure: npt.ArrayLike | None = None
    ) -> npt.NDArray[np.float64]:
        """Pressure [hPa] from ``surface_pressure``: sp * sigma + floor(level)."""
        _require("surface_pressure", surface_pressure, self)
        sp = np.asarray(surface_pressure, dtype=float)
        floor_p = np.floor(self.levels)
        sigma = self.levels - floor_p
        p = sp[..., None] * sigma + floor_p
        p[..., 0] = sp  # first hybrid level is always surface
        return p

    @override
    def to_height_agl(
        self,
        *,
        surface_pressure: npt.ArrayLike | None = None,
        temperature: npt.ArrayLike | None = None,
        hgts: npt.ArrayLike | None = None,
        terrain: npt.ArrayLike | None = None,
    ) -> npt.NDArray[np.float64]:
        """Height AGL [m] by hypsometric integration of ``surface_pressure`` and ``temperature``."""
        return _hypsometric_height(self, surface_pressure, temperature)


def _hypsometric_height(
    axis: SigmaAxis | HybridAxis,
    surface_pressure: npt.ArrayLike | None,
    temperature: npt.ArrayLike | None,
) -> npt.NDArray[np.float64]:
    """Shared sigma/hybrid ``to_height_agl``: pressure, then hypsometric heights."""
    _require("surface_pressure", surface_pressure, axis)
    _require("temperature", temperature, axis)
    assert surface_pressure is not None and temperature is not None
    p = axis.to_pressure(surface_pressure=surface_pressure)
    return hypsometric_z_agl(p, surface_pressure, temperature)


# Registry for from_flag
_FLAG_MAP: dict[int, type[VerticalAxis]] = {
    1: SigmaAxis,
    2: PressureAxis,
    3: TerrainAxis,
    4: HybridAxis,
}
