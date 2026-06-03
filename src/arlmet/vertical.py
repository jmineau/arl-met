"""
Vertical coordinate helpers for ARL meteorology grids.

This module keeps vertical metadata separate from the horizontal grid model in
``arlmet.grid`` and provides lightweight helpers for deriving level
coordinates.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from typing_extensions import override

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
        idx: list[int | slice] = [slice(None)] * arr.ndim
        idx[level_ax] = i
        return arr[tuple(idx)]

    def _take_range(
        arr: npt.NDArray[Any], start: int | None, stop: int | None
    ) -> npt.NDArray[Any]:
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


class VerticalAxis(ABC):
    """
    Abstract base class for ARL vertical coordinate axes.

    Use :meth:`from_flag` to construct from a raw ARL flag integer, or
    instantiate a subclass directly (e.g. ``PressureAxis(levels=[...])``).

    Parameters
    ----------
    levels : sequence of float
        Native level values stored in the file.
    offset : float, default 0.0
        Pressure offset used by sigma and hybrid coordinate conversions.
    """

    flag: int
    coord_system: str

    def __init__(
        self,
        levels: Sequence[float],
        *,
        offset: float = 0.0,
    ):
        self._levels = np.asarray(levels, dtype=float)
        self.offset = float(offset)

    @classmethod
    def from_flag(
        cls,
        flag: int,
        levels: Sequence[float],
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

    @property
    def levels(self) -> npt.NDArray[Any]:
        return self._levels.copy()

    def calculate_coords(self) -> dict[str, npt.NDArray[Any]]:
        """Return the native level coordinate values stored in the file."""
        return {"level": self._levels.copy()}

    @abstractmethod
    def to_pressure(self, **kwargs: Any) -> npt.NDArray[Any]:
        """Compute pressure [hPa] at each level."""
        ...

    @abstractmethod
    def to_height_agl(self, **kwargs: Any) -> npt.NDArray[Any]:
        """Compute height above ground level [m] at each level."""
        ...

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VerticalAxis):
            return False
        return (
            self.flag == other.flag
            and self.offset == other.offset
            and np.array_equal(self._levels, other._levels)
        )

    def __hash__(self) -> int:
        return hash((self.flag, self.offset, tuple(self._levels)))

    def __len__(self) -> int:
        return len(self._levels)

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}(n={len(self._levels)})"


class SigmaAxis(VerticalAxis):
    """Flag=1. Sigma coordinate — heights via hypsometric integration."""

    flag = 1
    coord_system = "sigma"

    @override
    def to_pressure(self, **kwargs: Any) -> npt.NDArray[Any]:
        sp = np.asarray(kwargs["surface_pressure"], dtype=float)
        return self.offset + (sp[..., None] - self.offset) * self._levels

    @override
    def to_height_agl(self, **kwargs: Any) -> npt.NDArray[Any]:
        p = self.to_pressure(surface_pressure=kwargs["surface_pressure"])
        return hypsometric_z_agl(p, kwargs["surface_pressure"], kwargs["temperature"])


class PressureAxis(VerticalAxis):
    """Flag=2. Stored levels are pressures. Heights come from HGTS."""

    flag = 2
    coord_system = "pressure"

    @override
    def to_pressure(self, **kwargs: Any) -> npt.NDArray[Any]:
        return self._levels.copy()

    @override
    def to_height_agl(self, **kwargs: Any) -> npt.NDArray[Any]:
        return np.asarray(kwargs["hgts"], dtype=float) - np.asarray(
            kwargs["terrain"], dtype=float
        )


class TerrainAxis(VerticalAxis):
    """Flag=3. Terrain-following — stored levels are heights AGL."""

    flag = 3
    coord_system = "terrain"

    @override
    def to_pressure(self, **kwargs: Any) -> npt.NDArray[Any]:
        raise ValueError(
            "Terrain-following (flag=3) files have no pressure coordinate."
        )

    @override
    def to_height_agl(self, **kwargs: Any) -> npt.NDArray[Any]:
        return self._levels.copy()


class HybridAxis(VerticalAxis):
    """Flag=4. ECMWF hybrid sigma-pressure — pressure then hypsometric."""

    flag = 4
    coord_system = "hybrid"

    @override
    def to_pressure(self, **kwargs: Any) -> npt.NDArray[Any]:
        sp = np.asarray(kwargs["surface_pressure"], dtype=float)
        floor_p = np.floor(self._levels)
        sigma = self._levels - floor_p
        p = sp[..., None] * sigma + floor_p
        p[..., 0] = sp  # first hybrid level is always surface
        return p

    @override
    def to_height_agl(self, **kwargs: Any) -> npt.NDArray[Any]:
        p = self.to_pressure(surface_pressure=kwargs["surface_pressure"])
        return hypsometric_z_agl(p, kwargs["surface_pressure"], kwargs["temperature"])


# Registry for from_flag
_FLAG_MAP: dict[int, type[VerticalAxis]] = {
    1: SigmaAxis,
    2: PressureAxis,
    3: TerrainAxis,
    4: HybridAxis,
}
