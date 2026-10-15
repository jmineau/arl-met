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
    "WrfAxis",
    "hypsometric_z_agl",
]

R_D = 287.05  # dry air gas constant [J/(kg·K)]
G = 9.80665  # standard gravity [m/s²]


def hypsometric_z_agl(
    pressure: npt.ArrayLike,
    surface_pressure: npt.ArrayLike,
    temperature: npt.ArrayLike,
    *,
    surface_temperature: npt.ArrayLike | None = None,
    level_axis: int = -1,
) -> npt.NDArray[Any]:
    """
    Height above ground level (m) at each level via the hypsometric equation.

    Pure NumPy helper shared by the xarray vertical helpers and point sampling,
    so vertical calculations are not tied to xarray. Each layer is
    ``(R_d / g) * T_mean * ln(p_below / p_above)`` thick.

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
        Temperature [K] at each level. Pass the virtual temperature for the
        heights HYSPLIT finds.
    surface_temperature : array-like, optional
        Temperature [K] at the ground, broadcastable to *temperature* with the
        level axis removed. When given, the first layer uses the mean of it and
        the first level's temperature, as HYSPLIT's ``prfsig`` does; otherwise
        the first level's temperature alone.
    level_axis : int, default -1
        Axis of *temperature* that indexes vertical levels.

    Returns
    -------
    numpy.ndarray
        Heights AGL [m], same shape as *temperature*. The first layer runs
        from ``surface_pressure`` to the first level; each layer above uses
        the mean temperature of its bounding levels.

    Examples
    --------
    Two levels above a 1000 hPa surface, with a 290 K ground:

    >>> import numpy as np
    >>> from arlmet.vertical import hypsometric_z_agl
    >>> z = hypsometric_z_agl(
    ...     [990.0, 950.0], 1000.0, [288.0, 286.0], surface_temperature=290.0
    ... )
    >>> np.round(z, 1)
    array([ 85. , 431.5])
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

    # Layer 0: from surface pressure to p[0], with the mean of the ground and
    # first-level temperatures, or the first level's alone.
    t0 = _take(temp_vals, 0)
    if surface_temperature is not None:
        t0 = 0.5 * (np.asarray(surface_temperature, dtype=float) + t0)
    dz0 = (R_D / G) * t0 * np.log(prss_vals / _take(p_vals, 0))
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


def _specific_humidity_from_relh(
    temperature: npt.NDArray[Any],
    relative_humidity: npt.NDArray[Any],
    pressure: npt.NDArray[Any],
) -> npt.NDArray[Any]:
    """Specific humidity [kg/kg] from RH [%], T [K] and p [hPa], as HYSPLIT's ``CALCSP``."""
    e_sat = np.exp(21.4 - 5351.0 / temperature)  # hPa, HYSPLIT's CALCESAT
    return 0.622 * (relative_humidity * e_sat / 100.0) / pressure


def _hysplit_z_agl(
    pressure: npt.ArrayLike,
    surface_pressure: npt.ArrayLike,
    temperature: npt.ArrayLike,
    *,
    first_sigma: npt.ArrayLike,
    surface_temperature: npt.ArrayLike | None = None,
    specific_humidity: npt.ArrayLike | None = None,
    relative_humidity: npt.ArrayLike | None = None,
    level_axis: int = -1,
) -> npt.NDArray[Any]:
    """
    Heights AGL [m] of the levels above ground, as HYSPLIT's ``prfsig``/``prfecm`` find them.

    The column starts at the ground (0 m, *surface_pressure*) and integrates the
    hypsometric equation with the virtual temperature ``T (1 + 0.61 q)``
    (``CALCTV``); each layer uses the mean of its two bounding levels.

    *pressure*, *temperature* and the humidity hold the levels above ground
    along *level_axis*. The ground temperature is *surface_temperature*
    (``T02M``), or without it the first level's temperature brought down
    adiabatically, ``T1 * first_sigma ** -0.286`` (``PRFSFCT``); HYSPLIT passes
    the first level's sigma (``PSG(1)``) for sigma files and its pressure over
    the surface pressure for hybrid files. Humidity is ``SPHU``, else ``RELH``
    converted with HYSPLIT's ``CALCSP``, else none (dry); a NaN humidity counts
    as none, as HYSPLIT does for a level without the field. The ground's
    humidity is the first level's.
    """
    t = np.asarray(temperature, dtype=float)
    p = np.asarray(pressure, dtype=float)
    p0 = np.asarray(surface_pressure, dtype=float)
    ax = level_axis % t.ndim
    t1 = np.take(t, 0, axis=ax)
    if surface_temperature is not None:
        t_ground = np.asarray(surface_temperature, dtype=float)
    else:
        t_ground = t1 * np.asarray(first_sigma, dtype=float) ** -0.286

    if specific_humidity is not None:
        q = np.nan_to_num(np.asarray(specific_humidity, dtype=float), nan=0.0)
        q_ground = np.take(q, 0, axis=ax)
    elif relative_humidity is not None:
        rh = np.nan_to_num(np.asarray(relative_humidity, dtype=float), nan=0.0)
        q = _specific_humidity_from_relh(t, rh, p)
        q_ground = _specific_humidity_from_relh(t_ground, np.take(rh, 0, axis=ax), p0)
    else:
        q = np.zeros_like(t)
        q_ground = np.zeros_like(t1)

    return hypsometric_z_agl(
        p,
        p0,
        t * (1.0 + 0.61 * q),
        surface_temperature=t_ground * (1.0 + 0.61 * q_ground),
        level_axis=level_axis,
    )


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
        Pressure offset [hPa] stored in the index record: the sigma offset,
        or WRF's model-top pressure for flag 5.

    Attributes
    ----------
    flag : int
        ARL vertical coordinate flag (1=sigma, 2=pressure, 3=terrain,
        4=hybrid, 5=WRF).
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
            If a required input is missing, or the levels alone do not give
            pressure (terrain-following and WRF axes; read the ``PRES`` field).
        """
        ...

    @abstractmethod
    def to_height_agl(
        self,
        *,
        surface_pressure: npt.ArrayLike | None = None,
        temperature: npt.ArrayLike | None = None,
        pressure: npt.ArrayLike | None = None,
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
            Surface pressure [hPa] (``PRSS``). Required for sigma, hybrid and
            WRF.
        temperature : array-like, optional
            Temperature [K] at each level (``TEMP``), levels on the last
            axis; the virtual temperature gives the heights HYSPLIT finds.
            Required for sigma, hybrid and WRF.
        pressure : array-like, optional
            Pressure [hPa] at each level (``PRES``), the same shape as
            ``temperature``. Required for WRF; sigma uses it in place of the
            nominal sigma pressure when given, as HYSPLIT's ``prfsig`` does;
            hybrid ignores it, as HYSPLIT's ``prfecm`` does.
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
    """
    Flag=1. Sigma coordinate — heights via hypsometric integration.

    The stored levels are sigma, with level 0 the surface. ``to_pressure`` gives
    the nominal pressure ``offset + (surface_pressure - offset) * sigma``. A file
    with a ``PRES`` field (HRRR, WRF converted by ``arw2arl``) holds each level's
    real pressure there, which can differ from the nominal one by tens of hPa
    aloft; HYSPLIT's ``prfsig`` then uses ``PRES``, and so do
    :func:`arlmet.sample_points`, :func:`arlmet.pressure` and
    :func:`arlmet.z_agl`. Pass it to ``to_height_agl`` as ``pressure``.

    Examples
    --------
    >>> import numpy as np
    >>> from arlmet import SigmaAxis
    >>> axis = SigmaAxis(levels=[1.0, 0.99, 0.95])
    >>> axis.to_pressure(surface_pressure=1000.0)
    array([1000.,  990.,  950.])
    >>> z = axis.to_height_agl(
    ...     surface_pressure=1000.0,
    ...     temperature=np.array([290.0, 288.0, 286.0]),
    ...     pressure=np.array([np.nan, 988.0, 945.0]),
    ... )
    >>> np.round(z, 1)
    array([  0. , 102.1, 475.9])
    """

    flag = 1
    coord_system = "sigma"

    @override
    def to_pressure(
        self, *, surface_pressure: npt.ArrayLike | None = None
    ) -> npt.NDArray[np.float64]:
        """Nominal pressure [hPa] from ``surface_pressure``: offset + (sp - offset) * sigma."""
        _require("surface_pressure", surface_pressure, self)
        sp = np.asarray(surface_pressure, dtype=float)
        return self.offset + (sp[..., None] - self.offset) * self.levels

    @override
    def to_height_agl(
        self,
        *,
        surface_pressure: npt.ArrayLike | None = None,
        temperature: npt.ArrayLike | None = None,
        pressure: npt.ArrayLike | None = None,
        hgts: npt.ArrayLike | None = None,
        terrain: npt.ArrayLike | None = None,
    ) -> npt.NDArray[np.float64]:
        """
        Height AGL [m] by hypsometric integration from ``surface_pressure`` with ``temperature``.

        ``temperature`` covers every level, level 0 (the ground) included.
        ``pressure`` (the ``PRES`` field, the same shape) replaces the nominal
        sigma pressure when given, as in HYSPLIT's ``prfsig``; its level 0 is
        taken from ``surface_pressure``. For HYSPLIT's heights pass the virtual
        temperature, with the ground temperature (``T02M``) at level 0.
        """
        if pressure is None:
            return _hypsometric_height(self, surface_pressure, temperature)
        _require("surface_pressure", surface_pressure, self)
        _require("temperature", temperature, self)
        assert surface_pressure is not None and temperature is not None
        sp = np.asarray(surface_pressure, dtype=float)
        p = np.array(pressure, dtype=float)
        p[..., 0] = sp
        return hypsometric_z_agl(p, sp, temperature)


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
        pressure: npt.ArrayLike | None = None,
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
        """Raise ValueError: terrain-following files have no pressure."""
        raise ValueError(
            "Terrain-following (flag=3) files have no pressure coordinate."
        )

    @override
    def to_height_agl(
        self,
        *,
        surface_pressure: npt.ArrayLike | None = None,
        temperature: npt.ArrayLike | None = None,
        pressure: npt.ArrayLike | None = None,
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
        pressure: npt.ArrayLike | None = None,
        hgts: npt.ArrayLike | None = None,
        terrain: npt.ArrayLike | None = None,
    ) -> npt.NDArray[np.float64]:
        """Height AGL [m] by hypsometric integration of ``surface_pressure`` and ``temperature``; ``pressure`` is ignored, as in HYSPLIT's ``prfecm``."""
        return _hypsometric_height(self, surface_pressure, temperature)


class WrfAxis(VerticalAxis):
    """
    Flag=5. WRF hybrid sigma-pressure levels, as HYSPLIT's ``arw2arl`` writes them.

    ``arw2arl`` writes flag 5 for a WRF run with ``hybrid_opt=2`` (flag 1
    otherwise). The stored levels are WRF's eta on the mass levels (``ZNU``),
    bottom up, with 1.0 for the surface, and ``offset`` is WRF's model-top
    pressure ``P_TOP`` [hPa].

    Eta alone does not give pressure on hybrid levels: that also takes WRF's
    hybrid coefficients, which ``arw2arl`` writes to a separate text file
    (``hybrid_wrfvcoords.txt``), not to the ARL file. So pressure is the
    ``PRES`` field, as in HYSPLIT's ``prfwrf``, and heights are hypsometric from
    ``PRSS`` up through ``PRES`` with ``TEMP``, as in HYSPLIT's ``prfsig``, which
    HYSPLIT uses for a flag-5 file when it does not have the coefficients.

    Examples
    --------
    >>> import numpy as np
    >>> from arlmet import WrfAxis
    >>> axis = WrfAxis(levels=[1.0, 0.995, 0.98], offset=50.0)
    >>> z = axis.to_height_agl(
    ...     surface_pressure=850.0,
    ...     temperature=np.array([285.0, 284.0]),
    ...     pressure=np.array([845.0, 832.0]),
    ... )
    >>> np.round(z, 1)
    array([ 49.2, 178.3])
    """

    flag = 5
    coord_system = "wrf"

    @override
    def to_pressure(
        self, *, surface_pressure: npt.ArrayLike | None = None
    ) -> npt.NDArray[np.float64]:
        """Raise ValueError: WRF hybrid levels need the ``PRES`` field for pressure."""
        raise ValueError(
            "WRF hybrid (flag=5) levels give pressure only with WRF's hybrid "
            "coefficients, which ARL files do not store; use the PRES field, "
            "as HYSPLIT does."
        )

    @override
    def to_height_agl(
        self,
        *,
        surface_pressure: npt.ArrayLike | None = None,
        temperature: npt.ArrayLike | None = None,
        pressure: npt.ArrayLike | None = None,
        hgts: npt.ArrayLike | None = None,
        terrain: npt.ArrayLike | None = None,
    ) -> npt.NDArray[np.float64]:
        """Height AGL [m] by hypsometric integration of ``pressure`` (PRES) from ``surface_pressure`` with ``temperature``."""
        _require("surface_pressure", surface_pressure, self)
        _require("temperature", temperature, self)
        _require("pressure", pressure, self)
        assert (
            surface_pressure is not None
            and temperature is not None
            and pressure is not None
        )
        return hypsometric_z_agl(pressure, surface_pressure, temperature)


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
    5: WrfAxis,
}
