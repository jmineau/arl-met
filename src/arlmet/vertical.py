"""
Vertical coordinate helpers for ARL meteorology grids.

This module keeps vertical metadata separate from the horizontal grid model in
``arlmet.grid`` and provides lightweight helpers for deriving level
coordinates.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import override
else:

    def override(f: object) -> object:
        return f


import numpy as np
import numpy.typing as npt


class VerticalAxis:
    """
    Vertical coordinate metadata for an ARL file.

    Parameters
    ----------
    flag : int
        ARL vertical coordinate flag. Supported values include ``1`` (sigma),
        ``2`` (pressure), ``3`` (terrain-following), ``4`` (hybrid), and
        ``5`` (WRF).
    levels : sequence of float
        Native level values stored in the file.
    offset : float, default 0.0
        Pressure offset used by sigma and hybrid coordinate conversions.

    Attributes
    ----------
    flag : int
        Raw ARL vertical flag.
    coord_system : str
        Human-readable coordinate system name.
    levels : numpy.ndarray
        Copy of the stored level values.
    offset : float
        Pressure offset for sigma or hybrid coordinates.

    Methods
    -------
    calculate_coords()
        Return the native level coordinate values.
    sigma_to_pressure(surface_pressure, levels)
        Convert sigma or hybrid levels to pressure at sample points.

    Examples
    --------
    >>> from arlmet.vertical import VerticalAxis
    >>> axis = VerticalAxis(flag=2, levels=[1000.0, 925.0, 850.0])
    >>> axis.coord_system
    'pressure'
    >>> axis.calculate_coords()["level"].tolist()
    [1000.0, 925.0, 850.0]
    """

    FLAGS: dict[int, str] = {
        1: "sigma",
        2: "pressure",
        3: "terrain",
        4: "hybrid",
        5: "wrf",
    }

    def __init__(
        self,
        flag: int,
        levels: Sequence[float],
        *,
        offset: float = 0.0,
    ):
        self.flag = flag
        self._levels = np.asarray(levels, dtype=float)
        self.offset = float(offset)

    @property
    def coord_system(self) -> str:
        return self.FLAGS.get(self.flag, "unknown")

    @property
    def levels(self) -> np.ndarray:
        return self._levels.copy()

    def calculate_coords(self) -> dict[str, np.ndarray]:
        """Return the native level coordinate values stored in the file."""
        return {"level": self._levels.copy()}

    def sigma_to_pressure(
        self,
        surface_pressure: npt.ArrayLike,
        levels: Sequence[int],
    ) -> np.ndarray:
        """
        Compute per-point pressure at each level for sigma or hybrid axes.

        Matches HYSPLIT metlvl.f: PLEVEL = OFFSET + (SFCP - OFFSET) * HEIGHT(LL)

        Parameters
        ----------
        surface_pressure : array-like of shape (n_points,)
            Surface pressure in hPa at each sample point.
        levels : sequence of int
            Level indices into self.levels to compute pressure for.

        Returns
        -------
        np.ndarray of shape (n_points, n_levels)
        """
        lv = self._levels[list(levels)]
        sp = np.asarray(surface_pressure, dtype=float)
        if self.flag == 1:
            # sigma: p = p_top + (p_surface - p_top) * sigma
            return self.offset + (sp[:, None] - self.offset) * lv[None, :]
        if self.flag == 4:
            # hybrid: level encoded as floor_pressure + sigma_fraction
            floor_p = np.floor(lv)
            sigma = lv - floor_p
            p = sp[:, None] * sigma[None, :] + floor_p[None, :]
            if len(levels) > 0 and levels[0] == 0:
                p[:, 0] = sp  # first hybrid level is always surface
            return p
        raise ValueError(
            f"sigma_to_pressure() is only valid for flag=1 (sigma) or flag=4 (hybrid); "
            f"got flag={self.flag} ({self.coord_system})."
        )

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
        return f"VerticalAxis({self.coord_system}, n={len(self._levels)})"
