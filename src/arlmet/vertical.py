from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from arlmet.grid import Grid, Projection


class VerticalAxis:
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

    def __eq__(self, other) -> bool:
        if not isinstance(other, VerticalAxis):
            return False
        return (
            self.flag == other.flag
            and self.offset == other.offset
            and np.array_equal(self._levels, other._levels)
        )

    def __hash__(self) -> int:
        return hash((self.flag, self.offset, tuple(self._levels)))


class Grid3D(Grid):
    def __init__(
        self,
        projection: Projection | None = None,
        nx: int = 0,
        ny: int = 0,
        vertical_axis: VerticalAxis | None = None,
        *,
        proj: Projection | None = None,
    ):
        projection = projection or proj
        if projection is None:
            raise TypeError("Grid3D requires `projection` or `proj`.")
        if vertical_axis is None:
            raise TypeError("Grid3D requires a `vertical_axis`.")

        super().__init__(projection=projection, nx=nx, ny=ny)
        self.vertical_axis = vertical_axis

    @property
    def dims(self) -> tuple[str, ...]:
        return ("level", *super().dims)

    def calculate_coords(self) -> dict[str, object]:
        coords = super().calculate_coords()
        vcoords = self.vertical_axis.calculate_coords()
        coords["level"] = ("level", vcoords["level"])
        return coords
