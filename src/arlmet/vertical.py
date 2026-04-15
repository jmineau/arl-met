import importlib.resources
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import xarray as xr

from arlmet.grid import Grid, Projection, wrap_lons


@dataclass
class ASCDataConfig:
    ll_lat: float
    ll_lon: float
    dlat: float
    dlon: float
    nlat: int
    nlon: int
    land_use: int
    roughness: float
    data_dir: Path

    @classmethod
    def from_file(cls, file: str) -> "ASCDataConfig":
        """Load ASC terrain configuration bundled with the package."""
        resources = importlib.resources.files("arlmet.resources")
        with importlib.resources.as_file(resources / file) as path, open(path) as handle:
            lines = handle.readlines()

        if len(lines) < 6:
            raise ValueError("Configuration file must have at least 6 lines.")

        ll = lines[0].split()[:2]
        d = lines[1].split()[:2]
        n = lines[2].split()[:2]
        land_use = lines[3].split()[0]
        roughness = lines[4].split()[0]
        data_dir = lines[5].split()[0].strip("'\"")

        return cls(
            ll_lat=float(ll[0]),
            ll_lon=float(ll[1]),
            dlat=float(d[0]),
            dlon=float(d[1]),
            nlat=int(n[0]),
            nlon=int(n[1]),
            land_use=int(land_use),
            roughness=float(roughness),
            data_dir=path.parent / data_dir,
        )


class DefaultTerrain:
    def __init__(self, filename: str = "TERRAIN.ASC", config: str = "ASCDATA.CFG"):
        self.filename = filename
        self.config = (
            config
            if isinstance(config, ASCDataConfig)
            else ASCDataConfig.from_file(config)
        )
        self._data: np.ndarray | None = None

    @property
    def data(self) -> np.ndarray:
        if self._data is None:
            terrain_file = self.config.data_dir / self.filename
            if not terrain_file.exists():
                raise FileNotFoundError(f"Terrain file not found: {terrain_file}")

            data = np.genfromtxt(terrain_file, delimiter=4, dtype="f4")
            if data.shape != (self.config.nlat, self.config.nlon):
                raise ValueError("Terrain data shape does not match configuration.")
            self._data = data
        return self._data

    @property
    def lats(self) -> np.ndarray:
        lats = self.config.ll_lat + np.arange(self.config.nlat) * self.config.dlat
        return lats[::-1]

    @property
    def lons(self) -> np.ndarray:
        lons = self.config.ll_lon + np.arange(self.config.nlon) * self.config.dlon
        return wrap_lons(lons)

    def to_xarray(self) -> xr.DataArray:
        return xr.DataArray(
            self.data,
            dims=("lat", "lon"),
            coords={"lat": self.lats, "lon": self.lons},
            name="terrain",
            attrs={"units": "m", "description": "Terrain height"},
        ).sortby("lat")


class Surface:
    DEFAULT_PRESSURE = 1013.0

    def __init__(
        self,
        terrain: npt.ArrayLike | None = None,
        pressure: npt.ArrayLike | None = None,
        default_terrain: DefaultTerrain | None = None,
    ):
        self._terrain = terrain
        self._pressure = pressure
        self._default_terrain = default_terrain

    @property
    def terrain(self) -> np.ndarray | None:
        if self._terrain is not None:
            return np.asarray(self._terrain, dtype=float)
        if self._default_terrain is not None:
            return np.asarray(self._default_terrain.data, dtype=float)
        return None

    @property
    def pressure(self) -> np.ndarray:
        if self._pressure is None:
            return np.asarray(self.DEFAULT_PRESSURE, dtype=float)
        return np.asarray(self._pressure, dtype=float)

    @property
    def mean_pressure(self) -> float:
        return float(np.asarray(self.pressure, dtype=float).mean())

    @property
    def mean_terrain(self) -> float:
        terrain = self.terrain
        if terrain is None:
            return 0.0
        return float(np.asarray(terrain, dtype=float).mean())


class VerticalAxis:
    FLAGS: dict[int, str] = {
        1: "sigma",
        2: "pressure",
        3: "terrain",
        4: "hybrid",
        5: "wrf",
    }

    Z_PHI1 = np.asarray((17.98, 14.73, 13.09, 11.98, 11.15, 10.52, 10.04, 9.75, 9.88))
    Z_PHI2 = np.asarray((31.37, 27.02, 24.59, 22.92, 21.65, 20.66, 19.83, 19.13, 18.51))

    def __init__(
        self,
        flag: int,
        heights: Sequence[float] | None = None,
        *,
        levels: Sequence[float] | None = None,
        offset: float = 0.0,
        surface: Surface | None = None,
    ):
        if heights is None:
            if levels is None:
                raise TypeError("VerticalAxis requires `heights` or `levels`.")
            heights = levels
        elif levels is not None:
            raise TypeError("Pass only one of `heights` or `levels`.")

        self.flag = flag
        self._heights = np.asarray(heights, dtype=float)
        self.offset = float(offset)
        self.surface = surface

    @property
    def coord_system(self) -> str:
        return self.FLAGS.get(self.flag, "unknown")

    @property
    def dims(self) -> tuple[str]:
        return ("level",)

    @property
    def heights(self) -> np.ndarray:
        return self._heights.copy()

    @property
    def levels(self) -> np.ndarray:
        return self.heights

    def with_surface(self, surface: Surface | None) -> "VerticalAxis":
        return VerticalAxis(
            flag=self.flag,
            heights=self._heights,
            offset=self.offset,
            surface=surface,
        )

    def calculate_coords(self) -> dict[str, np.ndarray]:
        coords = {
            "level": np.arange(len(self._heights), dtype=int),
            "height": self.height_agl(),
        }
        pressure = self.pressure()
        if pressure is not None:
            coords["pressure"] = pressure
        if self.flag == 1:
            coords["sigma"] = self.heights
        return coords

    def pressure(self) -> np.ndarray | None:
        p0 = self._surface_pressure()

        if self.flag == 1:
            return self.offset + (p0 - self.offset) * self._heights
        if self.flag == 2:
            return self.heights
        if self.flag == 3:
            return None
        if self.flag == 4:
            offsets = np.floor(self._heights)
            sigma = self._heights - offsets
            pressure = p0 * sigma + offsets
            if pressure.size:
                pressure[0] = p0
            return pressure
        if self.flag == 5:
            raise NotImplementedError("WRF hybrid vertical coordinates are not implemented.")
        return None

    def height_agl(self) -> np.ndarray:
        if self.flag == 3:
            return self.heights

        pressure = self.pressure()
        if pressure is None:
            raise ValueError(
                f"Cannot derive height for vertical coordinate system '{self.coord_system}'."
            )
        return self._p_to_z(pressure)

    def height_msl(self) -> np.ndarray:
        return self.height_agl() + self._surface_terrain()

    def _surface_pressure(self) -> float:
        if self.surface is None:
            return Surface.DEFAULT_PRESSURE
        return self.surface.mean_pressure

    def _surface_terrain(self) -> float:
        if self.surface is None:
            return 0.0
        return self.surface.mean_terrain

    def _p_to_z(self, p: npt.ArrayLike) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        delta_z = np.zeros_like(p, dtype=float)

        mask1 = p >= 100.0
        if np.any(mask1):
            idx1 = np.clip((p[mask1] // 100).astype(int), 0, len(self.Z_PHI1) - 1)
            delta_z[mask1] = self.Z_PHI1[idx1]

        mask2 = ~mask1
        if np.any(mask2):
            idx2 = np.clip((p[mask2] // 10).astype(int), 0, len(self.Z_PHI2) - 1)
            delta_z[mask2] = self.Z_PHI2[idx2]

        z = (self._surface_pressure() - p) * delta_z
        return np.where(z < 0.0, 0.0, z)

    def __eq__(self, other) -> bool:
        if not isinstance(other, VerticalAxis):
            return False
        return (
            self.flag == other.flag
            and self.offset == other.offset
            and np.array_equal(self._heights, other._heights)
        )

    def __hash__(self) -> int:
        return hash((self.flag, self.offset, tuple(self._heights)))


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
        coords["level"] = vcoords["level"]
        coords["height"] = ("level", vcoords["height"])
        if "pressure" in vcoords:
            coords["pressure"] = ("level", vcoords["pressure"])
        if "sigma" in vcoords:
            coords["sigma"] = ("level", vcoords["sigma"])
        return coords
