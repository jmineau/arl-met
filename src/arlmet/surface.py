"""Surface-state and terrain helpers for ARL meteorology."""

import importlib.resources
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import xarray as xr

from arlmet.grid import wrap_lons


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
