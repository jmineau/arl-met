import importlib.resources
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from arlmet.grid import Grid, Projection


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
        """Load ASCDataConfig from a configuration file."""
        resources = importlib.resources.files("arlmet.resources")
        with importlib.resources.as_file(resources / file) as path:
            with open(path) as f:
                lines = f.readlines()

            if len(lines) < 6:
                raise ValueError("Configuration file must have at least 6 lines.")

            ll = lines[0].split()[:2]
            d = lines[1].split()[:2]
            n = lines[2].split()[:2]
            land_use = lines[3].split()[0]
            roughness = lines[4].split()[0]
            data_dir = lines[5].split()[0].strip("'\"")

            # Get data_dir relative to config file location
            data_dir = path.parent / data_dir

        return cls(
            ll_lat=float(ll[0]),
            ll_lon=float(ll[1]),
            dlat=float(d[0]),
            dlon=float(d[1]),
            nlat=int(n[0]),
            nlon=int(n[1]),
            land_use=int(land_use),
            roughness=float(roughness),
            data_dir=data_dir,
        )


class DefaultTerrain:
    def __init__(self, file="TERRAIN.ASC", config="ASCDATA.CFG"):
        self.config = ASCDataConfig.from_file(config)
        self.data = self._load(file)

    def _load(self, file):
        terrain_file = self.config.data_dir / file
        if not terrain_file.exists():
            raise FileNotFoundError(f"Terrain file not found: {terrain_file}")

        data = np.genfromtxt(terrain_file, delimiter=4, dtype="f4")
        if data.shape != (self.config.nlat, self.config.nlon):
            raise ValueError("Terrain data shape does not match configuration.")
        return data

    @property
    def lats(self):
        lats = self.config.ll_lat + np.arange(self.config.nlat) * self.config.dlat
        return lats[::-1]  # flip to north-to-south

    @property
    def lons(self):
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
    SFCP = 1013.0  # Default surface pressure in hPa used by HYSPLIT if none provided

    def __init__(
        self,
        level: int,
        terrain: np.ndarray | None = None,
        pressure: float | np.ndarray | None = None,
    ):
        """
        Represents a surface description used by VerticalAxis.

        Parameters
        ----------
        level : int
            Surface level index (kept for API compatibility).
        terrain : Optional[np.ndarray]
            Optional 2D terrain heights (ny, nx) in meters. If None, DefaultTerrain
            will be used lazily via the `default` property.
        pressure : Optional[float | np.ndarray]
            Surface pressure in hPa. May be a scalar, 1D or 2D array. If None,
            VerticalAxis will fall back to VerticalAxis.SURFACE_PRESSURE.
        """
        self.level = level
        self.terrain = terrain
        self.pressure = pressure or self.SFCP

        self._default = None

    @property
    def default(self) -> DefaultTerrain:
        if self._default is None:
            # read TERRAIN.ASC for terrain heights
            self._default = DefaultTerrain()
        return self._default


class VerticalAxis:
    """
    Represents the vertical axis of the ARL data.

    Parameters
    ----------
    flag : int
        Vertical coordinate system type (1=sigma, 2=pressure, 3=terrain, 4=hybrid, 5=wrfhybrid).
    heights : Sequence[float]
        Sequence of vertical heights (drec heights) corresponding to the vertical coordinate system.
    surface : Surface
        Surface object containing terrain and surface pressure information.
    kwargs :
        Additional keyword arguments:
          - sigma_offset or grid_dummy : float used as "grid dummy" (default 0.0)
          - wrfvcoords : sequence for WRF hybrid (z_flag==5)
    """

    FLAGS = {
        1: "sigma",  # fraction
        2: "pressure",  # mb/hPa
        3: "terrain",  # fraction/height
        4: "hybrid",  # mb: offset.fraction
        5: "wrfhybrid",  # WRF hybrid
    }

    # meters per hPa by 100 hPa intervals (100 to 900)
    Z_PHI1 = (17.98, 14.73, 13.09, 11.98, 11.15, 10.52, 10.04, 9.75, 9.88)
    # meters per hPa by 10 hPa intervals (10 to 90)
    Z_PHI2 = (31.37, 27.02, 24.59, 22.92, 21.65, 20.66, 19.83, 19.13, 18.51)

    def __init__(self, flag: int, heights: Sequence[float], surface: Surface, **kwargs):
        self.flag = flag
        self.heights = np.asarray(heights, dtype=float)
        self.surface = surface
        self.kwargs = kwargs

    @property
    def coord_system(self) -> str:
        return VerticalAxis.FLAGS.get(self.flag, "unknown")

    def _p_to_z(self, p: float | np.ndarray) -> np.ndarray:
        """
        Vectorized conversion pressure -> approximate height using HYSPLIT lookup tables.
        p: array-like pressures (hPa). Accepted shapes:
           - scalar
           - 1D (nl,) interpreted as level-only and broadcast over horizontal grid
           - 3D (nl, ny, nx) explicit
        Returns heights in meters with shape (nl, ny, nx) matching broadcasting.
        """
        p = np.asarray(p, dtype=float)

        # normalize p to shape (nl, ny, nx)
        if p.ndim == 0:
            p = p.reshape((1, 1, 1))
        elif p.ndim == 1:
            p = p.reshape((p.size, 1, 1))
        elif p.ndim == 2:
            # treat as (nl, nx) -> promote to (nl, ny=1, nx)
            p = p.reshape((p.shape[0], p.shape[1], 1))

        delta_z = np.zeros_like(p, dtype=float)

        mask1 = p >= 100.0
        if np.any(mask1):
            idx1 = np.clip((p[mask1] // 100).astype(int), 0, 8)
            delta_z[mask1] = np.array(self.Z_PHI1)[idx1]

        mask2 = ~mask1
        if np.any(mask2):
            idx2 = np.clip((p[mask2] // 10).astype(int), 0, 8)
            delta_z[mask2] = np.array(self.Z_PHI2)[idx2]

        # determine surface pressure p0 (broadcast to p shape)
        p0 = self.surface.pressure
        if p0 is None:
            p0 = np.array(self.SURFACE_PRESSURE, dtype=float)
        else:
            p0 = np.asarray(p0, dtype=float)

        # ensure p0 shape is (ny, nx)
        if p0.ndim == 0:
            p0 = p0.reshape((1, 1))
        elif p0.ndim == 1:
            p0 = p0.reshape((1, p0.size))
        elif p0.ndim > 2:
            # unexpected, but try to accept (ny, nx)
            p0 = p0.squeeze()

        p0_b = p0[None, :, :]  # (1, ny, nx)

        z = (p0_b - p) * delta_z
        return z

    def _calculate_coords(self) -> dict[str, tuple[tuple[str, ...], np.ndarray]]:
        """
        Build vertical coordinates dictionary similar to HYSPLIT:
        returns mapping name -> ((dims...), ndarray).

        Dimensions for 3D arrays are ('level','y','x'); for 1D arrays ('level',).
        """
        heights = self.heights
        nlvl = heights.size

        # prepare surface pressure array p0 with shape (ny, nx)
        p0 = self.surface.pressure
        if p0 is None:
            p0 = np.array(self.SURFACE_PRESSURE, dtype=float)
        p0 = np.asarray(p0, dtype=float)
        if p0.ndim == 0:
            p0 = p0.reshape((1, 1))
        elif p0.ndim == 1:
            p0 = p0.reshape((1, p0.size))
        # now p0.shape == (ny, nx)

        ny, nx = p0.shape

        # canonical 1D pressure level (plevel_1d)
        sfcp = float(p0.mean())
        plevel_1d = None

        z_flag = self.flag
        grid_dummy = self.grid_dummy

        if z_flag == 1:  # sigma
            plevel_1d = grid_dummy + (sfcp - grid_dummy) * heights
        elif z_flag == 2:  # pressure
            plevel_1d = heights.copy()
            if nlvl >= 1:
                plevel_1d[0] = sfcp
        elif z_flag == 3:  # terrain
            plevel_1d = None
        elif z_flag == 4:  # hybrid
            offsets = np.floor(heights)
            psig = heights - offsets
            plevel_1d = sfcp * psig + offsets
            if nlvl >= 1:
                plevel_1d[0] = sfcp
        elif z_flag == 5:  # wrf hybrid
            if self.wrfvcoords is None:
                raise ValueError("wrfvcoords required for WRF hybrid (z_flag==5)")
            eta = np.asarray([row[6] for row in self.wrfvcoords], dtype=float)
            plevel_1d = (
                eta * (sfcp - grid_dummy)
                + ((heights - eta) * (1000.0 - grid_dummy))
                + grid_dummy
            )
        else:
            raise ValueError(f"unsupported z_flag: {z_flag}")

        # build 3D plevel where appropriate
        plevel_3d = None
        if z_flag != 3:
            if z_flag == 1:
                sigma = heights.reshape((nlvl, 1, 1))
                plevel_3d = grid_dummy + (p0[None, :, :] - grid_dummy) * sigma
            elif z_flag == 2:
                plevel_3d = plevel_1d.reshape((nlvl, 1, 1)) + np.zeros((nlvl, ny, nx))
            elif z_flag == 4:
                offsets = np.floor(heights).reshape((nlvl, 1, 1))
                psig = (heights - np.floor(heights)).reshape((nlvl, 1, 1))
                plevel_3d = p0[None, :, :] * psig + offsets
                if nlvl >= 1:
                    plevel_3d[0, :, :] = p0
            elif z_flag == 5:
                eta = np.asarray(
                    [row[6] for row in self.wrfvcoords], dtype=float
                ).reshape((nlvl, 1, 1))
                plevel_3d = (
                    eta * (p0[None, :, :] - grid_dummy)
                    + ((heights.reshape((nlvl, 1, 1)) - eta) * (1000.0 - grid_dummy))
                    + grid_dummy
                )

        # compute z_agl
        if z_flag == 3:
            z_agl = np.broadcast_to(
                heights.reshape((nlvl, 1, 1)), (nlvl, ny, nx)
            ).astype(float)
        else:
            if plevel_3d is None:
                raise RuntimeError("plevel_3d missing for non-terrain z_flag")
            z_agl = self._p_to_z(plevel_3d)
            z_agl = np.where(z_agl < 0.0, 0.0, z_agl)

        # z_msl if terrain provided (use surface.terrain or default)
        terrain = self.surface.terrain
        if terrain is None:
            try:
                terrain = self.surface.default.data
            except Exception:
                terrain = None

        if terrain is not None:
            terrain = np.asarray(terrain, dtype=float)
            if terrain.shape != (ny, nx):
                raise ValueError("terrain shape must match surface_pressure shape")
            z_msl = z_agl + terrain[None, :, :]
        else:
            z_msl = None

        # assemble coords dict
        coords: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
        coords["level"] = (("level",), np.arange(1, nlvl + 1))
        if plevel_1d is not None:
            coords["plev_1d"] = (("level",), plevel_1d)
            # also provide a compatibility 1D pressure variable named 'pressure'
            coords["pressure"] = (("level",), plevel_1d)
        if plevel_3d is not None:
            coords["plev"] = (("level", "y", "x"), plevel_3d)
            coords["pressure_3d"] = (("level", "y", "x"), plevel_3d)

        coords["z_agl"] = (("level", "y", "x"), z_agl)
        coords["height"] = (("level", "y", "x"), z_agl)  # compatibility

        if z_msl is not None:
            coords["z_msl"] = (("level", "y", "x"), z_msl)

        if z_flag == 1:
            coords["sigma"] = (("level",), heights.copy())

        coords["_hysplit_meta"] = (
            ("meta",),
            np.array([z_flag, grid_dummy, sfcp], dtype=float),
        )

        return coords


class Grid3D(Grid):
    def __init__(
        self,
        projection: Projection,
        nx: int,
        ny: int,
        vertical_flag: int,
        levels: Sequence[float],
        surface: Surface,
    ):
        super().__init__(projection, nx, ny)
        self.vertical_axis = VerticalAxis(
            flag=vertical_flag,
            heights=levels,
            surface=surface,
            sigma_offset=projection.reserved,
        )

    def _calculate_coords(self) -> dict[str, tuple[tuple[str, ...], np.ndarray]]:
        # Calculate horizontal coords from Grid
        coords = super()._calculate_coords()

        # GOAL: height anove ground level (height_agl), height above mean sea level (height_msl) (to interpolate from)

        # Possible scenarios:
        # if SHGT is not available,
        #  - load terrain from ASC file (DefaultTerrain)
        #    - need to interpolate terrain to model grid
        # if PRSS is not available,
        #  - use default surface pressure (1013.0 hPa)

        # if z_flag == 3 (terrain):
        #  - heights are AGL (1D)
        #    - no way to go from agl to p so nothing to do
        # if z_flag == 2 (pressure):
        #  - heights are in pressure (1D)
        #  - can convert to AGL using surface pressure and HYSPLIT lookup table (3D)
        #    - this is approximate and there are definitely better ways (but not worrying now)
        # if z_flag in (1,4,5) (sigma, hybrid, wrfhybrid):
        #  - heights are in fraction or hybrid (1D)
        #  - can convert to pressure using surface pressure (3D)
        #  - can convert to AGL using surface pressure and HYSPLIT lookup table (3D)

        # Add vertical coords from VerticalAxis
        # Possible coords:
        # - 'level' : raw values (1D)
        # - 'height_agl' : 3D heights above ground level (1D for z_flag==3)
        # - 'plev_1d' : 1D pressure levels when i have this? th met should always have pressure fields right?
        #   well so if its z_flag==2, there is geopotential fields but no pressure fields

        coords.update(self.vertical_axis.coords)

        # Calculate MSL heights if terrain is available
        # - 'height_msl' : 3D heights above mean sea level (if terrain available)

        return coords
