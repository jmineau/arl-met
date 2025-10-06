"""
Grid and projection definitions for ARL meteorological data.

This module provides classes for representing ARL grid projections and
coordinate systems, including horizontal grids, vertical axes, and 3D grids.
Supports various map projections (lat-lon, polar stereographic, Lambert conformal, Mercator)
used in ARL meteorological files.
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Tuple

import numpy as np
import pyproj


def wrap_lons(lons: np.ndarray) -> np.ndarray:
    """
    Wrap longitude values to -180 to 180 degree range.

    Parameters
    ----------
    lons : np.ndarray
        Longitude values in degrees

    Returns
    -------
    np.ndarray
        Longitude values wrapped to [-180, 180] range
    """
    return ((lons + 180) % 360) - 180


@dataclass
class Projection:
    """
    ARL Grid Projection

    Parameters
    ----------
    pole_lat : float
        Pole latitude position of the grid projection. Most projections will be defined
        at +90 or -90 depending upon the hemisphere. For lat-lon grids: latitude of the
        grid point with the maximum grid point value.
    pole_lon : float
        Pole longitude position of the grid projection. The longitude 180 degrees from
        which the projection is cut. For lat-lon grids: longitude of the grid point with
        the maximum grid point value.
    tangent_lat : float
        Reference latitude at which the grid spacing is defined. For lat-lon grids:
        grid spacing in degrees latitude.
    tangent_lon : float
        Reference longitude at which the grid spacing is defined. For lat-lon grids:
        grid spacing in degrees longitude.
    grid_size : float
        Grid spacing in km at the reference position. For lat-lon grids: value of zero
        signals that the grid is a lat-lon grid.
    orientation : float
        Grid orientation or the angle at the reference point made by the y-axis and the
        local direction of north. For lat-lon grids: value always = 0.
    cone_angle : float
        Angle between the axis and the surface of the cone. For regular projections it
        equals the latitude at which the grid is tangent to the earth's surface. Polar
        stereographic: ±90, Mercator: 0, Lambert Conformal: between limits, Oblique
        stereographic: 90. For lat-lon grids: value always = 0.
    sync_x : float
        Grid x-coordinate used to equate a position on the grid with a position on earth
        (paired with sync_y, sync_lat, sync_lon).
    sync_y : float
        Grid y-coordinate used to equate a position on the grid with a position on earth
        (paired with sync_x, sync_lat, sync_lon).
    sync_lat : float
        Earth latitude corresponding to the grid position (sync_x, sync_y). For lat-lon
        grids: latitude of the (0,0) grid point position.
    sync_lon : float
        Earth longitude corresponding to the grid position (sync_x, sync_y). For lat-lon
        grids: longitude of the (0,0) grid point position.
    reserved : float
        Reserved for future use.

    Attributes
    ----------
    crs : pyproj.CRS
        The pyproj CRS object representing the base grid projection.
        The projection is defined without false easting/northing offsets.
    is_latlon : bool
        True if the grid is a lat-lon grid (grid_size == 0).
    """

    pole_lat: float
    pole_lon: float
    tangent_lat: float
    tangent_lon: float
    grid_size: float
    orientation: float
    cone_angle: float
    sync_x: float
    sync_y: float
    sync_lat: float
    sync_lon: float
    reserved: float

    params: Dict[str, Any] = field(init=False, repr=False)

    PARAMS: ClassVar[Dict[str, Any]] = {
        "ellps": "WGS84",
        "R": 6371.2 * 1e3,  # Use a fixed radius to match HYSPLIT
        "units": "m",
    }

    def __post_init__(self):
        """Initialize projection parameters after dataclass initialization."""
        if self.orientation != 0.0:
            raise NotImplementedError(
                "Rotated grids with non-zero orientation are not supported."
            )

        self.params = self._get_params()

    @property
    def is_latlon(self) -> bool:
        """
        Check if this is a lat-lon grid.

        Returns
        -------
        bool
            True if grid_size is 0 (indicating a lat-lon grid), False otherwise.
        """
        return self.grid_size == 0.0

    def _get_params(self) -> Dict[str, Any]:
        """
        Get pyproj projection parameters based on grid configuration.

        Returns
        -------
        Dict[str, Any]
            Dictionary of pyproj parameters for the projection.
        """
        params = self.PARAMS.copy()

        if self.is_latlon:  # Lat/Lon grid
            params.pop("units")
            params.update(
                {
                    "proj": "latlong",
                }
            )
        elif abs(self.cone_angle) == 90.0:  # Stereographic
            if abs(self.pole_lat) == 90.0:  # Polar Stereographic
                params.update(
                    {
                        "proj": "stere",
                        "lat_0": self.pole_lat,
                        "lon_0": self.tangent_lon,
                        "lat_ts": self.tangent_lat,
                    }
                )
            else:  # Oblique Stereographic
                params.update(
                    {
                        "proj": "sterea",
                        "lat_0": self.pole_lat,
                        "lon_0": self.tangent_lon,
                        "lat_ts": self.tangent_lat,
                    }
                )
        elif self.cone_angle == 0.0:  # Mercator
            params.update(
                {
                    "proj": "merc",
                    "lat_ts": self.tangent_lat,
                    "lon_0": self.tangent_lon,
                }
            )
        else:  # Lambert Conformal Conic
            params.update(
                {
                    "proj": "lcc",
                    "lat_0": self.tangent_lat,
                    "lon_0": self.tangent_lon,
                    "lat_1": self.cone_angle,
                }
            )

        return params


@dataclass
class Grid:
    """
    Represents the horizontal grid of the ARL data.

    Parameters
    ----------
    proj : Projection
        The grid projection information.
    nx : int
        Number of grid points in the x-direction (columns).
    ny : int
        Number of grid points in the y-direction (rows).

    Attributes
    ----------
    origin : Tuple[float, float]
        Origin (lower-left corner) in the base CRS (projected coordinates).
    crs : pyproj.CRS
        Coordinate reference system for the grid.
    coords : Dict[str, Any]
        Coordinates of the grid points in the base CRS (projected coordinates).
    """

    proj: Projection
    nx: int
    ny: int

    origin: Tuple[float, float] = field(init=False)
    crs: pyproj.CRS = field(init=False)
    coords: Dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self):
        """Initialize grid coordinates and CRS after dataclass initialization."""
        self.origin = self._calculate_origin()
        self.crs = self._calculate_crs()
        self.coords = self._calculate_coords()

    @property
    def is_latlon(self) -> bool:
        """
        Check if this grid uses a lat-lon projection.

        Returns
        -------
        bool
            True if the projection is lat-lon, False otherwise.
        """
        return self.proj.is_latlon

    @property
    def dims(self) -> tuple:
        """
        Get the dimension names for this grid.

        Returns
        -------
        tuple
            ("lat", "lon") for lat-lon grids, ("y", "x") for projected grids.
        """
        if self.is_latlon:
            return ("lat", "lon")
        return ("y", "x")

    def _calculate_origin(self) -> Tuple[float, float]:
        """
        Calculate the origin (lower-left corner) in the base CRS.

        Returns
        -------
        Tuple[float, float]
            Origin coordinates (x, y) or (lon, lat) for lat-lon grids.
        """
        proj = self.proj

        if self.is_latlon:
            # For lat-lon grids, the origin is simply the sync point
            return proj.sync_lon, proj.sync_lat

        # Calculate what the projected coordinates of the sync point should be
        base_crs = pyproj.CRS.from_dict(proj.params)
        transformer = pyproj.Transformer.from_proj(
            "EPSG:4326", base_crs, always_xy=True
        )
        sync_proj_x, sync_proj_y = transformer.transform(proj.sync_lon, proj.sync_lat)

        # Convert sync grid coordinates to projected coordinates
        # Grid coordinates are 1-based, so sync_x=1, sync_y=1 means bottom-left corner
        sync_grid_x_m = (proj.sync_x - 1) * proj.grid_size * 1000  # convert km to m
        sync_grid_y_m = (proj.sync_y - 1) * proj.grid_size * 1000  # convert km to m

        # Calculate the origin offset to align grid coordinates with projected coordinates
        origin_x = sync_grid_x_m - sync_proj_x
        origin_y = sync_grid_y_m - sync_proj_y
        return origin_x, origin_y

    def _calculate_crs(self) -> pyproj.CRS:
        """
        Calculate the coordinate reference system for this grid.

        Returns
        -------
        pyproj.CRS
            Coordinate reference system with false easting/northing applied.
        """
        params = self.proj.params.copy()

        if self.is_latlon:
            # Use specific ellps and
            return pyproj.CRS.from_dict(params)

        # Create new pyproj CRS with false easting/northing
        params.update({"x_0": self.origin[0], "y_0": self.origin[1]})
        return pyproj.CRS.from_dict(params)

    def _calculate_coords(self):
        """
        Calculate grid coordinates in both projected and geographic systems.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing coordinate arrays:
            - For lat-lon grids: "lon" and "lat" 1D arrays
            - For projected grids: "x", "y" 1D arrays and "lon", "lat" 2D arrays
        """
        proj = self.proj

        if self.is_latlon:
            lon_0, lat_0 = self.origin
            dlat = proj.tangent_lat
            dlon = proj.tangent_lon
            lats = lat_0 + np.arange(self.ny) * dlat
            lons = lon_0 + np.arange(self.nx) * dlon
            lons = wrap_lons(lons)
            return {"lon": lons, "lat": lats}

        # Calculate the coordinates in the projection space
        grid_size = proj.grid_size * 1000  # km to m
        x_coords = np.arange(self.nx) * grid_size
        y_coords = np.arange(self.ny) * grid_size

        # Create a transformer from the projection to lat/lon
        transformer = pyproj.Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)

        # Transform the coordinates to lat/lon
        xx, yy = np.meshgrid(x_coords, y_coords)
        lons, lats = transformer.transform(xx, yy)
        lons = wrap_lons(lons)

        return {
            "x": x_coords,
            "y": y_coords,
            "lon": (("y", "x"), lons),
            "lat": (("y", "x"), lats),
        }


class VerticalAxis:
    """
    Represents the vertical axis of the ARL data.

    Parameters
    ----------
    vertical_flag : int
        Vertical coordinate system type (1=sigma, 2=pressure, 3=terrain, 4=hybrid).
    levels : List[float]
        List of vertical levels corresponding to the vertical coordinate system.
    """

    FLAGS = {
        1: "sigma",  # (fraction)
        2: "pressure",  # (mb)
        3: "terrain",  # (fraction)
        4: "hybrid",  # (mb: offset.fraction)
    }

    def __init__(self, vertical_flag: int, levels: List[float]):
        """
        Initialize the vertical axis.

        Parameters
        ----------
        vertical_flag : int
            Vertical coordinate system type (1=sigma, 2=pressure, 3=terrain, 4=hybrid).
        levels : List[float]
            List of vertical levels corresponding to the vertical coordinate system.
        """
        self.flag = vertical_flag
        self.levels = levels

    @property
    def coord_type(self) -> str:
        """
        Get the vertical coordinate system type name.

        Returns
        -------
        str
            Name of the vertical coordinate type (e.g., "sigma", "pressure", "terrain", "hybrid").
        """
        return VerticalAxis.FLAGS.get(self.flag, "unknown")

    def calculate_heights(self) -> np.ndarray | None:
        """
        Calculate heights in meters for each vertical level.
        """
        if self.coord_type == "sigma":
            # fraction
            pass
        elif self.coord_type == "pressure":
            # mb
            pass
        elif self.coord_type == "terrain":
            # fraction
            pass
        elif self.coord_type == "hybrid":
            # mb offset.fraction
            pass
        else:
            raise ValueError(f"Unknown vertical coordinate type")

        return None


class Grid3D(Grid):
    """
    Represents a 3D grid with horizontal and vertical dimensions.

    Parameters
    ----------
    proj: Projection
        The grid projection information.
    nx : int
        Number of grid points in the x-direction (columns).
    ny : int
        Number of grid points in the y-direction (rows).
    vertical_axis : VerticalAxis
        Vertical axis information including coordinate type and levels.
    """

    def __init__(self, proj: Projection, nx: int, ny: int, vertical_axis: VerticalAxis):
        """
        Initialize a 3D grid.

        Parameters
        ----------
        proj : Projection
            The grid projection information.
        nx : int
            Number of grid points in the x-direction (columns).
        ny : int
            Number of grid points in the y-direction (rows).
        vertical_axis : VerticalAxis
            Vertical axis information including coordinate type and levels.
        """
        super().__init__(proj=proj, nx=nx, ny=ny)
        self.vertical_axis = vertical_axis

        # self.coords['level'] = self.vertical_axis.levels

    @property
    def dims(self) -> tuple:
        """
        Get the dimension names for this 3D grid.

        Returns
        -------
        tuple
            Dimension names for the horizontal grid (vertical dimension not yet implemented).
        """
        xy_dims = super().dims
        return xy_dims  # TODO: add vertical dimension
