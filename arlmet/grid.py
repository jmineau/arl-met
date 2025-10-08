"""
Grid and projection definitions for ARL meteorological data.

This module provides classes for representing ARL grid projections and
coordinate systems, including horizontal grids, vertical axes, and 3D grids.
Supports various map projections (lat-lon, polar stereographic, Lambert conformal, Mercator)
used in ARL meteorological files.
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar, Sequence

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

    params: dict[str, Any] = field(init=False, repr=False)

    PARAMS: ClassVar[dict[str, Any]] = {
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

    def _get_params(self) -> dict[str, Any]:
        """
        Get pyproj projection parameters based on grid configuration.

        Returns
        -------
        dict[str, Any]
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

    def __hash__(self) -> int:
        return hash(
            (
                self.pole_lat,
                self.pole_lon,
                self.tangent_lat,
                self.tangent_lon,
                self.grid_size,
                self.orientation,
                self.cone_angle,
                self.sync_x,
                self.sync_y,
                self.sync_lat,
                self.sync_lon,
                self.reserved,
            )
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, Projection):
            return False
        return (
            self.pole_lat == other.pole_lat
            and self.pole_lon == other.pole_lon
            and self.tangent_lat == other.tangent_lat
            and self.tangent_lon == other.tangent_lon
            and self.grid_size == other.grid_size
            and self.orientation == other.orientation
            and self.cone_angle == other.cone_angle
            and self.sync_x == other.sync_x
            and self.sync_y == other.sync_y
            and self.sync_lat == other.sync_lat
            and self.sync_lon == other.sync_lon
            and self.reserved == other.reserved
        )


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
    origin : tuple[float, float]
        Origin (lower-left corner) in the base CRS (projected coordinates).
    crs : pyproj.CRS
        Coordinate reference system for the grid.
    coords : dict[str, Any]
        Coordinates of the grid points in the base CRS (projected coordinates).
    """

    def __init__(self, projection: Projection, nx: int, ny: int):
        self.projection = projection
        self.nx = nx
        self.ny = ny

        self._origin = None
        self._crs = None
        self._coords = None

    @property
    def is_latlon(self) -> bool:
        """
        Check if this grid uses a lat-lon projection.

        Returns
        -------
        bool
            True if the projection is lat-lon, False otherwise.
        """
        return self.projection.is_latlon

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

    @property
    def origin(self) -> tuple[float, float]:
        """
        Origin (lower-left corner) in the base CRS.

        Returns
        -------
        tuple[float, float]
            Origin coordinates (x, y) or (lon, lat) for lat-lon grids.
        """
        if self._origin is None:
            proj = self.projection

            if self.is_latlon:
                # For lat-lon grids, the origin is simply the sync point
                return proj.sync_lon, proj.sync_lat

            # Calculate what the projected coordinates of the sync point should be
            base_crs = pyproj.CRS.from_dict(proj.params)
            transformer = pyproj.Transformer.from_proj(
                proj_from="EPSG:4326", proj_to=base_crs, always_xy=True
            )
            sync_proj_x, sync_proj_y = transformer.transform(
                proj.sync_lon, proj.sync_lat
            )

            # Convert sync grid coordinates to projected coordinates
            # Grid coordinates are 1-based, so sync_x=1, sync_y=1 means bottom-left corner
            sync_grid_x_m = (proj.sync_x - 1) * proj.grid_size * 1000  # convert km to m
            sync_grid_y_m = (proj.sync_y - 1) * proj.grid_size * 1000  # convert km to m

            # Calculate the origin offset to align grid coordinates with projected coordinates
            origin_x = sync_grid_x_m - sync_proj_x
            origin_y = sync_grid_y_m - sync_proj_y
            self._origin = (origin_x, origin_y)

        return self._origin

    @property
    def crs(self) -> pyproj.CRS:
        """
        Coordinate reference system for this grid.

        Returns
        -------
        pyproj.CRS
            Coordinate reference system with false easting/northing applied.
        """
        if self._crs is None:
            params = self.projection.params.copy()

            if self.is_latlon:
                # Use specific ellps and
                self._crs = pyproj.CRS.from_dict(params)
            else:
                # Create new pyproj CRS with false easting/northing
                params.update({"x_0": self.origin[0], "y_0": self.origin[1]})
                self._crs = pyproj.CRS.from_dict(params)

        return self._crs

    @property
    def coords(self) -> dict[str, Any]:
        """
        Grid coordinates in both projected and geographic systems.

        Returns
        -------
        dict[str, Any]
            Dictionary containing coordinate arrays:
            - For lat-lon grids: "lon" and "lat" 1D arrays
            - For projected grids: "x", "y" 1D arrays and "lon", "lat" 2D arrays
        """
        if self._coords is None:
            proj = self.projection

            if self.is_latlon:
                lon_0, lat_0 = self.origin
                dlat = proj.tangent_lat
                dlon = proj.tangent_lon
                lats = lat_0 + np.arange(self.ny) * dlat
                lons = lon_0 + np.arange(self.nx) * dlon
                lons = wrap_lons(lons)
                return {
                    "lon": lons,
                    "lat": lats,
                }

            # Calculate the coordinates in the projection space
            grid_size = proj.grid_size * 1000  # km to m
            x_coords = np.arange(self.nx) * grid_size
            y_coords = np.arange(self.ny) * grid_size

            # Create a transformer from the projection to lat/lon
            transformer = pyproj.Transformer.from_crs(
                self.crs, "EPSG:4326", always_xy=True
            )

            # Transform the coordinates to lat/lon
            xx, yy = np.meshgrid(x_coords, y_coords)
            lons, lats = transformer.transform(xx, yy)
            lons = wrap_lons(lons)

            self._coords = {
                "x": x_coords,
                "y": y_coords,
                "lon": (("y", "x"), lons),
                "lat": (("y", "x"), lats),
            }

        return self._coords

    def get_coord_attrs(self) -> dict[str, dict[str, Any]]:
        """
        Get CF-compliant attributes for coordinates.

        Returns
        -------
        dict[str, dict[str, Any]]
            Dictionary mapping coordinate names to their attributes.
        """
        attrs = {}

        if self.is_latlon:
            attrs["lon"] = {
                "units": "degrees_east",
                "long_name": "longitude",
                "standard_name": "longitude",
            }
            attrs["lat"] = {
                "units": "degrees_north",
                "long_name": "latitude",
                "standard_name": "latitude",
            }
        else:
            attrs["x"] = {
                "units": "m",
                "long_name": "x coordinate of projection",
                "standard_name": "projection_x_coordinate",
            }
            attrs["y"] = {
                "units": "m",
                "long_name": "y coordinate of projection",
                "standard_name": "projection_y_coordinate",
            }
            attrs["lon"] = {
                "units": "degrees_east",
                "long_name": "longitude",
                "standard_name": "longitude",
            }
            attrs["lat"] = {
                "units": "degrees_north",
                "long_name": "latitude",
                "standard_name": "latitude",
            }

        return attrs

    def __repr__(self) -> str:
        return f"Grid(projection={self.projection}, nx={self.nx}, ny={self.ny})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Grid):
            return False
        return (
            self.projection == other.projection
            and self.nx == other.nx
            and self.ny == other.ny
        )

    def __hash__(self) -> int:
        return hash((self.projection, self.nx, self.ny))


@dataclass
class VerticalAxis:

    flag: int
    levels: Sequence[float]

    FLAGS: ClassVar[dict[int, str]] = {
        1: "sigma",  # fraction
        2: "pressure",  # mb/hPa
        3: "terrain",  # fraction/height
        4: "hybrid",  # mb: offset.fraction
        5: "wrf",  # WRF
    }

    @property
    def coord_system(self) -> str:
        return self.FLAGS.get(self.flag, "unknown")

    def get_coord_attrs(self) -> dict[str, Any]:
        """
        Get CF-compliant attributes for the vertical coordinate.

        Returns
        -------
        dict[str, Any]
            Dictionary of attributes for the level coordinate.
        """
        coord_system = self.coord_system

        if coord_system == "sigma":
            return {
                "units": "1",
                "long_name": "sigma levels",
                "standard_name": "atmosphere_sigma_coordinate",
                "positive": "down",
            }
        elif coord_system == "pressure":
            return {
                "units": "hPa",
                "long_name": "pressure",
                "standard_name": "air_pressure",
                "positive": "down",
            }
        elif coord_system == "terrain":
            return {
                "units": "m",
                "long_name": "height above ground",
                "standard_name": "height",
                "positive": "up",
            }
        elif coord_system == "hybrid":
            return {
                "units": "1",
                "long_name": "hybrid levels",
                "positive": "down",
            }
        else:
            return {
                "long_name": "vertical level",
            }

    def __eq__(self, other) -> bool:
        if not isinstance(other, VerticalAxis):
            return False
        return self.flag == other.flag and np.array_equal(self.levels, other.levels)

    def __hash__(self) -> int:
        return hash((self.flag, tuple(self.levels)))


class Grid3D(Grid):
    def __init__(
        self, projection: Projection, nx: int, ny: int, vertical_axis: VerticalAxis
    ):
        super().__init__(projection, nx, ny)
        self.vertical_axis = vertical_axis

        self._coords = None  # reset coords cache
        self._levels = None

    @property
    def dims(self) -> tuple:
        """
        Get the dimension names for this 3D grid.

        Returns
        -------
        tuple
            ("level", "lat", "lon") for lat-lon grids, ("level", "y", "x") for projected grids.
        """
        return ("level",) + super().dims

    @property
    def coords(self) -> dict[str, Any]:
        """
        3D Grid coordinates including vertical levels.

        Returns
        -------
        dict[str, Any]
            Dictionary containing coordinate arrays:
            - "level": 1D array of vertical levels
            - For lat-lon grids: "lon" and "lat" 1D arrays
            - For projected grids: "x", "y" 1D arrays and "lon", "lat" 2D arrays
        """
        if self._coords is None:
            coords = super().coords.copy()
            coords["level"] = self.levels
            self._coords = coords
        return self._coords

    @property
    def levels(self) -> np.ndarray:
        """
        Get the vertical levels as a numpy array.

        Returns
        -------
        np.ndarray
            Array of vertical levels.
        """
        return np.array(self.vertical_axis.levels)

    def get_coord_attrs(self) -> dict[str, dict[str, Any]]:
        """
        Get CF-compliant attributes for all coordinates.

        Returns
        -------
        dict[str, dict[str, Any]]
            Dictionary mapping coordinate names to their attributes.
        """
        # Get horizontal coordinate attrs from parent
        attrs = super().get_coord_attrs()

        # Add vertical coordinate attrs
        attrs["level"] = self.vertical_axis.get_coord_attrs()

        # Add time attrs
        attrs["time"] = {
            "long_name": "time",
            "standard_name": "time",
        }

        # Add forecast attrs
        attrs["forecast"] = {
            "units": "hours",
            "long_name": "forecast hour",
        }

        return attrs

    def __repr__(self) -> str:
        return (
            f"Grid3D(projection={self.projection}, nx={self.nx}, ny={self.ny}, "
            f"vertical_axis={self.vertical_axis})"
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, Grid3D):
            return False
        # Check parent equality and vertical axis equality
        return super().__eq__(other) and self.vertical_axis == other.vertical_axis

    def __hash__(self) -> int:
        # Combine the parent's hash with the vertical axis's hash
        return hash((super().__hash__(), self.vertical_axis))
