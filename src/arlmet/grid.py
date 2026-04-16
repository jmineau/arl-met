"""
Grid and projection definitions for ARL meteorological data.

This module provides classes for representing ARL grid projections and
horizontal coordinate systems used in ARL meteorological files. Vertical
coordinates live in ``arlmet.vertical``.
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar

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
        )


@dataclass(frozen=True)
class GridWindow:
    """
    Rectangular subset of a grid using zero-based half-open indices.

    Parameters
    ----------
    x_start, x_stop : int
        Inclusive start and exclusive stop indices in the x direction.
    y_start, y_stop : int
        Inclusive start and exclusive stop indices in the y direction.
    """

    x_start: int
    x_stop: int
    y_start: int
    y_stop: int

    def __post_init__(self) -> None:
        if any(value < 0 for value in vars(self).values()):
            raise ValueError("GridWindow indices must be non-negative.")
        if self.x_stop <= self.x_start:
            raise ValueError("GridWindow x_stop must be greater than x_start.")
        if self.y_stop <= self.y_start:
            raise ValueError("GridWindow y_stop must be greater than y_start.")

    @property
    def nx(self) -> int:
        return self.x_stop - self.x_start

    @property
    def ny(self) -> int:
        return self.y_stop - self.y_start

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    @property
    def x_slice(self) -> slice:
        return slice(self.x_start, self.x_stop)

    @property
    def y_slice(self) -> slice:
        return slice(self.y_start, self.y_stop)


class Grid:
    """
    Represents the 2D horizontal grid of the ARL data.

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
    crs : pyproj.CRS
        Coordinate reference system for the grid.
    dims : tuple
        Dimension names for the grid ("lat", "lon") or ("y", "x").
    is_latlon : bool
        True if the grid uses a lat-lon projection.
    origin : tuple[float, float]
        Origin (lower-left corner) in the base CRS (projected coordinates).

    Methods
    -------
    calculate_coords() -> dict[str, Any]
        Calculate grid coordinates in both projected and geographic systems.
    """

    def __init__(self, projection: Projection, nx: int, ny: int):
        self.projection = projection
        self.nx = nx
        self.ny = ny

        self._origin = None
        self._crs = None

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
    def coords(self) -> dict[str, Any]:
        """
        Return the calculated coordinate variables for this grid.
        """
        return self.calculate_coords()

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

    def calculate_coords(self) -> dict[str, Any]:
        """
        Grid coordinates in both projected and geographic systems.

        Returns
        -------
        dict[str, Any]
            Dictionary containing coordinate arrays:
            - For lat-lon grids: "lon" and "lat" 1D arrays
            - For projected grids: "x", "y" 1D arrays and "lon", "lat" 2D arrays
        """
        proj = self.projection

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

        coords = {
            "x": x_coords,
            "y": y_coords,
            "lon": (("y", "x"), lons),
            "lat": (("y", "x"), lats),
        }

        return coords

    def fractional_indices(
        self, lon: np.ndarray | float, lat: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert geographic coordinates to zero-based fractional grid indices.

        Parameters
        ----------
        lon, lat : array-like or float
            Geographic coordinates in degrees.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Fractional ``(x, y)`` grid indices with the same broadcast shape as the
            input coordinates.
        """
        lon_arr, lat_arr = np.broadcast_arrays(
            np.asarray(lon, dtype=float),
            np.asarray(lat, dtype=float),
        )

        if self.is_latlon:
            lon_0, lat_0 = self.origin
            dlon = self.projection.tangent_lon
            dlat = self.projection.tangent_lat
            if dlon == 0.0 or dlat == 0.0:
                raise ValueError(
                    "Lat/lon grids require non-zero tangent_lon and tangent_lat spacing."
                )

            lon_offset = ((lon_arr - lon_0 + 180.0) % 360.0) - 180.0
            x = lon_offset / dlon
            y = (lat_arr - lat_0) / dlat
            return x.astype(float, copy=False), y.astype(float, copy=False)

        transformer = pyproj.Transformer.from_crs(
            "EPSG:4326",
            self.crs,
            always_xy=True,
        )
        proj_x, proj_y = transformer.transform(lon_arr, lat_arr)
        step = self.projection.grid_size * 1000.0
        if step == 0.0:
            raise ValueError("Projected grids require a non-zero grid_size.")
        x = np.asarray(proj_x, dtype=float) / step
        y = np.asarray(proj_y, dtype=float) / step
        return x, y

    def full_window(self) -> GridWindow:
        """
        Full-grid window spanning the entire domain.
        """
        return GridWindow(x_start=0, x_stop=self.nx, y_start=0, y_stop=self.ny)

    def window_from_bbox(
        self, bbox: tuple[float, float, float, float]
    ) -> GridWindow:
        """
        Resolve a geographic bounding box to grid indices.

        Parameters
        ----------
        bbox : tuple[float, float, float, float]
            Bounding box as ``(west, south, east, north)`` in degrees.
            If ``west > east``, the box is assumed to cross the dateline.
        """
        west, south, east, north = bbox
        if south > north:
            raise ValueError("bbox south must be less than or equal to north.")

        coords = self.calculate_coords()
        lon_coord = coords["lon"]
        lat_coord = coords["lat"]

        if isinstance(lon_coord, tuple):
            lons = np.asarray(lon_coord[1], dtype=float)
        else:
            lons = np.asarray(lon_coord, dtype=float)

        if isinstance(lat_coord, tuple):
            lats = np.asarray(lat_coord[1], dtype=float)
        else:
            lats = np.asarray(lat_coord, dtype=float)

        if west <= east:
            lon_mask = (lons >= west) & (lons <= east)
        else:
            lon_mask = (lons >= west) | (lons <= east)
        lat_mask = (lats >= south) & (lats <= north)

        if self.is_latlon:
            x_idx = np.flatnonzero(lon_mask)
            y_idx = np.flatnonzero(lat_mask)
        else:
            mask = lon_mask & lat_mask
            if not np.any(mask):
                raise ValueError("bbox does not intersect the grid.")
            y_idx, x_idx = np.nonzero(mask)

        if x_idx.size == 0 or y_idx.size == 0:
            raise ValueError("bbox does not intersect the grid.")

        return GridWindow(
            x_start=int(x_idx.min()),
            x_stop=int(x_idx.max()) + 1,
            y_start=int(y_idx.min()),
            y_stop=int(y_idx.max()) + 1,
        )

    def subset(self, window: GridWindow) -> "Grid":
        """
        Build a new grid definition for a rectangular subset.
        """
        if window.x_stop > self.nx or window.y_stop > self.ny:
            raise ValueError("GridWindow extends beyond the grid bounds.")

        coords = self.calculate_coords()
        if self.is_latlon:
            sync_lon = float(np.asarray(coords["lon"])[window.x_start])
            sync_lat = float(np.asarray(coords["lat"])[window.y_start])
        else:
            sync_lon = float(np.asarray(coords["lon"][1])[window.y_start, window.x_start])
            sync_lat = float(np.asarray(coords["lat"][1])[window.y_start, window.x_start])

        projection = self.projection
        subset_projection = Projection(
            pole_lat=projection.pole_lat,
            pole_lon=projection.pole_lon,
            tangent_lat=projection.tangent_lat,
            tangent_lon=projection.tangent_lon,
            grid_size=projection.grid_size,
            orientation=projection.orientation,
            cone_angle=projection.cone_angle,
            sync_x=1.0,
            sync_y=1.0,
            sync_lat=sync_lat,
            sync_lon=sync_lon,
        )
        return Grid(projection=subset_projection, nx=window.nx, ny=window.ny)

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
