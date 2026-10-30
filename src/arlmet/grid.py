"""
Grid and projection definitions for ARL meteorological data.

This module provides classes for representing ARL grid projections and
horizontal coordinate systems used in ARL meteorological files. Vertical
coordinates live in ``arlmet.vertical``.
"""

import math
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
import pyproj
from typing_extensions import override

__all__ = ["Projection", "Grid", "GridWindow"]

# One coordinate variable: ``(dims, values)``, as accepted by ``xr.Dataset``.
_Coord = tuple[tuple[str, ...], npt.NDArray[Any]]


def wrap_lons(lons: npt.NDArray[Any]) -> npt.NDArray[Any]:
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


def _lambert_scale_factor(cone_lat: float, ref_lat: float) -> float:
    """
    Scale factor that makes a tangent Lambert cone true to scale at ``ref_lat``.

    A spherical Lambert conformal cone tangent at ``cone_lat`` has map scale
    ``k = cos(cone_lat) / cos(ref_lat) * (t(ref_lat) / t(cone_lat)) ** n`` at
    ``ref_lat``, with ``t(lat) = tan(pi/4 - lat/2)`` and ``n = sin(cone_lat)``.
    Returns ``1 / k`` (exactly 1 when the two latitudes are equal), the
    ``k_0`` that scales the cone as HYSPLIT's ``stcm1p`` does.
    """
    if ref_lat == cone_lat:
        return 1.0
    phi_c = math.radians(cone_lat)
    phi_r = math.radians(ref_lat)
    n = math.sin(phi_c)
    t_c = math.tan(math.pi / 4 - phi_c / 2)
    t_r = math.tan(math.pi / 4 - phi_r / 2)
    k = math.cos(phi_c) / math.cos(phi_r) * (t_r / t_c) ** n
    return 1.0 / k


@dataclass(frozen=True)
class Projection:
    """
    Horizontal projection metadata from an ARL index record.

    Projections are immutable, hashable value objects; use
    :func:`dataclasses.replace` to derive a modified copy.

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
        Reference latitude at which the grid spacing is defined. It may differ
        from ``cone_angle`` (WRF grids converted by ``arw2arl``). For lat-lon
        grids: grid spacing in degrees latitude.
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
        stereographic: ±90, Mercator: 0, Lambert Conformal: between limits. As in
        HYSPLIT, this value alone picks the projection, so ±90 is always polar
        stereographic. For lat-lon grids: value always = 0.
    sync_x : float
        Grid x-coordinate used to equate a position on the grid with a position on earth
        (paired with sync_y, sync_lat, sync_lon). One-based; it may lie outside the
        grid, as in a subset of a projected grid (:meth:`Grid.subset`).
    sync_y : float
        Grid y-coordinate used to equate a position on the grid with a position on earth
        (paired with sync_x, sync_lat, sync_lon). One-based; it may lie outside the
        grid.
    sync_lat : float
        Earth latitude corresponding to the grid position (sync_x, sync_y). For lat-lon
        grids: latitude of the (0,0) grid point position.
    sync_lon : float
        Earth longitude corresponding to the grid position (sync_x, sync_y). For lat-lon
        grids: longitude of the (0,0) grid point position.

    Attributes
    ----------
    params : dict[str, Any]
        pyproj parameter dictionary derived from the ARL metadata (a new
        dict on each access). False easting and northing offsets are applied
        at the Grid level (:attr:`Grid.crs`).
    is_latlon : bool
        True if the grid is a lat-lon grid (grid_size == 0).

    Examples
    --------
    >>> from arlmet.grid import Projection
    >>> proj = Projection(
    ...     pole_lat=90.0,
    ...     pole_lon=180.0,
    ...     tangent_lat=1.0,
    ...     tangent_lon=1.0,
    ...     grid_size=0.0,
    ...     orientation=0.0,
    ...     cone_angle=0.0,
    ...     sync_x=1.0,
    ...     sync_y=1.0,
    ...     sync_lat=-90.0,
    ...     sync_lon=-180.0,
    ... )
    >>> proj.is_latlon
    True
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

    PARAMS: ClassVar[dict[str, Any]] = {
        "ellps": "WGS84",
        "R": 6371.2 * 1e3,  # Use a fixed radius to match HYSPLIT
        "units": "m",
    }

    def __post_init__(self):
        """Reject projections arlmet cannot represent."""
        if self.orientation != 0.0:
            raise NotImplementedError(
                "Rotated grids with non-zero orientation are not supported."
            )

    @property
    def params(self) -> dict[str, Any]:
        """Parameters of the base projection for pyproj (a new dict on each access)."""
        return self._get_params()

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
        elif abs(self.cone_angle) == 90.0:  # Polar stereographic
            # As HYSPLIT's stlmbr: the cone angle alone picks the projection,
            # so +-90 is stereographic about that pole whatever pole_lat says.
            params.update(
                {
                    "proj": "stere",
                    "lat_0": math.copysign(90.0, self.cone_angle),
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
            # The cone is tangent at cone_angle; grid_size holds at tangent_lat
            # (HYSPLIT's stlmbr then stcm1p, as metini.f sets them up).
            params.update(
                {
                    "proj": "lcc",
                    "lat_0": self.tangent_lat,
                    "lon_0": self.tangent_lon,
                    "lat_1": self.cone_angle,
                    "k_0": _lambert_scale_factor(self.cone_angle, self.tangent_lat),
                }
            )

        return params

    @override
    def __repr__(self) -> str:
        proj = self.params.get("proj", "unknown")
        if proj == "latlong":
            proj = "latlon"
        return f"Projection({proj})"


@dataclass(frozen=True)
class GridWindow:
    """
    Rectangular subset of a grid using zero-based half-open indices.

    Parameters
    ----------
    x_start, x_stop : int
        Inclusive start and exclusive stop indices in the x direction. On a
        lat/lon grid that spans 360 degrees (:attr:`Grid.wraps_lon`),
        ``x_stop`` may pass the grid's ``nx``: the window then runs on from the
        last column to the first, as a crop across the grid's seam does.
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
        """Number of selected x-grid points."""
        return self.x_stop - self.x_start

    @property
    def ny(self) -> int:
        """Number of selected y-grid points."""
        return self.y_stop - self.y_start

    @property
    def shape(self) -> tuple[int, int]:
        """Window shape as ``(ny, nx)``."""
        return (self.ny, self.nx)

    @property
    def x_slice(self) -> slice:
        """Slice object for selecting the x range."""
        return slice(self.x_start, self.x_stop)

    @property
    def y_slice(self) -> slice:
        """Slice object for selecting the y range."""
        return slice(self.y_start, self.y_stop)


def _split_at_seam(window: GridWindow, grid: "Grid") -> tuple[GridWindow, ...]:
    """
    Split a window that runs past the grid's last column into its two parts.

    A window within the grid comes back alone; one across the seam of a grid
    that spans 360 degrees comes back as its eastern part (up to the last
    column) and its western part (from the first), in that order.
    """
    nx = grid.nx
    if window.x_stop <= nx:
        return (window,)
    if not grid.wraps_lon or window.nx > nx:
        raise ValueError("GridWindow extends beyond the grid bounds.")
    return (
        replace(window, x_stop=nx),
        replace(window, x_start=0, x_stop=window.x_stop - nx),
    )


@dataclass(frozen=True)
class Grid:
    """
    Two-dimensional horizontal grid definition for ARL data.

    Grids are immutable, hashable value objects (the derived ``crs`` and
    ``origin`` are computed once and cached). Use :meth:`subset` or
    :func:`dataclasses.replace` to derive a new grid.

    Parameters
    ----------
    projection : Projection
        Grid projection metadata.
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
    calculate_coords() -> dict[str, tuple[tuple[str, ...], np.ndarray]]
        Calculate grid coordinates as ``name -> (dims, values)``.
    fractional_indices(lon, lat)
        Convert lon/lat positions to fractional grid indices.
    window_from_bbox(bbox)
        Resolve a geographic bounding box to an inclusive grid window.
    subset(window)
        Build a new Grid describing a rectangular subset.

    Examples
    --------
    >>> from arlmet.grid import Grid, Projection
    >>> proj = Projection(
    ...     pole_lat=90.0,
    ...     pole_lon=180.0,
    ...     tangent_lat=1.0,
    ...     tangent_lon=1.0,
    ...     grid_size=0.0,
    ...     orientation=0.0,
    ...     cone_angle=0.0,
    ...     sync_x=1.0,
    ...     sync_y=1.0,
    ...     sync_lat=40.0,
    ...     sync_lon=-120.0,
    ... )
    >>> grid = Grid(projection=proj, nx=3, ny=2)
    >>> tuple(grid.dims)
    ('lat', 'lon')
    """

    projection: Projection
    nx: int
    ny: int

    def __post_init__(self) -> None:
        """Validate the grid shape."""
        if self.nx < 1 or self.ny < 1:
            raise ValueError(
                f"Grid dimensions must be positive, got nx={self.nx}, ny={self.ny}."
            )

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
    def dims(self) -> tuple[str, str]:
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

    @cached_property
    def origin(self) -> tuple[float, float]:
        """
        Origin (lower-left corner) in the base CRS.

        Returns
        -------
        tuple[float, float]
            Origin coordinates (x, y) or (lon, lat) for lat-lon grids.
        """
        proj = self.projection

        if self.is_latlon:
            # For lat-lon grids, the origin is simply the sync point
            return proj.sync_lon, proj.sync_lat

        # Calculate what the projected coordinates of the sync point should be
        base_crs = pyproj.CRS.from_dict(proj.params)
        transformer = pyproj.Transformer.from_proj(
            proj_from="EPSG:4326", proj_to=base_crs, always_xy=True
        )
        sync_proj_x, sync_proj_y = transformer.transform(proj.sync_lon, proj.sync_lat)

        # Convert sync grid coordinates to projected coordinates
        # Grid coordinates are 1-based, so sync_x=1, sync_y=1 means bottom-left corner
        sync_grid_x_m = (proj.sync_x - 1) * proj.grid_size * 1000  # convert km to m
        sync_grid_y_m = (proj.sync_y - 1) * proj.grid_size * 1000  # convert km to m

        # Calculate the origin offset to align grid coordinates with projected coordinates
        origin_x = sync_grid_x_m - sync_proj_x
        origin_y = sync_grid_y_m - sync_proj_y
        return (origin_x, origin_y)

    @cached_property
    def crs(self) -> pyproj.CRS:
        """
        Coordinate reference system for this grid.

        Returns
        -------
        pyproj.CRS
            Coordinate reference system with false easting/northing applied.
        """
        params = self.projection.params
        if not self.is_latlon:
            # Apply the grid origin as false easting/northing
            params.update({"x_0": self.origin[0], "y_0": self.origin[1]})
        return pyproj.CRS.from_dict(params)

    def calculate_coords(self) -> dict[str, _Coord]:
        """
        Grid coordinates in both projected and geographic systems.

        Returns
        -------
        dict[str, tuple[tuple[str, ...], numpy.ndarray]]
            Coordinate variables as ``name -> (dims, values)``, ready for
            ``xr.Dataset(coords=...)``. Arrays are newly allocated on each call.

            - lat-lon grids: 1-D ``"lon"`` ``(("lon",), ...)`` and ``"lat"``
              ``(("lat",), ...)``.
            - projected grids: 1-D ``"x"``/``"y"`` in meters and 2-D
              ``"lon"``/``"lat"`` with dims ``("y", "x")``.
        """
        proj = self.projection

        if self.is_latlon:
            lon_0, lat_0 = self.origin
            dlat = proj.tangent_lat
            dlon = proj.tangent_lon
            lats = lat_0 + np.arange(self.ny) * dlat
            # Normalize only the start to [-180, 180]; keep the sequence monotonic
            lon_start = ((lon_0 + 180) % 360) - 180
            lons = lon_start + np.arange(self.nx) * dlon
            return {"lon": (("lon",), lons), "lat": (("lat",), lats)}

        # Calculate the coordinates in the projection space
        grid_size = proj.grid_size * 1000  # km to m
        x_coords = np.arange(self.nx) * grid_size
        y_coords = np.arange(self.ny) * grid_size

        # Create a transformer from the projection to lat/lon
        transformer = pyproj.Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)

        # Transform the coordinates to lat/lon
        xx, yy = np.meshgrid(x_coords, y_coords)
        lons, lats = transformer.transform(xx, yy)
        lons = wrap_lons(np.asarray(lons, dtype=float))

        return {
            "x": (("x",), x_coords),
            "y": (("y",), y_coords),
            "lon": (("y", "x"), lons),
            "lat": (("y", "x"), np.asarray(lats, dtype=float)),
        }

    @property
    def wraps_lon(self) -> bool:
        """
        Whether the grid is lat/lon and its columns span all 360 degrees of longitude.

        On such a global grid the last column is adjacent to the first, so
        points between them interpolate across the seam.
        """
        if not self.is_latlon:
            return False
        return bool(np.isclose(self.nx * abs(self.projection.tangent_lon), 360.0))

    def fractional_indices(
        self, lon: npt.NDArray[Any] | float, lat: npt.NDArray[Any] | float
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
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

            # Longitudes are periodic: measure each point eastward from the
            # grid origin, in [0, 360). (Wrapping to [-180, 180) instead put
            # every point more than 180 degrees east of the origin, e.g. the
            # whole western hemisphere on a global 0-360 grid, off the grid.)
            lon_offset = (lon_arr - lon_0) % 360.0
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

    def meridian_convergence(
        self, lon: npt.NDArray[Any] | float, lat: npt.NDArray[Any] | float
    ) -> npt.NDArray[Any]:
        """
        Angle from grid north to true north at each point, in degrees.

        Positive where true north lies clockwise of the grid y-axis. Zero
        everywhere on a lat/lon grid.

        Parameters
        ----------
        lon, lat : array-like or float
            Geographic coordinates in degrees.

        Returns
        -------
        np.ndarray
            Convergence angle in degrees with the broadcast shape of the inputs.
        """
        lon_arr, lat_arr = np.broadcast_arrays(
            np.asarray(lon, dtype=float),
            np.asarray(lat, dtype=float),
        )
        if self.is_latlon:
            return np.zeros(lon_arr.shape, dtype=float)
        factors = pyproj.Proj(self.crs).get_factors(lon_arr, lat_arr)
        return np.asarray(factors.meridian_convergence, dtype=float).reshape(
            lon_arr.shape
        )

    def rotate_winds(
        self,
        u: npt.NDArray[Any] | float,
        v: npt.NDArray[Any] | float,
        lon: npt.NDArray[Any] | float,
        lat: npt.NDArray[Any] | float,
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """
        Rotate grid-relative wind components to earth-relative ones.

        Winds in ARL files on projected grids are stored relative to the grid
        axes, which is what HYSPLIT expects. This rotates them by the meridian
        convergence at each point so that ``u`` points east and ``v`` north.
        Lat/lon grids are returned unchanged.

        Parameters
        ----------
        u, v : array-like or float
            Grid-relative wind components.
        lon, lat : array-like or float
            Geographic coordinates of each wind, in degrees.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Earth-relative ``(u, v)`` with the broadcast shape of the inputs.
        """
        u_arr, v_arr, lon_arr, lat_arr = np.broadcast_arrays(
            np.asarray(u, dtype=float),
            np.asarray(v, dtype=float),
            np.asarray(lon, dtype=float),
            np.asarray(lat, dtype=float),
        )
        if self.is_latlon:
            return u_arr.copy(), v_arr.copy()
        angle = np.radians(self.meridian_convergence(lon_arr, lat_arr))
        cos = np.cos(angle)
        sin = np.sin(angle)
        return cos * u_arr + sin * v_arr, cos * v_arr - sin * u_arr

    def full_window(self) -> GridWindow:
        """
        Return a GridWindow spanning the full horizontal domain.

        Returns
        -------
        GridWindow
            Window covering all x and y indices in the grid.
        """
        return GridWindow(x_start=0, x_stop=self.nx, y_start=0, y_stop=self.ny)

    def window_from_bbox(self, bbox: tuple[float, float, float, float]) -> GridWindow:
        """
        Resolve a geographic bounding box to grid indices.

        Parameters
        ----------
        bbox : tuple[float, float, float, float]
            Bounding box as ``(west, south, east, north)`` in degrees.
            If ``west > east``, the box is assumed to cross the dateline.

        Notes
        -----
        On a lat/lon grid the window holds the cells whose centers lie inside
        the box, edges included, as HYSPLIT's ``xtrct_grid`` crops.

        On a projected grid the window is the smallest rectangle of cells
        that holds the whole box: the box's outline (its corners and points
        along its edges, at most half a cell apart) is mapped to grid indices,
        and the lowest and highest index on each axis is rounded to the
        nearest cell, like Fortran ``NINT``. Every point of the box then lies
        in a cell of the window. Parallels and meridians curve on these grids,
        so an extreme is often the south-east or north-west corner, or a point
        partway along an edge. ``xtrct_grid`` uses only the south-west and
        north-east corners, so its crop can leave out part of the box; this
        window holds that crop and can be a few cells larger.
        """
        west, south, east, north = bbox
        if south > north:
            raise ValueError("bbox south must be less than or equal to north.")

        if not self.is_latlon:
            if west > east:
                raise ValueError(
                    "Projected-grid bboxes must not cross the dateline (west <= east)."
                )

            x, y = self._bbox_outline_indices(west, south, east, north)
            if x.size == 0:
                raise ValueError("bbox does not intersect the grid.")

            def nint(value: float) -> int:
                """Round like Fortran NINT for HYSPLIT-compatible window bounds."""
                if value >= 0.0:
                    return int(np.floor(value + 0.5))
                return int(np.ceil(value - 0.5))

            xmin_1based = nint(float(x.min()) + 1.0)
            xmax_1based = nint(float(x.max()) + 1.0)
            ymin_1based = nint(float(y.min()) + 1.0)
            ymax_1based = nint(float(y.max()) + 1.0)

            if (
                xmax_1based < 1
                or xmin_1based > self.nx
                or ymax_1based < 1
                or ymin_1based > self.ny
            ):
                raise ValueError("bbox does not intersect the grid.")

            x_start = max(0, xmin_1based - 1)
            x_stop = min(self.nx, xmax_1based)
            y_start = max(0, ymin_1based - 1)
            y_stop = min(self.ny, ymax_1based)

            if x_stop <= x_start or y_stop <= y_start:
                raise ValueError("bbox does not intersect the grid.")

            return GridWindow(
                x_start=x_start,
                x_stop=x_stop,
                y_start=y_start,
                y_stop=y_stop,
            )

        # Lat/lon grids only reach here, so lon/lat are 1-D.
        coords = self.calculate_coords()
        lons = coords["lon"][1]
        lats = coords["lat"][1]

        # Normalize to [-180, 180] for comparison with bbox (which is always in
        # EPSG:4326 degrees). Grid lons may be in [0, 360] for global files.
        lons_norm = wrap_lons(lons)

        if west <= east:
            lon_mask = (lons_norm >= west) & (lons_norm <= east)
        else:
            lon_mask = (lons_norm >= west) | (lons_norm <= east)
        lat_mask = (lats >= south) & (lats <= north)

        x_idx = np.flatnonzero(lon_mask)
        y_idx = np.flatnonzero(lat_mask)

        if x_idx.size == 0 or y_idx.size == 0:
            raise ValueError("bbox does not intersect the grid.")

        x_start, x_stop = int(x_idx.min()), int(x_idx.max()) + 1
        gap = np.flatnonzero(~lon_mask[x_start:x_stop])
        if gap.size:
            # The bbox's columns are at both ends of the grid: it crosses the
            # grid's seam (0 E on a grid from 0 E, 180 on one from 180 W).
            if not self.wraps_lon:
                raise ValueError(
                    f"bbox {bbox} selects columns at both ends of a lat/lon grid "
                    "that does not span 360 degrees, which no single grid holds."
                )
            # Start at the eastern run and wrap on past the last column.
            x_start, x_stop = (
                x_start + int(gap[-1]) + 1,
                self.nx + x_start + int(gap[0]),
            )

        return GridWindow(
            x_start=x_start,
            x_stop=x_stop,
            y_start=int(y_idx.min()),
            y_stop=int(y_idx.max()) + 1,
        )

    def _bbox_outline_indices(
        self, west: float, south: float, east: float, north: float
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """
        Fractional indices of points along a lon/lat box's outline (projected grids).

        Each edge is sampled every 0.1 degree, and more finely where that
        leaves neighboring points on or near the grid more than half a cell
        apart, so the outline's extremes on each axis are found to a small
        fraction of a cell where an edge curves. The corners are always
        included; points the projection cannot map are dropped.
        """
        edges = (
            (west, south, east, south),
            (east, south, east, north),
            (east, north, west, north),
            (west, north, west, south),
        )
        counts = [
            max(2, math.ceil(max(abs(lon1 - lon0), abs(lat1 - lat0)) / 0.1) + 1)
            for lon0, lat0, lon1, lat1 in edges
        ]
        for _ in range(3):
            lon = np.concatenate(
                [np.linspace(e[0], e[2], n) for e, n in zip(edges, counts, strict=True)]
            )
            lat = np.concatenate(
                [np.linspace(e[1], e[3], n) for e, n in zip(edges, counts, strict=True)]
            )
            x, y = self.fractional_indices(lon, lat)
            # Only the spacing on or next to the grid matters.
            xc = np.clip(x, -1.0, self.nx)
            yc = np.clip(y, -1.0, self.ny)
            needed = []
            start = 0
            for n in counts:
                steps = np.hypot(
                    np.diff(xc[start : start + n]), np.diff(yc[start : start + n])
                )
                steps = steps[np.isfinite(steps)]
                step = float(steps.max()) if steps.size else 0.0
                needed.append(n if step <= 0.5 else math.ceil((n - 1) * step / 0.5) + 1)
                start += n
            if needed == counts:
                break
            counts = needed
        finite = np.isfinite(x) & np.isfinite(y)
        return x[finite], y[finite]

    def subset(self, window: GridWindow) -> "Grid":
        """
        Build a new grid definition for a rectangular subset.

        Parameters
        ----------
        window : GridWindow
            Zero-based half-open window into the parent grid.

        Returns
        -------
        Grid
            New grid whose cells are the window's cells of this grid, at the
            same positions.

        Notes
        -----
        A projected grid keeps this grid's sync point (``sync_lat``,
        ``sync_lon``) and moves only its grid coordinates, ``sync_x`` and
        ``sync_y``, by the window's start, so they may lie outside the subset
        (``sync_y = -326`` for a crop that starts at row 328). A new corner
        sync point would be the corner cell's latitude and longitude, which
        the index record stores as 7 characters (0.01 degree for a longitude
        past 100 degrees): the whole subset would move by up to 0.15 cells of
        a 3 km grid. HYSPLIT's map routines take a sync point anywhere.

        A lat-lon grid's sync point is always its first cell (HYSPLIT ignores
        ``sync_x`` and ``sync_y`` there), so the subset's is the window's
        corner cell, a whole number of cells from this grid's. It is stored
        exactly when the spacing is a multiple of 0.01 degree. The subset is
        written as HYSPLIT's ``xtrct_grid`` writes a crop: ``sync_lon`` in
        0-360 east and ``pole_lat``/``pole_lon`` its last cell, so
        ``pole_lon`` can pass 360. HYSPLIT's ``gbl2xy`` measures a longitude
        from a negative ``sync_lon`` only when ``pole_lon`` is not negative
        (``gblset``'s ``PRIME``); a 0-360 ``sync_lon`` needs no such case.
        The window covering the whole grid returns this grid unchanged.
        """
        x_limit = 2 * self.nx if self.wraps_lon else self.nx
        if window.x_stop > x_limit or window.nx > self.nx or window.y_stop > self.ny:
            raise ValueError("GridWindow extends beyond the grid bounds.")

        if self.is_latlon:
            return self._latlon_subset(window)
        subset_projection = replace(
            self.projection,
            sync_x=self.projection.sync_x - window.x_start,
            sync_y=self.projection.sync_y - window.y_start,
        )
        return Grid(projection=subset_projection, nx=window.nx, ny=window.ny)

    def _latlon_subset(self, window: GridWindow) -> "Grid":
        """Build the lat/lon subset :meth:`subset` describes, as ``xtrct_grid`` does."""
        if window == self.full_window():
            return self
        proj = self.projection
        dlon, dlat = proj.tangent_lon, proj.tangent_lat
        sync_lat = proj.sync_lat + window.y_start * dlat
        # East of Greenwich, so HYSPLIT's gbl2xy needs no PRIME (xtrct_grid's form)
        sync_lon = (proj.sync_lon + window.x_start * dlon) % 360.0
        subset_projection = replace(
            proj,
            pole_lat=sync_lat + (window.ny - 1) * dlat,
            pole_lon=sync_lon + (window.nx - 1) * dlon,
            sync_x=1.0,
            sync_y=1.0,
            sync_lat=sync_lat,
            sync_lon=sync_lon,
        )
        return Grid(projection=subset_projection, nx=window.nx, ny=window.ny)

    @override
    def __repr__(self) -> str:
        proj = self.projection.params.get("proj", "unknown")
        if proj == "latlong":
            proj = "latlon"
        if self.projection.is_latlon:
            return f"Grid({proj}, {self.nx}\u00d7{self.ny})"
        return f"Grid({proj} {self.projection.grid_size:g}km, {self.nx}\u00d7{self.ny})"
