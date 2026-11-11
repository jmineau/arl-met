"""
Grid and projection definitions for ARL meteorological data.

This module provides classes for representing ARL grid projections and
horizontal coordinate systems used in ARL meteorological files. Vertical
coordinates live in ``arlmet.vertical``.
"""

import math
from collections.abc import Callable
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


#: The most points a projected bbox's edge is sampled at (``Grid.window_from_bbox``).
#: Half a cell apart, a 360-degree edge on HRRR's 3 km grid needs about 26,000.
_OUTLINE_MAX_POINTS = 100_000

#: How many points first sample each edge of a bbox given in a CRS, to measure its
#: spacing on the grid (``Grid._crs_box_points_on_grid``).
_OUTLINE_FIRST_POINTS = 1_001

#: How far, in cells, a lat/lon column or row may lie from a bbox's edge and
#: still count as on it. The modulo-360 test of longitudes rounds at the size of a
#: number up to 360, about 6e-14 degrees, and a latitude ``lat_0 + j * dlat`` at
#: the size of the product (~1e-14 degrees when ``dlat`` is not exact in binary,
#: as 0.1 is not); 1e-6 cells (the same as ``_ops/sample/horizontal.py``'s ``_EDGE_TOLERANCE``)
#: is far above both and a few centimeters on a 0.25-degree grid.
_EDGE_TOLERANCE = 1e-6


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


def _wrap_lon(lon: float) -> float:
    """Wrap one longitude to [-180, 180)."""
    return (lon + 180.0) % 360.0 - 180.0


def _bbox_crs(bbox_crs: Any) -> pyproj.CRS | None:
    """
    Return the CRS a bbox is given in, or None for degrees of EPSG:4326.

    A system equal to EPSG:4326 apart from its axis order (``"OGC:CRS84"``) is
    None too: a bbox is always x before y.
    """
    if bbox_crs is None:
        return None
    crs = pyproj.CRS.from_user_input(bbox_crs)
    return None if crs.equals("EPSG:4326", ignore_axis_order=True) else crs


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
        Angle, in degrees clockwise from north, of the grid's y-axis at the reference
        longitude (``tangent_lon``); a nonzero value rotates a Lambert conformal or polar
        stereographic grid, as HYSPLIT's ``stcm1p`` does. A rotated Mercator grid raises
        ``NotImplementedError``. For lat-lon grids: ignored, as HYSPLIT ignores it
        (normally 0).
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
    >>> from arlmet._grid import Projection
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
        if self.orientation != 0.0 and not self.is_latlon and self.cone_angle == 0.0:
            # PROJ has no Mercator turned about a point; a cone can be turned by
            # moving its central meridian (``_rotation``).
            raise NotImplementedError(
                "Rotated Mercator grids (orientation != 0) are not supported."
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
            params.update(self._rotation())
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
            params.update(self._rotation())

        return params

    def _rotation(self) -> dict[str, Any]:
        """
        Return the pyproj parameters that turn a cone's grid by ``orientation``.

        HYSPLIT's ``stcm1p`` turns the projected plane by ``orientation``
        degrees about the reference point. On a cone (Lambert, or polar
        stereographic with a cone constant of +-1) a turn of the plane is a
        turn about the cone's apex plus a shift, and a turn about the apex is
        a move of the central meridian by ``orientation / sin(cone_angle)``
        degrees; the shift is taken up by the sync point (:attr:`Grid.origin`).
        HYSPLIT still cuts the cone at 180 degrees from ``tangent_lon``, so
        PROJ's own wrap about the moved meridian is turned off (``over``);
        :meth:`Grid._crs_lons` and :meth:`Grid._lonlat_at` put longitudes on
        HYSPLIT's side of that cut. Empty when not rotated.
        """
        if self.orientation == 0.0:
            return {}
        # The turn as the smallest angle; HYSPLIT's stlmbr wraps the reference
        # longitude to [-180, 180).
        turn = _wrap_lon(self.orientation)
        ref_lon = _wrap_lon(self.tangent_lon)
        gamma = math.sin(math.radians(self.cone_angle))
        return {"lon_0": ref_lon - turn / gamma, "over": True}

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
    >>> from arlmet._grid import Grid, Projection
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
        sync_proj_x, sync_proj_y = transformer.transform(
            self._crs_lons(proj.sync_lon), proj.sync_lat
        )

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
            Its x and y are the grid's axes, rotated grids included: there the
            central meridian is moved and PROJ's longitude wrap is off
            (``+over``), so give it longitudes within 180 degrees of
            ``tangent_lon``, where HYSPLIT cuts the projection.
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

        ii, jj = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
        lons, lats = self._lonlat_at(ii, jj)
        lons = wrap_lons(lons)

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
        proj_x, proj_y = transformer.transform(self._crs_lons(lon_arr), lat_arr)
        step = self.projection.grid_size * 1000.0
        if step == 0.0:
            raise ValueError("Projected grids require a non-zero grid_size.")
        x = np.asarray(proj_x, dtype=float) / step
        y = np.asarray(proj_y, dtype=float) / step
        return x, y

    def _crs_lons(self, lon: npt.NDArray[Any] | float) -> Any:
        """
        Longitudes as :attr:`crs` takes them: within 180 degrees of ``tangent_lon``.

        HYSPLIT measures a longitude from the reference longitude in
        [-180, 180) (``cnllxy``). PROJ does the same about its central
        meridian, which a rotated grid moves (:meth:`Projection._rotation`),
        so a rotated grid's CRS has that wrap turned off and its longitudes
        are put in place here. Other grids get their longitudes back as given.
        """
        proj = self.projection
        if proj.orientation == 0.0 or proj.is_latlon:
            return lon
        ref_lon = _wrap_lon(proj.tangent_lon)
        return (
            ref_lon + (np.asarray(lon, dtype=float) - ref_lon + 180.0) % 360.0 - 180.0
        )

    def meridian_convergence(
        self, lon: npt.NDArray[Any] | float, lat: npt.NDArray[Any] | float
    ) -> npt.NDArray[Any]:
        """
        Angle from true north to the grid's y-axis at each point, in degrees.

        Positive where the grid's y-axis points east of true north (true north
        lies counterclockwise of it), so at ``tangent_lon`` it is the grid's
        ``orientation``. Zero everywhere on a lat/lon grid.

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
        factors = pyproj.Proj(self.crs).get_factors(self._crs_lons(lon_arr), lat_arr)
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

    def window_from_bbox(
        self, bbox: tuple[float, float, float, float], *, bbox_crs: Any = None
    ) -> GridWindow:
        """
        Resolve a bounding box to grid indices.

        Without ``bbox_crs`` the box is in degrees of longitude and latitude,
        as described next; with it, see ``bbox_crs`` below.

        On a lat/lon grid the window holds the cells whose centers lie inside
        the box, edges included. A column or row within a millionth of a cell of
        an edge counts as on it, so rounding in the arithmetic (an east edge of
        ``-110`` against a column stored as 250 degrees, a north edge of
        ``41.4`` against a row stored as 41.400000000000006) never decides a
        column or row. HYSPLIT's ``xtrct_grid`` takes the column nearest each
        edge, so it agrees for edges on columns and can keep one more column for
        edges between them.

        On a projected grid the window is the smallest rectangle of cells
        that holds every point of the box that lies on the grid (in one of its
        cells). Its edges come from the box's outline, sampled at most half a
        cell apart on the grid, where it lies on the grid or crosses the grid's
        border, and from the grid's own corners that lie inside the box; the
        lowest and highest index on each axis is rounded to the nearest cell,
        like Fortran ``NINT``. Parallels and meridians curve on these grids, so
        an extreme is often the box's south-east or north-west corner, or a
        point partway along an edge. Points off the grid do not widen the
        window, so a box that reaches past the grid, across the meridian
        opposite the projection's center or to a pole the projection cannot
        map, gives the cells of its part on the grid (all of them, for a box
        that holds the whole grid). ``xtrct_grid`` uses only the south-west and
        north-east corners, on or off the grid, so its crop can leave out part
        of the box; for a box inside the grid this window holds that crop and
        can be a few cells larger. Longitudes count modulo 360 (``(440, 0, 450,
        10)`` is ``(80, 0, 90, 10)``), and a box 360 degrees wide or more holds
        every longitude.

        Raises ``ValueError`` when no point of the box lies on the grid.

        Parameters
        ----------
        bbox : tuple[float, float, float, float]
            Bounding box as ``(west, south, east, north)`` in degrees.
            Longitudes count modulo 360, so 180 and -180 are the same meridian
            and a box may reach past 180. ``west > east`` crosses the dateline,
            so ``(170, -170)`` is the box ``(170, 190)``: a window where it
            reaches the grid (GFS, Alaska) and ``ValueError`` where it does not
            (HRRR). An infinite width (``west=-inf`` or ``east=inf``) holds
            every longitude; a NaN longitude raises ``ValueError``. Edges may be any
            real numbers (NumPy scalars, ``Decimal``): they are converted to
            ``float`` first.
        bbox_crs : pyproj.CRS or str, optional
            The coordinate reference system of ``bbox``, as anything
            :meth:`pyproj.CRS.from_user_input` takes (``"EPSG:32612"``, a PROJ
            string, :attr:`crs`). The box is then ``(xmin, ymin, xmax, ymax)``
            in that system's units, easting before northing (longitude before
            latitude), whatever order the system names its axes in; the edges
            must be finite, with ``xmin <= xmax``. On either kind of grid the
            window is then the smallest rectangle of cells that holds every
            point of the box on the grid, found as on a projected grid above:
            from the box's outline, sampled about half a cell apart on the
            grid, and the grid's corners inside the box. A global lat/lon grid
            is cut at its seam, so a box across the seam gets every column. The
            default, None, and any system equal to EPSG:4326 (in either axis
            order, so ``"OGC:CRS84"`` too) mean degrees, as above. On a
            projected grid, ``bbox_crs=grid.crs`` gives the box in meters from
            the first cell's center, so cell ``(i, j)`` is at ``(i, j) *
            grid_size * 1000``.
        """
        # Python floats from here on: a float32 edge would keep ``width`` at float32,
        # whose rounding (~1e-6 degrees) swamps the tolerance below, and a Decimal
        # cannot be subtracted from a float.
        west, south, east, north = (float(v) for v in bbox)
        if south > north:
            raise ValueError("bbox south must be less than or equal to north.")

        crs = _bbox_crs(bbox_crs)
        if crs is not None:
            if not all(math.isfinite(v) for v in (west, south, east, north)):
                raise ValueError(f"bbox {bbox}: edges must be finite.")
            if west > east:
                raise ValueError(
                    f"bbox {bbox}: west must be less than or equal to east in bbox_crs."
                )
            return self._window_around(
                *self._crs_box_points_on_grid(crs, west, south, east, north),
                miss=lambda: self._bbox_misses(bbox, bbox_crs),
            )

        # From here on west <= east: ``west > east`` crosses the dateline, so the
        # box's east edge moves a turn east of where it was given.
        if east - west >= 360.0:  # every longitude, whatever the edges (inf included)
            west, east = -180.0, 180.0
        elif not (math.isfinite(west) and math.isfinite(east)):
            raise ValueError(
                f"bbox {bbox}: longitudes must not be NaN, and west must not be "
                "+inf nor east -inf."
            )
        elif west > east:
            east = west + (east - west) % 360.0

        if not self.is_latlon:
            return self._window_around(
                *self._bbox_points_on_grid(west, south, east, north),
                miss=lambda: self._bbox_misses(bbox, None),
            )

        # Lat/lon grids only reach here, so lon/lat are 1-D.
        coords = self.calculate_coords()
        lons = coords["lon"][1]
        lats = coords["lat"][1]

        # A column is inside when it lies between the box's west and east edges,
        # measured east of the west edge. Longitudes count modulo 360, so 180 and
        # -180 are one meridian (an edge there keeps the column, whichever way the
        # grid writes it) and a box may reach past 180, as HYSPLIT's xtrct_grid
        # allows. The modulo rounds at the size of a number up to 360 and the
        # width at the size of the box, so a column exactly on an edge can miss it
        # by ~1e-14 degrees: a column within _EDGE_TOLERANCE cells of an edge
        # counts as on it. A row is the same: ``lat_0 + j * dlat`` rounds in the
        # product, so a row at a latitude as typed (41.4) can lie a hair beyond it.
        tol = _EDGE_TOLERANCE * abs(self.projection.tangent_lon)
        lat_tol = _EDGE_TOLERANCE * abs(self.projection.tangent_lat)
        width = east - west
        if width >= 360.0:
            lon_mask = np.ones(lons.shape, dtype=bool)
        else:
            east_of_west = (lons - west) % 360.0
            lon_mask = (east_of_west <= width + tol) | (east_of_west >= 360.0 - tol)
        lat_mask = (lats >= south - lat_tol) & (lats <= north + lat_tol)

        x_idx = np.flatnonzero(lon_mask)
        y_idx = np.flatnonzero(lat_mask)

        if x_idx.size == 0 or y_idx.size == 0:
            raise self._bbox_misses(bbox, None)

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

    def _lonlat_at(
        self, x: npt.NDArray[Any], y: npt.NDArray[Any]
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """
        Longitude and latitude of fractional grid indices (projected grids).

        On a rotated Lambert grid PROJ's inverse can land a longitude a whole
        turn of the cone (``360 / sin(cone_angle)`` degrees) away from
        HYSPLIT's, when the grid is turned by more than
        ``180 * (1 - |sin(cone_angle)|)`` degrees; it is moved back to within
        180 degrees of ``tangent_lon``. Longitudes are not wrapped.
        """
        transformer = pyproj.Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        proj = self.projection
        step = proj.grid_size * 1000.0
        lon, lat = transformer.transform(np.asarray(x) * step, np.asarray(y) * step)
        lon = np.asarray(lon, dtype=float)
        if proj.orientation != 0.0:
            gamma = math.sin(math.radians(proj.cone_angle))
            turns = np.round((lon - _wrap_lon(proj.tangent_lon)) * gamma / 360.0)
            lon = lon - turns * 360.0 / gamma
        return lon, np.asarray(lat, dtype=float)

    @cached_property
    def _outline_spacing(self) -> tuple[float, float]:
        """
        Degrees of longitude and of latitude that move a point at most half a cell.

        The rates (cells per degree) are measured at the grid's border nodes:
        on a Lambert, polar stereographic or Mercator grid the scale grows
        away from the standard parallel or the pole, so its largest value on
        the grid is on the border.
        """
        i = np.arange(self.nx, dtype=float)
        j = np.arange(self.ny, dtype=float)
        lon, lat = self._lonlat_at(
            np.concatenate([i, i, np.zeros(self.ny), np.full(self.ny, self.nx - 1.0)]),
            np.concatenate([np.zeros(self.nx), np.full(self.nx, self.ny - 1.0), j, j]),
        )
        d = 1e-3
        lat_lo, lat_hi = np.maximum(lat - d, -90.0), np.minimum(lat + d, 90.0)

        Arr = npt.NDArray[Any]

        def half_cell(lon0: Arr, lat0: Arr, lon1: Arr, lat1: Arr, span: Any) -> float:
            """Degrees that move at most half a cell, from the largest rate."""
            x0, y0 = self.fractional_indices(lon0, lat0)
            x1, y1 = self.fractional_indices(lon1, lat1)
            with np.errstate(invalid="ignore"):
                rate = np.hypot(x1 - x0, y1 - y0) / span
            rate = rate[np.isfinite(rate)]
            return 0.5 / float(rate.max()) if rate.size and rate.max() > 0 else 0.1

        return (
            half_cell(lon - d, lat, lon + d, lat, 2 * d),
            half_cell(lon, lat_lo, lon, lat_hi, lat_hi - lat_lo),
        )

    def _bbox_points_on_grid(
        self, west: float, south: float, east: float, north: float
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """
        Fractional indices that bound the part of a lon/lat box on a projected grid.

        The grid's cells span ``[-0.5, n - 0.5]`` on each axis. Returned: the
        points of the box's outline inside the cells, where each outline
        segment enters and leaves them, and the cells' corners that lie inside
        the box; empty when no point of the box is on the grid. The outline is
        sampled at ``_outline_spacing`` (at most ``_OUTLINE_MAX_POINTS`` per
        edge). A segment longer than two cells jumps across the projection's
        cut or past a point it cannot map, so it is not followed across the
        grid; its ends still count.

        Longitudes count modulo 360: the box moves so that ``west`` lies in
        [-180, 180) and ``east`` is at most 360 degrees east of it (a wider box
        holds every longitude). PROJ returns infinity for a longitude beyond
        10 radians (573 degrees), so an outline left as given would lose the
        part of the box past that.
        """
        width = min(east - west, 360.0)
        west = (west + 180.0) % 360.0 - 180.0
        east = west + width
        dlon, dlat = self._outline_spacing
        edges = (
            (west, south, east, south, dlon),
            (east, south, east, north, dlat),
            (east, north, west, north, dlon),
            (west, north, west, south, dlat),
        )
        lons: list[npt.NDArray[Any]] = []
        lats: list[npt.NDArray[Any]] = []
        for lon0, lat0, lon1, lat1, spacing in edges:
            span = max(abs(lon1 - lon0), abs(lat1 - lat0))
            n = min(_OUTLINE_MAX_POINTS, max(2, math.ceil(span / spacing) + 1))
            lons.append(np.linspace(lon0, lon1, n))
            lats.append(np.linspace(lat0, lat1, n))
        x, y = self.fractional_indices(np.concatenate(lons), np.concatenate(lats))
        xs, ys = self._outline_on_grid(x, y)

        # The cells' corners inside the box (a box larger than the grid).
        cx, cy = self._cell_corners()
        clon, clat = self._lonlat_at(cx, cy)
        inside = (clat >= south) & (clat <= north)
        inside &= (clon - west) % 360.0 <= east - west
        return np.concatenate([xs, cx[inside]]), np.concatenate([ys, cy[inside]])

    def _cell_corners(self) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """Fractional indices of the four outer corners of the grid's cells."""
        lo, x_hi, y_hi = -0.5, self.nx - 0.5, self.ny - 0.5
        return np.array([lo, x_hi, lo, x_hi]), np.array([lo, lo, y_hi, y_hi])

    def _outline_on_grid(
        self, x: npt.NDArray[Any], y: npt.NDArray[Any]
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """
        Return the points of an outline (fractional indices) that bound its part on the grid.

        Returned: the outline's points inside the cells (``[-0.5, n - 0.5]``
        on each axis) and where each segment between two points enters and
        leaves them. A segment longer than two cells jumps across the
        projection's cut or past a point it cannot map, so it is not followed
        across the grid; its ends still count.
        """
        lo, x_hi, y_hi = -0.5, self.nx - 0.5, self.ny - 0.5
        finite = np.isfinite(x) & np.isfinite(y)
        on = finite & (x >= lo) & (x <= x_hi) & (y >= lo) & (y <= y_hi)
        xs, ys = [x[on]], [y[on]]

        # Where each segment enters and leaves the cells (Liang-Barsky clipping).
        # Points the projection cannot map are dropped first (inf - inf).
        x, y = np.where(finite, x, 0.0), np.where(finite, y, 0.0)
        x0, y0, dx, dy = x[:-1], y[:-1], np.diff(x), np.diff(y)
        keep = finite[:-1] & finite[1:] & (np.hypot(dx, dy) <= 2.0)
        t0, t1 = np.zeros(dx.size), np.ones(dx.size)
        with np.errstate(divide="ignore", invalid="ignore"):
            for p, q in (
                (-dx, x0 - lo),
                (dx, x_hi - x0),
                (-dy, y0 - lo),
                (dy, y_hi - y0),
            ):
                keep &= (p != 0) | (q >= 0)
                r = q / p
                t0 = np.where(p < 0, np.maximum(t0, r), t0)
                t1 = np.where(p > 0, np.minimum(t1, r), t1)
        keep &= t0 <= t1
        for t in (t0[keep], t1[keep]):
            xs.append(np.clip(x0[keep] + t * dx[keep], lo, x_hi))
            ys.append(np.clip(y0[keep] + t * dy[keep], lo, y_hi))
        return np.concatenate(xs), np.concatenate(ys)

    def _bbox_misses(self, bbox: Any, bbox_crs: Any) -> ValueError:
        """
        Return the error for a bbox with no cell of the grid in it.

        It names the box, the grid's extent and the order of the edges, since a
        box that misses is most often in (south, west, ...) order or in meters.
        """
        coords = self.calculate_coords()
        lons, lats = coords["lon"][1], coords["lat"][1]
        extent = (
            f"longitudes {np.nanmin(lons):.6g} to {np.nanmax(lons):.6g} and "
            f"latitudes {np.nanmin(lats):.6g} to {np.nanmax(lats):.6g}"
        )
        if bbox_crs is None:
            order = (
                "A bbox is (west, south, east, north) in degrees; pass bbox_crs= "
                "for one in another coordinate system."
            )
        else:
            order = (
                f"In bbox_crs={bbox_crs!r} a bbox is (xmin, ymin, xmax, ymax) in "
                "that system's units, easting first."
            )
        return ValueError(
            f"bbox {tuple(bbox)} does not intersect the grid, which covers {extent}. "
            f"{order}"
        )

    def _window_around(
        self,
        x: npt.NDArray[Any],
        y: npt.NDArray[Any],
        *,
        miss: Callable[[], ValueError],
    ) -> GridWindow:
        """
        Return the smallest window of cells that holds points on the grid.

        The points are fractional indices in ``[-0.5, n - 0.5]``; the lowest
        and highest on each axis are rounded to a cell like Fortran ``NINT``.
        Raises ``miss()`` when there are none (the box missed the grid).
        """
        if x.size == 0:
            raise miss()

        def nint(value: float) -> int:
            """Round like Fortran NINT (the 1-based values here are >= 0.5)."""
            return int(np.floor(value + 0.5))

        return GridWindow(
            x_start=max(0, nint(float(x.min()) + 1.0) - 1),
            x_stop=min(self.nx, nint(float(x.max()) + 1.0)),
            y_start=max(0, nint(float(y.min()) + 1.0) - 1),
            y_stop=min(self.ny, nint(float(y.max()) + 1.0)),
        )

    def _crs_box_points_on_grid(
        self, crs: pyproj.CRS, west: float, south: float, east: float, north: float
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """
        Fractional indices that bound the part of a box in ``crs`` on the grid.

        :meth:`_bbox_points_on_grid` for a rectangle in ``crs``'s own
        coordinates (x before y), on either kind of grid. The spacing of a
        system's units on the grid is not known ahead, so each edge is first
        sampled at ``_OUTLINE_FIRST_POINTS`` points; its longest step that can
        reach the grid (both ends mapped, not both past the same border of
        the cells) sets how many points the edge gets, to put them about half
        a cell apart (at most ``_OUTLINE_MAX_POINTS``). A step across a cut
        counts too, which costs points but not accuracy.

        A lat/lon grid's longitudes are measured within 180 degrees of its
        middle column, so a box over its west border is not split onto the far
        east (as :meth:`fractional_indices`, which counts east of the first
        column, would split it); a global grid is cut at its seam.
        """
        to_crs = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        proj = self.projection
        mid_x = (self.nx - 1) / 2.0

        def indices(
            bx: npt.NDArray[Any], by: npt.NDArray[Any]
        ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
            """Fractional indices of points in ``crs``."""
            lon, lat = to_crs.transform(bx, by, direction="INVERSE")
            lon, lat = np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
            if not self.is_latlon:
                return self.fractional_indices(lon, lat)
            lon_0, lat_0 = self.origin
            mid_lon = lon_0 + mid_x * proj.tangent_lon
            east_of_mid = (lon - mid_lon + 180.0) % 360.0 - 180.0
            return (
                mid_x + east_of_mid / proj.tangent_lon,
                (lat - lat_0) / proj.tangent_lat,
            )

        lo, x_hi, y_hi = -0.5, self.nx - 0.5, self.ny - 0.5

        def half_cell_points(x: npt.NDArray[Any], y: npt.NDArray[Any]) -> int:
            """Points that put an edge sampled as ``x, y`` about half a cell apart."""
            finite = np.isfinite(x) & np.isfinite(y)
            x, y = np.where(finite, x, 0.0), np.where(finite, y, 0.0)
            reach = finite[:-1] & finite[1:]
            for v, hi in ((x, x_hi), (y, y_hi)):
                reach &= ~((v[:-1] < lo) & (v[1:] < lo))
                reach &= ~((v[:-1] > hi) & (v[1:] > hi))
            steps = np.hypot(np.diff(x), np.diff(y))[reach]
            cells = float(steps.max()) * (x.size - 1) if steps.size else 0.0
            return min(_OUTLINE_MAX_POINTS, max(2, math.ceil(2.0 * cells) + 1))

        def outline(
            counts: list[int],
        ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
            """Fractional indices of the outline, ``counts[k]`` points along edge k."""
            corners = [(west, south), (east, south), (east, north), (west, north)]
            bx: list[npt.NDArray[Any]] = []
            by: list[npt.NDArray[Any]] = []
            for (x0, y0), (x1, y1), n in zip(
                corners, corners[1:] + corners[:1], counts, strict=True
            ):
                t = np.linspace(0.0, 1.0, n)
                bx.append(x0 + t * (x1 - x0))
                by.append(y0 + t * (y1 - y0))
            return indices(np.concatenate(bx), np.concatenate(by))

        # One transform per pass: building a transformer costs more than the points.
        first = _OUTLINE_FIRST_POINTS
        x, y = outline([first] * 4)
        counts = [
            half_cell_points(
                x[k * first : (k + 1) * first], y[k * first : (k + 1) * first]
            )
            for k in range(4)
        ]
        x, y = outline(counts)
        on_x, on_y = self._outline_on_grid(x, y)

        # The cells' corners inside the box (a box larger than the grid).
        cx, cy = self._cell_corners()
        if self.is_latlon:
            lon_0, lat_0 = self.origin
            clon, clat = lon_0 + cx * proj.tangent_lon, lat_0 + cy * proj.tangent_lat
        else:
            clon, clat = self._lonlat_at(cx, cy)
        bx, by = to_crs.transform(clon, clat)
        bx, by = np.asarray(bx, dtype=float), np.asarray(by, dtype=float)
        inside = (bx >= west) & (bx <= east) & (by >= south) & (by <= north)
        return (
            np.concatenate([on_x, cx[inside]]),
            np.concatenate([on_y, cy[inside]]),
        )

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
