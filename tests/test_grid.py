"""Tests for arlmet.grid module."""

from dataclasses import FrozenInstanceError, replace
from decimal import Decimal

import numpy as np
import pyproj
import pytest

from arlmet.grid import _OUTLINE_MAX_POINTS, Grid, GridWindow, Projection, wrap_lons
from arlmet.vertical import (
    HybridAxis,
    PressureAxis,
    SigmaAxis,
    TerrainAxis,
    VerticalAxis,
)


class TestWrapLons:
    """Tests for wrap_lons function."""

    def test_wrap_lons_positive(self):
        """Test wrapping positive longitudes."""
        lons = np.array([0, 90, 180, 270, 360])
        expected = np.array([0, 90, -180, -90, 0])  # 180 wraps to -180
        result = wrap_lons(lons)
        np.testing.assert_array_equal(result, expected)

    def test_wrap_lons_negative(self):
        """Test wrapping negative longitudes."""
        lons = np.array([-180, -90, 0, 90])
        expected = np.array([-180, -90, 0, 90])  # -180 stays as -180
        result = wrap_lons(lons)
        np.testing.assert_array_equal(result, expected)

    def test_wrap_lons_out_of_range(self):
        """Test wrapping out of range longitudes."""
        lons = np.array([540, -360, 720])
        expected = np.array([-180, 0, 0])  # 540->180->-180
        result = wrap_lons(lons)
        np.testing.assert_array_equal(result, expected)


class TestProjection:
    """Tests for Projection class."""

    def test_latlon_projection(self):
        """Test lat-lon projection initialization."""
        proj = Projection(
            pole_lat=90.0,
            pole_lon=0.0,
            tangent_lat=1.0,
            tangent_lon=1.0,
            grid_size=0.0,
            orientation=0.0,
            cone_angle=0.0,
            sync_x=1.0,
            sync_y=1.0,
            sync_lat=0.0,
            sync_lon=0.0,
        )
        assert proj.is_latlon is True
        assert proj.params["proj"] == "latlong"

    def test_polar_stereographic_projection(self):
        """Test polar stereographic projection initialization."""
        proj = Projection(
            pole_lat=90.0,
            pole_lon=0.0,
            tangent_lat=60.0,
            tangent_lon=0.0,
            grid_size=50.0,
            orientation=0.0,
            cone_angle=90.0,
            sync_x=1.0,
            sync_y=1.0,
            sync_lat=60.0,
            sync_lon=0.0,
        )
        assert proj.is_latlon is False
        assert proj.params["proj"] == "stere"
        assert proj.params["lat_0"] == 90.0

    def test_lambert_conformal_projection(self):
        """Test Lambert conformal projection initialization."""
        proj = Projection(
            pole_lat=90.0,
            pole_lon=0.0,
            tangent_lat=40.0,
            tangent_lon=-100.0,
            grid_size=12.0,
            orientation=0.0,
            cone_angle=30.0,
            sync_x=1.0,
            sync_y=1.0,
            sync_lat=40.0,
            sync_lon=-100.0,
        )
        assert proj.is_latlon is False
        assert proj.params["proj"] == "lcc"
        assert proj.params["lat_1"] == 30.0

    def test_lambert_scale_is_one_at_equal_latitudes(self):
        """A cone true to scale at its tangent latitude is not rescaled."""
        proj = Projection(
            pole_lat=90.0,
            pole_lon=0.0,
            tangent_lat=38.5,
            tangent_lon=-97.5,
            grid_size=3.0,
            orientation=0.0,
            cone_angle=38.5,
            sync_x=1.0,
            sync_y=1.0,
            sync_lat=21.1381,
            sync_lon=-122.719,
        )
        assert proj.params["k_0"] == 1.0

    @pytest.mark.parametrize(
        ("tangent_lat", "cone_angle"), [(25.0, 37.8589), (45.0, 38.5), (-40.0, -30.0)]
    )
    def test_lambert_grid_size_holds_at_reference_latitude(
        self, tangent_lat, cone_angle
    ):
        """The map scale is 1 at tangent_lat, where HYSPLIT defines grid_size."""
        proj = Projection(
            pole_lat=90.0,
            pole_lon=0.0,
            tangent_lat=tangent_lat,
            tangent_lon=-98.0,
            grid_size=27.0,
            orientation=0.0,
            cone_angle=cone_angle,
            sync_x=1.0,
            sync_y=1.0,
            sync_lat=tangent_lat,
            sync_lon=-98.0,
        )
        grid = Grid(projection=proj, nx=10, ny=10)
        factors = pyproj.Proj(grid.crs).get_factors(
            [-98.0, -120.0], [tangent_lat, tangent_lat]
        )
        np.testing.assert_allclose(factors.parallel_scale, 1.0, rtol=1e-9)
        np.testing.assert_allclose(factors.meridional_scale, 1.0, rtol=1e-9)

    def test_lambert_reference_latitude_matches_hysplit(self):
        """A WRF grid from arw2arl (reference latitude 25, cone 37.86) as HYSPLIT."""
        # The index record of a 27 km WRF run converted by arw2arl. The corners
        # are from HYSPLIT's stlmbr/stcm1p/cxy2ll, set up as metini.f does.
        proj = Projection(
            pole_lat=37.8589,
            pole_lon=-98.0,
            tangent_lat=25.0,
            tangent_lon=-98.0,
            grid_size=27.0,
            orientation=0.0,
            cone_angle=37.8589,
            sync_x=108.5,
            sync_y=87.5,
            sync_lat=37.0,
            sync_lon=-98.0,
        )
        grid = Grid(projection=proj, nx=216, ny=174)
        coords = grid.calculate_coords()
        lat, lon = coords["lat"][1], coords["lon"][1]
        hysplit = {
            (0, 0): (12.625857, -123.338722),
            (-1, -1): (52.003647, -54.417603),
            (0, -1): (12.625857, -72.661278),
            (-1, 0): (52.003647, -141.582397),
        }
        for (j, i), (hlat, hlon) in hysplit.items():
            assert lat[j, i] == pytest.approx(hlat, abs=2e-4)
            assert lon[j, i] == pytest.approx(hlon, abs=2e-4)
        x, y = grid.fractional_indices(-54.417603, 52.003647)
        assert float(x) == pytest.approx(215.0, abs=1e-3)
        assert float(y) == pytest.approx(173.0, abs=1e-3)

    def test_stereographic_follows_cone_angle_not_pole_lat(self):
        """As HYSPLIT's stlmbr, cone_angle +-90 is polar stereographic."""
        fields = dict(
            pole_lon=0.0,
            tangent_lat=60.0,
            tangent_lon=-150.0,
            grid_size=11.904,
            orientation=0.0,
            sync_x=1.0,
            sync_y=1.0,
            sync_lat=40.53,
            sync_lon=-178.571,
        )
        north = Projection(pole_lat=0.0, cone_angle=90.0, **fields)
        assert north.params["proj"] == "stere"
        assert north.params["lat_0"] == 90.0
        assert (
            north.params == Projection(pole_lat=90.0, cone_angle=90.0, **fields).params
        )
        south = Projection(pole_lat=0.0, cone_angle=-90.0, **fields)
        assert south.params["lat_0"] == -90.0

    def test_mercator_projection(self):
        """Test Mercator projection initialization."""
        proj = Projection(
            pole_lat=90.0,
            pole_lon=0.0,
            tangent_lat=0.0,
            tangent_lon=0.0,
            grid_size=50.0,
            orientation=0.0,
            cone_angle=0.0,
            sync_x=1.0,
            sync_y=1.0,
            sync_lat=0.0,
            sync_lon=0.0,
        )
        assert proj.is_latlon is False
        assert proj.params["proj"] == "merc"

    def test_rotated_mercator_raises(self):
        """A rotated Mercator grid raises NotImplementedError (PROJ cannot turn one)."""
        with pytest.raises(NotImplementedError, match="Mercator"):
            Projection(
                pole_lat=90.0,
                pole_lon=0.0,
                tangent_lat=1.0,
                tangent_lon=1.0,
                grid_size=50.0,
                orientation=15.0,  # non-zero orientation
                cone_angle=0.0,
                sync_x=1.0,
                sync_y=1.0,
                sync_lat=0.0,
                sync_lon=0.0,
            )


# Real grids, their 12 projection fields as the files store them: the
# 2020-02-08 HRRR file, and the NAMS Alaska and Hawaii files of 2025.
_HRRR = Projection(
    pole_lat=90.0,
    pole_lon=0.0,
    tangent_lat=38.5,
    tangent_lon=262.5,
    grid_size=3.0,
    orientation=0.0,
    cone_angle=38.5,
    sync_x=1.0,
    sync_y=1.0,
    sync_lat=21.1381,
    sync_lon=237.281,
)
_NAMS_AK = Projection(
    pole_lat=90.0,
    pole_lon=0.0,
    tangent_lat=60.0,
    tangent_lon=210.0,
    grid_size=11.904,
    orientation=0.0,
    cone_angle=90.0,
    sync_x=1.0,
    sync_y=1.0,
    sync_lat=40.53,
    sync_lon=181.429,
)
_NAMS_HI = Projection(
    pole_lat=90.0,
    pole_lon=0.0,
    tangent_lat=20.0,
    tangent_lon=206.131,
    grid_size=2.5,
    orientation=0.0,
    cone_angle=0.0,
    sync_x=1.0,
    sync_y=1.0,
    sync_lat=18.073,
    sync_lon=198.475,
)

_NAM12 = Projection(
    pole_lat=90.0,
    pole_lon=0.0,
    tangent_lat=25.0,
    tangent_lon=265.0,
    grid_size=12.191,
    orientation=0.0,
    cone_angle=25.0,
    sync_x=1.0,
    sync_y=1.0,
    sync_lat=12.19,
    sync_lon=226.541,
)
_NAMS_CONUS = Projection(
    pole_lat=90.0,
    pole_lon=0.0,
    tangent_lat=52.0,
    tangent_lon=254.0,
    grid_size=12.0,
    orientation=0.0,
    cone_angle=52.0,
    sync_x=1.0,
    sync_y=1.0,
    sync_lat=19.254,
    sync_lon=235.41,
)


def _nint(value: float) -> int:
    """Round like Fortran NINT."""
    value = float(value)
    if value >= 0.0:
        return int(np.floor(value + 0.5))
    return int(np.ceil(value - 0.5))


def _dense_outline(grid: Grid, bbox, n: int = 20001):
    """Fractional indices of *n* points along each edge of a lon/lat box."""
    west, south, east, north = bbox
    t = np.linspace(0.0, 1.0, n)
    lon = np.concatenate(
        [west + (east - west) * t, np.full(n, east), east - (east - west) * t]
        + [np.full(n, west)]
    )
    lat = np.concatenate(
        [np.full(n, south), south + (north - south) * t, np.full(n, north)]
        + [north - (north - south) * t]
    )
    return grid.fractional_indices(lon, lat)


def _corner_window(grid: Grid, bbox) -> GridWindow | None:
    """HYSPLIT ``xtrct_grid``'s window: the SW and NE corners only, NINT."""
    x, y = grid.fractional_indices([bbox[0], bbox[2]], [bbox[1], bbox[3]])
    x1 = [_nint(v + 1.0) for v in x]
    y1 = [_nint(v + 1.0) for v in y]
    x_start, x_stop = max(0, min(x1) - 1), min(grid.nx, max(x1))
    y_start, y_stop = max(0, min(y1) - 1), min(grid.ny, max(y1))
    if x_stop <= x_start or y_stop <= y_start:
        return None  # the two corners' window misses the grid
    return GridWindow(x_start=x_start, x_stop=x_stop, y_start=y_start, y_stop=y_stop)


def _reference_window(grid: Grid, bbox, n: int = 20001, per_cell: int = 10):
    """
    The window of the part of *bbox* on *grid*, by brute force, or None.

    Dense outline points inside the grid's cells plus dense points of the
    cells' border inside the box, rounded with NINT.
    """
    x, y = _dense_outline(grid, bbox, n)
    on = (x >= -0.5) & (x <= grid.nx - 0.5) & (y >= -0.5) & (y <= grid.ny - 0.5)
    xs, ys = [x[on]], [y[on]]
    tx = np.linspace(-0.5, grid.nx - 0.5, per_cell * grid.nx + 1)
    ty = np.linspace(-0.5, grid.ny - 0.5, per_cell * grid.ny + 1)
    bx = np.concatenate(
        [tx, tx, np.full(ty.size, -0.5), np.full(ty.size, grid.nx - 0.5)]
    )
    by = np.concatenate(
        [np.full(tx.size, -0.5), np.full(tx.size, grid.ny - 0.5), ty, ty]
    )
    step = grid.projection.grid_size * 1000.0
    to_lonlat = pyproj.Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    lon, lat = to_lonlat.transform(bx * step, by * step)
    lon, lat = np.asarray(lon), np.asarray(lat)
    west, south, east, north = bbox
    inside = (lat >= south) & (lat <= north) & ((lon - west) % 360.0 <= east - west)
    xs.append(bx[inside])
    ys.append(by[inside])
    x, y = np.concatenate(xs), np.concatenate(ys)
    if not x.size:
        return None
    return GridWindow(
        x_start=max(0, _nint(x.min() + 1.0) - 1),
        x_stop=min(grid.nx, _nint(x.max() + 1.0)),
        y_start=max(0, _nint(y.min() + 1.0) - 1),
        y_stop=min(grid.ny, _nint(y.max() + 1.0)),
    )


def _assert_close_windows(a: GridWindow, b: GridWindow, cells: int = 1) -> None:
    """Assert that two windows' edges are at most *cells* apart."""
    for name in ("x_start", "x_stop", "y_start", "y_stop"):
        assert abs(getattr(a, name) - getattr(b, name)) <= cells, (name, a, b)


def _assert_in_window_cells(grid: Grid, window: GridWindow, x, y, tol=1e-3):
    """Assert that points on the grid lie in a cell (center +- 0.5) of *window*."""
    x, y = np.atleast_1d(x), np.atleast_1d(y)
    on = (x >= -0.5) & (x <= grid.nx - 0.5) & (y >= -0.5) & (y <= grid.ny - 0.5)
    x, y = x[on], y[on]
    if not x.size:
        return
    assert x.min() >= window.x_start - 0.5 - tol
    assert x.max() <= window.x_stop - 0.5 + tol
    assert y.min() >= window.y_start - 0.5 - tol
    assert y.max() <= window.y_stop - 0.5 + tol


class TestGrid:
    """Tests for Grid class."""

    def test_latlon_grid(self):
        """Test lat-lon grid initialization."""
        proj = Projection(
            pole_lat=90.0,
            pole_lon=180.0,
            tangent_lat=0.5,
            tangent_lon=0.5,
            grid_size=0.0,
            orientation=0.0,
            cone_angle=0.0,
            sync_x=1.0,
            sync_y=1.0,
            sync_lat=-90.0,
            sync_lon=-180.0,
        )
        grid = Grid(projection=proj, nx=360, ny=180)

        assert grid.is_latlon is True
        assert grid.dims == ("lat", "lon")
        coords = grid.calculate_coords()
        assert set(coords) == {"lon", "lat"}
        assert coords["lon"][0] == ("lon",)
        assert coords["lat"][0] == ("lat",)
        assert len(coords["lon"][1]) == 360
        assert len(coords["lat"][1]) == 180

    def test_projected_grid(self):
        """Test projected grid initialization."""
        proj = Projection(
            pole_lat=90.0,
            pole_lon=0.0,
            tangent_lat=60.0,
            tangent_lon=0.0,
            grid_size=50.0,
            orientation=0.0,
            cone_angle=90.0,
            sync_x=1.0,
            sync_y=1.0,
            sync_lat=60.0,
            sync_lon=0.0,
        )
        grid = Grid(projection=proj, nx=100, ny=100)

        assert grid.is_latlon is False
        assert grid.dims == ("y", "x")
        coords = grid.calculate_coords()
        assert set(coords) == {"x", "y", "lon", "lat"}
        assert coords["x"][0] == ("x",)
        assert coords["y"][0] == ("y",)
        assert coords["lon"][0] == ("y", "x")
        assert coords["lat"][0] == ("y", "x")
        assert coords["lon"][1].shape == (100, 100)

    def test_window_from_bbox_latlon(self):
        proj = Projection(
            pole_lat=90.0,
            pole_lon=180.0,
            tangent_lat=1.0,
            tangent_lon=1.0,
            grid_size=0.0,
            orientation=0.0,
            cone_angle=0.0,
            sync_x=1.0,
            sync_y=1.0,
            sync_lat=-10.0,
            sync_lon=20.0,
        )
        grid = Grid(projection=proj, nx=20, ny=20)

        window = grid.window_from_bbox((23.0, -7.0, 25.0, -5.0))

        assert window == GridWindow(x_start=3, x_stop=6, y_start=3, y_stop=6)

    def test_window_from_bbox_latlon_0_360_grid(self):
        """Global 0-360 grid (GDAS-style) with a negative-longitude bbox."""
        proj = Projection(
            pole_lat=90.0,
            pole_lon=180.0,
            tangent_lat=1.0,
            tangent_lon=1.0,
            grid_size=0.0,
            orientation=0.0,
            cone_angle=0.0,
            sync_x=1.0,
            sync_y=1.0,
            sync_lat=-90.0,
            sync_lon=0.0,
        )
        # 360×181 global lat-lon grid, lons 0→359, lats -90→90
        grid = Grid(projection=proj, nx=360, ny=181)

        # SLV bbox in [-180, 180] convention — should find western NA cells
        bbox = (-114.0, 39.5, -110.5, 42.0)
        window = grid.window_from_bbox(bbox)

        # Lons 246–249 correspond to -114 to -111 in 0-360
        assert window.x_start >= 246
        assert window.x_stop <= 250
        assert window.x_start < window.x_stop
        assert window.y_start < window.y_stop

    def test_window_from_bbox_projected_rounds_the_outline(self):
        """
        A projected window is NINT of the extremes of the box's part on the grid (#131).

        Far west of the Lambert grid's central meridian the box is rotated in
        index space: its south-east corner is the lowest row and its
        north-west corner the highest, so the south-west/north-east window
        that ``xtrct_grid`` crops leaves both out.
        """
        proj = _lambert_projection(sync_lat=30.0, sync_lon=-125.0)
        grid = Grid(projection=proj, nx=800, ny=600)
        bbox = (-124.0, 31.0, -120.0, 35.0)

        window = grid.window_from_bbox(bbox)
        # The box reaches below the grid: only its part on the grid counts.
        assert window == _reference_window(grid, bbox)
        assert window.y_start == 0
        corner = _corner_window(grid, bbox)
        assert corner is not None
        assert window.y_start < corner.y_start and window.y_stop > corner.y_stop
        for lon, lat in ((bbox[2], bbox[1]), (bbox[0], bbox[3])):  # SE, NW
            _assert_in_window_cells(grid, window, *grid.fractional_indices(lon, lat))

    def test_window_from_bbox_hrrr_holds_every_node_in_the_box(self):
        """
        On the real HRRR grid every node inside the SLV box is in the crop (#131).

        The grid is the 2020-02-08 HRRR file's (its 12 projection fields as
        stored). ``xtrct_grid``'s window, from the south-west and north-east
        corners only, is x 444:543, y 606:685 and misses 1,076 of these nodes.
        """
        grid = Grid(projection=_HRRR, nx=1799, ny=1059)
        bbox = (-113.5, 39.5, -110.5, 42.0)

        window = grid.window_from_bbox(bbox)
        assert window == GridWindow(x_start=444, x_stop=543, y_start=592, y_stop=698)
        assert _corner_window(grid, bbox) == GridWindow(
            x_start=444, x_stop=543, y_start=606, y_stop=685
        )
        coords = grid.calculate_coords()
        lon, lat = wrap_lons(coords["lon"][1]), coords["lat"][1]
        inside = (lon >= bbox[0]) & (lon <= bbox[2]) & (lat >= bbox[1])
        inside &= lat <= bbox[3]
        rows, cols = np.nonzero(inside)
        assert rows.size == 7818
        assert window.x_start <= cols.min() and cols.max() < window.x_stop
        assert window.y_start <= rows.min() and rows.max() < window.y_stop

    @pytest.mark.parametrize(
        ("projection", "nx", "ny"),
        [
            pytest.param(_HRRR, 1799, 1059, id="lambert"),
            pytest.param(_NAMS_AK, 387, 276, id="polar-stereographic"),
            pytest.param(_NAMS_HI, 321, 225, id="mercator"),
            pytest.param(
                replace(_HRRR, orientation=15.0), 1799, 1059, id="rotated-lambert"
            ),
            pytest.param(
                replace(_NAMS_AK, orientation=-40.0), 387, 276, id="rotated-stereo"
            ),
        ],
    )
    def test_window_from_bbox_projected_holds_random_boxes(self, projection, nx, ny):
        """Every point of a random box on the grid lies in a cell of its window (#131)."""
        grid = Grid(projection=projection, nx=nx, ny=ny)
        coords = grid.calculate_coords()
        lons, lats = wrap_lons(coords["lon"][1]), coords["lat"][1]
        lon_span, lat_span = np.ptp(lons), np.ptp(lats)
        rng = np.random.default_rng(131)
        for _ in range(25):
            # boxes from a few cells to wider than the grid
            row, col = rng.integers(0, ny), rng.integers(0, nx)
            half_lon = rng.uniform(0.01, 0.6) * lon_span
            half_lat = rng.uniform(0.01, 0.6) * lat_span
            lon0, lat0 = float(lons[row, col]), float(lats[row, col])
            bbox = (
                lon0 - half_lon,
                max(lat0 - half_lat, -80.0),
                lon0 + half_lon,
                min(lat0 + half_lat, 80.0),
            )
            window = grid.window_from_bbox(bbox)
            # tight: within a cell of the brute-force window on every side
            reference = _reference_window(grid, bbox, n=4001, per_cell=4)
            assert reference is not None
            _assert_close_windows(window, reference)
            # holds every node in the box and every sampled point of it on the grid
            inbox = (lons >= bbox[0]) & (lons <= bbox[2])
            inbox &= (lats >= bbox[1]) & (lats <= bbox[3])
            rows, cols = np.nonzero(inbox)
            assert window.x_start <= cols.min() and cols.max() < window.x_stop
            assert window.y_start <= rows.min() and rows.max() < window.y_stop
            _assert_in_window_cells(grid, window, *_dense_outline(grid, bbox, n=2001))
            lon = rng.uniform(bbox[0], bbox[2], 2000)
            lat = rng.uniform(bbox[1], bbox[3], 2000)
            _assert_in_window_cells(grid, window, *grid.fractional_indices(lon, lat))
            # a box inside the grid holds xtrct_grid's window
            x, y = grid.fractional_indices(
                [bbox[0], bbox[2], bbox[2], bbox[0]],
                [bbox[1], bbox[1], bbox[3], bbox[3]],
            )
            if np.all((x >= 0) & (x <= nx - 1) & (y >= 0) & (y <= ny - 1)):
                corner = _corner_window(grid, bbox)
                assert corner is not None
                assert window.x_start <= corner.x_start
                assert window.y_start <= corner.y_start
                assert corner.x_stop <= window.x_stop
                assert corner.y_stop <= window.y_stop

    def test_window_from_bbox_projected_unmappable_box_raises(self):
        """A box the projection cannot map (a northern Lambert's south pole) raises."""
        grid = Grid(projection=_HRRR, nx=1799, ny=1059)
        with pytest.raises(ValueError, match="does not intersect"):
            grid.window_from_bbox((-160.0, -90.0, -150.0, -90.0))

    @pytest.mark.parametrize(
        ("projection", "nx", "ny", "bbox"),
        [
            pytest.param(_HRRR, 1799, 1059, (-180, 0, 180, 90), id="hrrr-north"),
            pytest.param(_HRRR, 1799, 1059, (-180, -90, 180, 90), id="hrrr-globe"),
            pytest.param(_HRRR, 1799, 1059, (-180, 20, 180, 60), id="hrrr-band"),
            pytest.param(_NAM12, 614, 428, (-180, 0, 180, 90), id="nam12-north"),
            pytest.param(_NAMS_CONUS, 461, 401, (-180, 0, 180, 90), id="conus-north"),
            pytest.param(_NAMS_HI, 321, 225, (-180, 0, 180, 60), id="hawaii-band"),
            pytest.param(_NAMS_AK, 387, 276, (-180, -90, 180, 90), id="alaska-globe"),
        ],
    )
    def test_window_from_bbox_box_across_the_cut_holds_the_grid(
        self, monkeypatch, projection, nx, ny, bbox
    ):
        """
        A box across the projection's cut or to its far pole is bounded (#139 review).

        These boxes hold the whole grid. The outline is sampled at a fixed
        spacing with a cap, never refined across the jump at the cut, so the
        points mapped stay bounded (an earlier version ran out of memory).
        """
        largest = [0]
        original = Grid.fractional_indices

        def counting(self, lon, lat):
            largest[0] = max(largest[0], np.size(lon))
            return original(self, lon, lat)

        monkeypatch.setattr(Grid, "fractional_indices", counting)
        grid = Grid(projection=projection, nx=nx, ny=ny)
        assert grid.window_from_bbox(bbox) == grid.full_window()
        assert largest[0] <= 4 * _OUTLINE_MAX_POINTS

    @pytest.mark.parametrize("turns", [-3, -1, 1, 2, 3])
    def test_window_from_bbox_longitudes_beyond_a_turn_count_modulo_360(self, turns):
        """
        A box whose longitudes pass 573 degrees is the same box shifted (#139 review).

        PROJ maps no longitude beyond 10 radians (infinity), so sampling the
        outline as given left the part past 573 degrees out and returned a
        window too small, with no error.
        """
        grid = Grid(projection=_NAM12, nx=614, ny=428)
        west, south, east, north = 445.0004 - 360, -50.2358, 608.6778 - 360, 59.9860
        window = grid.window_from_bbox((west, south, east, north))
        assert (window.x_start, window.x_stop) == (0, 254)
        shifted = (west + 360 * turns, south, east + 360 * turns, north)
        assert grid.window_from_bbox(shifted) == window

    @pytest.mark.parametrize(
        "bbox",
        [(170, 50, -170, 60), (179, 55, -175, 65), (-200, 50, -210, 60)],
    )
    def test_window_from_bbox_across_the_dateline_on_a_projected_grid(self, bbox):
        """``west > east`` crosses the dateline on a projected grid too (#25)."""
        grid = Grid(projection=_NAMS_AK, nx=387, ny=276)
        west, south, east, north = bbox
        unwrapped = (west, south, west + (east - west) % 360.0, north)
        window = grid.window_from_bbox(bbox)
        assert window == grid.window_from_bbox(unwrapped)
        assert window == _reference_window(grid, unwrapped)

    def test_window_from_bbox_across_the_dateline_off_a_projected_grid_raises(self):
        """A box across the dateline that misses the grid (HRRR) raises (#25)."""
        grid = Grid(projection=_HRRR, nx=1799, ny=1059)
        with pytest.raises(ValueError, match="does not intersect"):
            grid.window_from_bbox((170, 20, -170, 60))

    def test_window_from_bbox_projected_nan_or_infinite_edges(self):
        """Longitudes are checked as on a lat/lon grid; -inf to 0 holds the grid."""
        grid = Grid(projection=_NAM12, nx=614, ny=428)
        inf, nan = float("inf"), float("nan")
        for bbox in [(10, 0, nan, 10), (nan, 0, 10, 10), (inf, 0, -inf, 10)]:
            with pytest.raises(ValueError, match="longitudes"):
                grid.window_from_bbox(bbox)
        assert grid.window_from_bbox((-inf, 0, 0, 90)) == grid.full_window()

    def test_window_from_bbox_box_wider_than_a_turn_holds_the_grid(self):
        """A box 2000 degrees wide holds every longitude, wherever it starts (#139 review)."""
        grid = Grid(projection=_NAM12, nx=614, ny=428)
        assert grid.window_from_bbox((-1000, 0, 1000, 90)) == grid.full_window()
        assert grid.window_from_bbox((-180, 0, 180, 90)) == grid.full_window()

    @pytest.mark.parametrize(
        ("projection", "nx", "ny", "bbox", "expected"),
        [
            # the lat 40-41 band reaches the grid only at its south-west corner
            pytest.param(
                _NAMS_AK,
                387,
                276,
                (-179.99, 40, 179.99, 41),
                (0, 13, 0, 7),
                id="ak-band",
            ),
            # south of the grid's south edge only its part on the grid counts
            pytest.param(
                _HRRR,
                1799,
                1059,
                (-140, 10, -100, 25),
                (0, 814, 0, 155),
                id="hrrr-south",
            ),
        ],
    )
    def test_window_from_bbox_counts_only_the_box_on_the_grid(
        self, projection, nx, ny, bbox, expected
    ):
        """Points of the box off the grid do not widen the window (#139 review)."""
        grid = Grid(projection=projection, nx=nx, ny=ny)
        window = grid.window_from_bbox(bbox)
        assert (
            window.x_start,
            window.x_stop,
            window.y_start,
            window.y_stop,
        ) == expected
        assert window == _reference_window(grid, bbox)

    @pytest.mark.parametrize(
        "bbox",
        [
            (0, 20, 170, 60),  # the other side of the globe, across the cut
            (-60.114, 39.66, -46.543, 56.634),  # east of the grid
            (-72.524, 20.47, -71.557, 21.001),  # off the south-east corner
        ],
    )
    def test_window_from_bbox_box_off_the_grid_raises(self, bbox):
        """
        A box with no point on the grid's cells raises (#139 review).

        ``main`` gave the last box a window of 2 x 3 cells from its clipped
        corners, though none of its points is on the grid.
        """
        grid = Grid(projection=_HRRR, nx=1799, ny=1059)
        assert _reference_window(grid, bbox) is None
        with pytest.raises(ValueError, match="does not intersect"):
            grid.window_from_bbox(bbox)

    def test_subset_updates_lower_left_sync_point(self):
        proj = Projection(
            pole_lat=90.0,
            pole_lon=180.0,
            tangent_lat=1.0,
            tangent_lon=1.0,
            grid_size=0.0,
            orientation=0.0,
            cone_angle=0.0,
            sync_x=1.0,
            sync_y=1.0,
            sync_lat=-10.0,
            sync_lon=20.0,
        )
        grid = Grid(projection=proj, nx=20, ny=20)
        subset = grid.subset(GridWindow(x_start=3, x_stop=6, y_start=3, y_stop=5))

        assert subset.nx == 3
        assert subset.ny == 2
        coords = subset.calculate_coords()
        np.testing.assert_allclose(coords["lon"][1][0], 23.0)
        np.testing.assert_allclose(coords["lat"][1][0], -7.0)
        # HYSPLIT reads a lat-lon sync point as the first cell's position
        assert (subset.projection.sync_x, subset.projection.sync_y) == (1.0, 1.0)

    def test_subset_of_projected_grid_keeps_sync_point(self):
        """
        A projected subset's cells sit exactly on the source's cells (#78).

        The sync latitude and longitude stay the source's and only their grid
        position moves, so no corner coordinate is rounded to 7 characters.
        """
        hrrr = Grid(
            projection=Projection(
                pole_lat=90.0,
                pole_lon=0.0,
                tangent_lat=38.5,
                tangent_lon=-97.5,
                grid_size=3.0,
                orientation=0.0,
                cone_angle=38.5,
                sync_x=1.0,
                sync_y=1.0,
                sync_lat=21.1381,
                sync_lon=-122.719,
            ),
            nx=1799,
            ny=1059,
        )
        window = GridWindow(x_start=5, x_stop=25, y_start=327, y_stop=342)
        subset = hrrr.subset(window)

        p = subset.projection
        assert (p.sync_lat, p.sync_lon) == (21.1381, -122.719)
        assert (p.sync_x, p.sync_y) == (-4.0, -326.0)
        assert replace(p, sync_x=1.0, sync_y=1.0) == hrrr.projection
        source = hrrr.calculate_coords()
        coords = subset.calculate_coords()
        for name in ("lon", "lat"):
            np.testing.assert_allclose(
                coords[name][1],
                source[name][1][window.y_slice, window.x_slice],
                rtol=0,
                atol=1e-9,
            )

        # a subset of the subset is the same as subsetting the source directly
        inner = subset.subset(GridWindow(x_start=3, x_stop=8, y_start=2, y_stop=6))
        direct = hrrr.subset(GridWindow(x_start=8, x_stop=13, y_start=329, y_stop=333))
        assert inner == direct

    @staticmethod
    def _global_grid(sync_lon: float, pole_lon: float) -> Grid:
        """A global 1-degree grid, as GDAS (0 E) or one starting at 180 W."""
        return Grid(
            projection=Projection(
                pole_lat=90.0,
                pole_lon=pole_lon,
                tangent_lat=1.0,
                tangent_lon=1.0,
                grid_size=0.0,
                orientation=0.0,
                cone_angle=0.0,
                sync_x=1.0,
                sync_y=1.0,
                sync_lat=-90.0,
                sync_lon=sync_lon,
            ),
            nx=360,
            ny=181,
        )

    @pytest.mark.parametrize(
        ("source", "bbox", "corners"),
        [
            # as xtrct_grid writes a crop: sync_lon in 0-360, pole_lon its last cell
            ((0.0, 359.0), (-125, 30, -100, 50), (30.0, 235.0, 50.0, 260.0)),
            ((0.0, 359.0), (100, -10, 120, 10), (-10.0, 100.0, 10.0, 120.0)),
            ((-180.0, 179.0), (-10, 40, 10, 60), (40.0, 350.0, 60.0, 370.0)),
            ((-180.0, 179.0), (-125, 30, -100, 50), (30.0, 235.0, 50.0, 260.0)),
            ((-180.0, 179.0), (100, -10, 120, 10), (-10.0, 100.0, 10.0, 120.0)),
        ],
    )
    def test_latlon_subset_writes_hysplit_longitudes(self, source, bbox, corners):
        """
        A lat/lon subset's sync point is its first cell, its pole its last.

        Its longitudes are ones HYSPLIT's gbl2xy places correctly (#82).
        """
        grid = self._global_grid(*source)
        subset = grid.subset(grid.window_from_bbox(bbox))

        p = subset.projection
        assert (p.sync_lat, p.sync_lon, p.pole_lat, p.pole_lon) == corners
        assert (p.sync_x, p.sync_y) == (1.0, 1.0)
        coords = subset.calculate_coords()
        assert coords["lon"][1][0] == pytest.approx(bbox[0])
        assert coords["lon"][1][-1] == pytest.approx(bbox[2])
        assert coords["lat"][1][0] == pytest.approx(bbox[1])

    def test_latlon_subset_of_subset_stays_0_360(self):
        """A crop of a crop is written as xtrct_grid writes it (#82)."""
        grid = self._global_grid(0.0, 359.0)
        crop = grid.subset(grid.window_from_bbox((-125, 30, -100, 50)))
        inner = crop.subset(crop.window_from_bbox((-120, 35, -105, 45)))
        p = inner.projection
        assert (p.sync_lat, p.sync_lon, p.pole_lat, p.pole_lon) == (
            35.0,
            240.0,
            45.0,
            255.0,
        )
        assert inner == grid.subset(grid.window_from_bbox((-120, 35, -105, 45)))

    @pytest.mark.parametrize(
        ("source", "bbox", "window", "corners"),
        [
            # across 0 E on a grid from 0 E: columns 350 ... 359, 0 ... 10
            ((0.0, 359.0), (-10, 40, 10, 60), (350, 371), (40.0, 350.0, 60.0, 370.0)),
            # across 180 on a grid from 180 W: columns 350 ... 359, 0 ... 10
            (
                (-180.0, 179.0),
                (170, -10, -170, 10),
                (350, 371),
                (-10.0, 170.0, 10.0, 190.0),
            ),
            # the same boxes on the other grid are inside it
            (
                (0.0, 359.0),
                (170, -10, -170, 10),
                (170, 191),
                (-10.0, 170.0, 10.0, 190.0),
            ),
            (
                (-180.0, 179.0),
                (-10, 40, 10, 60),
                (170, 191),
                (40.0, 350.0, 60.0, 370.0),
            ),
        ],
    )
    def test_latlon_window_across_the_seam(self, source, bbox, window, corners):
        """A bbox across a global grid's seam keeps only its own columns (#83)."""
        grid = self._global_grid(*source)
        win = grid.window_from_bbox(bbox)
        assert (win.x_start, win.x_stop, win.nx, win.ny) == (*window, 21, 21)

        subset = grid.subset(win)
        p = subset.projection
        assert (p.sync_lat, p.sync_lon, p.pole_lat, p.pole_lon) == corners
        lons = subset.calculate_coords()["lon"][1]
        np.testing.assert_allclose(np.diff(lons), 1.0)
        assert lons[0] == pytest.approx(bbox[0])
        x, _ = subset.fractional_indices(np.array([bbox[0], bbox[2]]), 0.0)
        np.testing.assert_allclose(x, [0.0, 20.0])

    def test_latlon_bbox_at_both_ends_of_a_regional_grid_raises(self):
        """Columns at both ends of a grid short of 360 degrees are not one grid (#83)."""
        grid = Grid(
            projection=replace(
                self._global_grid(100.0, 299.0).projection, sync_lat=0.0
            ),
            nx=200,
            ny=20,
        )
        with pytest.raises(ValueError, match="both ends"):
            grid.window_from_bbox((-80, 0, 110, 10))

    @staticmethod
    def _latlon_grid(sync_lat, sync_lon, spacing, nx, ny) -> Grid:
        """A lat/lon grid from its first cell, as HYSPLIT writes one."""
        return Grid(
            projection=Projection(
                pole_lat=sync_lat + (ny - 1) * spacing,
                pole_lon=sync_lon + (nx - 1) * spacing,
                tangent_lat=spacing,
                tangent_lon=spacing,
                grid_size=0.0,
                orientation=0.0,
                cone_angle=0.0,
                sync_x=1.0,
                sync_y=1.0,
                sync_lat=sync_lat,
                sync_lon=sync_lon,
            ),
            nx=nx,
            ny=ny,
        )

    @pytest.mark.parametrize("sync_lon", [0.0, -180.0], ids=["from_0E", "from_180W"])
    @pytest.mark.parametrize(
        ("bbox", "west", "n_cols"),
        [
            # an east edge at 180 keeps the 180 column, a west edge at -180 too (#135)
            ((170, 0, 180, 10), 170.0, 41),
            ((-180, 0, -170, 10), 180.0, 41),
            ((179.75, 0, 180, 10), 179.75, 2),
            # a box past 180 is wrapped, not cut at 180
            ((170, 0, 190, 10), 170.0, 81),
            ((170, 0, -170, 10), 170.0, 81),
            ((190, 0, 200, 10), 190.0, 41),
            ((-200, 0, -190, 10), 160.0, 41),
            ((350, 0, 370, 10), 350.0, 81),
            ((246, 0, 249, 10), 246.0, 13),
            # unchanged: a box across 0 E
            ((-10, 0, 10, 10), 350.0, 81),
        ],
    )
    def test_latlon_bbox_edges_at_and_past_180_on_a_global_grid(
        self, sync_lon, bbox, west, n_cols
    ):
        """GFS 0.25 degree layout: 1440 x 721, from 0 E or from 180 W (#135)."""
        grid = self._latlon_grid(-90.0, sync_lon, 0.25, 1440, 721)
        win = grid.window_from_bbox(bbox)
        assert win.nx == n_cols
        lons = grid.subset(win).calculate_coords()["lon"][1]
        assert lons[0] % 360 == pytest.approx(west % 360)
        np.testing.assert_allclose(np.diff(lons), 0.25)

    @pytest.mark.parametrize("sync_lon", [0.0, -180.0], ids=["from_0E", "from_180W"])
    def test_latlon_bbox_with_equivalent_longitudes_selects_the_same_columns(
        self, sync_lon
    ):
        grid = self._latlon_grid(-90.0, sync_lon, 0.25, 1440, 721)
        boxes = [(170, 0, 190, 10), (170, 0, -170, 10), (-190, 0, -170, 10)]
        windows = {grid.window_from_bbox(box) for box in boxes}
        assert len(windows) == 1
        assert (
            grid.window_from_bbox((190, 0, 200, 10)).nx
            == grid.window_from_bbox((-170, 0, -160, 10)).nx
        )

    @pytest.mark.parametrize("sync_lon", [0.0, -180.0], ids=["from_0E", "from_180W"])
    def test_latlon_bbox_of_all_longitudes_keeps_every_column(self, sync_lon):
        grid = self._latlon_grid(-90.0, sync_lon, 0.25, 1440, 721)
        for bbox in [(-180, 0, 180, 10), (0, 0, 360, 10), (-90, 0, 270, 10)]:
            assert grid.window_from_bbox(bbox).nx == 1440

    def test_latlon_bbox_at_and_past_180_on_a_regional_grid(self):
        """A grid stored from 160 E to 200 E (81 x 41, half a degree) (#135)."""
        grid = self._latlon_grid(20.0, 160.0, 0.5, 81, 41)
        for bbox in [(170, 25, 190, 30), (170, 25, -170, 30), (-190, 25, -170, 30)]:
            win = grid.window_from_bbox(bbox)
            assert (win.x_start, win.x_stop) == (20, 61)
        assert grid.window_from_bbox((170, 25, 180, 30)).nx == 21
        assert grid.window_from_bbox((-180, 25, -170, 30)).nx == 21
        assert grid.window_from_bbox((-175, 25, -165, 30)).nx == 21

    def test_latlon_bbox_past_180_outside_a_regional_grid_raises(self):
        grid = self._latlon_grid(20.0, 160.0, 0.5, 81, 41)
        with pytest.raises(ValueError, match="does not intersect"):
            grid.window_from_bbox((-150, 25, -140, 30))

    @pytest.mark.parametrize("sync_lon", [0.0, -180.0], ids=["from_0E", "from_180W"])
    @pytest.mark.parametrize(
        ("bbox", "first", "last"),
        [
            # An east edge on a column keeps it, whichever way the arithmetic
            # rounds: (250 - -112.3) % 360 is 2.3000000000000114 but
            # -110 - -112.3 is 2.299999999999997 (#138 review).
            ((-112.3, 39, -110, 42), -112.25, -110.0),
            ((-105.3, 39, -104, 41), -105.25, -104.0),
            ((-80.6, 25, -80, 27), -80.5, -80.0),
            ((-114.1, 30, -100, 45), -114.0, -100.0),
            # A west edge on a column keeps it.
            ((-112.25, 39, -110.3, 42), -112.25, -110.5),
        ],
    )
    def test_latlon_bbox_keeps_a_column_on_its_edge(self, sync_lon, bbox, first, last):
        """GFS 0.25 degree layout: an edge typed to two decimals meets a column."""
        grid = self._latlon_grid(-90.0, sync_lon, 0.25, 1440, 721)
        win = grid.window_from_bbox(bbox)
        lons = grid.subset(win).calculate_coords()["lon"][1]
        assert lons[0] % 360 == pytest.approx(first % 360)
        assert lons[-1] % 360 == pytest.approx(last % 360)

    def test_latlon_bbox_keeps_a_column_on_its_edge_on_a_one_degree_grid(self):
        """GDAS 1 degree layout (360 x 181 from 0 E): -80.6 to -80 holds -80."""
        grid = self._latlon_grid(-90.0, 0.0, 1.0, 360, 181)
        win = grid.window_from_bbox((-80.6, 25, -80, 27))
        assert (win.x_start, win.x_stop) == (280, 281)
        win = grid.window_from_bbox((-112.3, 39, -110, 42))
        assert (win.x_start, win.x_stop) == (248, 251)

    @pytest.mark.parametrize("sync_lon", [0.0, -180.0], ids=["from_0E", "from_180W"])
    def test_latlon_bbox_edge_within_rounding_of_a_column_is_on_it(self, sync_lon):
        """An edge a few ulps off a column counts as on it, on either side."""
        grid = self._latlon_grid(-90.0, sync_lon, 0.25, 1440, 721)
        on = grid.window_from_bbox((-2.0, 26.7, 163.5, 71.7))
        for west in (np.nextafter(-2.0, 0), np.nextafter(-2.0, -9)):
            assert grid.window_from_bbox((west, 26.7, 163.5, 71.7)) == on
        for east in (np.nextafter(163.5, 0), np.nextafter(163.5, 999)):
            assert grid.window_from_bbox((-2.0, 26.7, east, 71.7)) == on
        assert on.nx == 663

    def test_latlon_bbox_edge_within_rounding_of_a_column_on_a_regional_grid(self):
        """A west edge a few ulps from the grid's last column counts as on it."""
        grid = self._latlon_grid(20.0, 160.0, 0.5, 81, 41)
        # -160 is the grid's last column (200 E): the box has columns at both ends,
        # with the edge on the column or an ulp to either side of it.
        for west in (-160.0, np.nextafter(-160.0, 0), np.nextafter(-160.0, -999)):
            with pytest.raises(ValueError, match="both ends"):
                grid.window_from_bbox((west, 21.5, 175.5, 34.5))
        win = grid.window_from_bbox((-159.75, 21.5, 175.5, 34.5))
        assert (win.x_start, win.x_stop) == (0, 32)

    @pytest.mark.parametrize("sync_lon", [0.0, -180.0], ids=["from_0E", "from_180W"])
    def test_latlon_bbox_column_clearly_outside_an_edge_stays_out(self, sync_lon):
        """The tolerance is rounding only: 1e-4 degrees outside is outside, on either edge."""
        grid = self._latlon_grid(-90.0, sync_lon, 0.25, 1440, 721)
        edge = grid.window_from_bbox((-112.3, 39, -110, 42))
        east_out = grid.window_from_bbox((-112.3, 39, -110.0001, 42))
        west_out = grid.window_from_bbox((-112.2499, 39, -110, 42))
        assert (east_out.x_start, east_out.nx) == (edge.x_start, edge.nx - 1)
        assert (west_out.x_start, west_out.nx) == (edge.x_start + 1, edge.nx - 1)

    @pytest.mark.parametrize("sync_lon", [0.0, -180.0], ids=["from_0E", "from_180W"])
    @pytest.mark.parametrize("side", ["east", "west"])
    @pytest.mark.parametrize(
        ("spacing", "nx", "ny"),
        [(0.25, 1440, 721), (1.0, 360, 181)],
        ids=["gfs", "gdas"],
    )
    def test_latlon_bbox_tolerance_is_a_millionth_of_a_cell(
        self, sync_lon, side, spacing, nx, ny
    ):
        """An edge 0.99e-6 cells past a column keeps it; 1.01e-6 cells past drops it."""
        grid = self._latlon_grid(-90.0, sync_lon, spacing, nx, ny)
        column = -110.0  # a column of both layouts (250 E on a grid from 0 E)
        tol = 1e-6 * spacing
        for cells, kept in [(0.99, True), (1.01, False)]:
            if side == "east":  # the edge is west of the column
                bbox = (column - 3 * spacing, 39, column - cells * tol, 42)
                expected = column if kept else column - spacing
            else:  # the edge is east of the column
                bbox = (column + cells * tol, 39, column + 3 * spacing, 42)
                expected = column if kept else column + spacing
            win = grid.window_from_bbox(bbox)
            lons = grid.subset(win).calculate_coords()["lon"][1]
            edge = lons[-1] if side == "east" else lons[0]
            assert edge % 360 == pytest.approx(expected % 360), (cells, bbox)

    def test_latlon_bbox_keeps_a_row_on_its_north_edge_on_a_tenth_degree_grid(self):
        """The grid of #147: 41.4 north of 25 is stored as 41.400000000000006."""
        grid = self._latlon_grid(25.0, -125.0, 0.1, 650, 300)
        assert grid.calculate_coords()["lat"][1][164] > 41.4
        # rows 0 to 164 (25.0 to 41.4), then 165 (41.5)
        assert grid.window_from_bbox((-120.0, 25.0, -110.0, 41.4)).y_stop == 165
        assert grid.window_from_bbox((-120.0, 25.0, -110.0, 41.5)).y_stop == 166

    @pytest.mark.parametrize("side", ["south", "north"])
    @pytest.mark.parametrize(
        ("spacing", "sync_lat"),
        [
            (0.1, 25.0),
            (0.1, -25.0),
            (0.05, -25.0),
            (0.05, 0.0),
            (0.3, -25.0),
            (0.3, 25.0),
        ],
    )
    def test_latlon_bbox_keeps_every_row_on_its_edge_when_the_spacing_is_inexact(
        self, spacing, sync_lat, side
    ):
        """Each interior row, edge typed as the row's latitude, alone (#147)."""
        ny = 300
        grid = self._latlon_grid(sync_lat, -125.0, spacing, 650, ny)
        for j in range(2, ny - 2):
            edge = round(sync_lat + j * spacing, 10)  # the latitude as a user types it
            if side == "south":  # the row is the box's southernmost
                bbox = (-120.0, edge, -110.0, edge + 0.4 * spacing)
            else:  # the row is the box's northernmost
                bbox = (-120.0, edge - 0.4 * spacing, -110.0, edge)
            win = grid.window_from_bbox(bbox)
            assert (win.y_start, win.y_stop) == (j, j + 1), (j, bbox)

    @pytest.mark.parametrize("spacing", [0.1, 0.05, 0.3])
    def test_latlon_bbox_row_clearly_outside_an_edge_stays_out(self, spacing):
        """The tolerance is rounding only: 1e-4 degrees past the row's latitude is past it."""
        grid = self._latlon_grid(-25.0, -125.0, spacing, 650, 300)
        j = 120
        edge = round(-25.0 + j * spacing, 10)
        # a south edge just north of row j drops it; a north edge just south drops it
        south_out = grid.window_from_bbox(
            (-120, edge + 1e-4, -110, edge + 1.5 * spacing)
        )
        north_out = grid.window_from_bbox(
            (-120, edge - 1.5 * spacing, -110, edge - 1e-4)
        )
        assert (south_out.y_start, south_out.y_stop) == (j + 1, j + 2)
        assert (north_out.y_start, north_out.y_stop) == (j - 1, j)
        # and a box with nothing else in it raises
        with pytest.raises(ValueError, match="does not intersect"):
            grid.window_from_bbox((-120, edge + 1e-4, -110, edge + 0.4 * spacing))

    @pytest.mark.parametrize("side", ["south", "north"])
    @pytest.mark.parametrize("spacing", [0.1, 0.3])
    def test_latlon_bbox_row_tolerance_is_a_millionth_of_a_cell(self, spacing, side):
        """An edge 0.99e-6 cells past a row keeps it; 1.01e-6 cells past drops it."""
        grid = self._latlon_grid(-25.0, -125.0, spacing, 650, 300)
        j = 120
        row = round(-25.0 + j * spacing, 10)
        tol = 1e-6 * spacing
        for cells, kept in [(0.99, True), (1.01, False)]:
            if side == "south":  # the edge is north of the row
                bbox = (-120, row + cells * tol, -110, row + 3.5 * spacing)
                first = j if kept else j + 1
                expected = (first, j + 4)
            else:  # the edge is south of the row
                bbox = (-120, row - 3.5 * spacing, -110, row - cells * tol)
                last = j if kept else j - 1
                expected = (j - 3, last + 1)
            win = grid.window_from_bbox(bbox)
            assert (win.y_start, win.y_stop) == expected, (cells, bbox)

    @pytest.mark.parametrize(
        "kind", [np.float32, np.float16, np.int32, np.float64, Decimal]
    )
    def test_latlon_bbox_edges_of_any_numeric_type_equal_those_as_floats(self, kind):
        """A box of NumPy scalars or Decimals is its values as Python floats (#138 review)."""
        grid = self._latlon_grid(-90.0, 0.0, 1.0, 360, 181)
        for bbox in [(13, -70, 60, 70), (-112, 39, -110, 42), (170, 0, 190, 10)]:
            window = grid.window_from_bbox(tuple(kind(v) for v in bbox))
            assert window == grid.window_from_bbox(bbox)

    @pytest.mark.parametrize(
        ("sync_lon", "bbox", "first", "last"),
        [
            (0.0, ("13.04", "-70", "60", "70"), 13.25, 60.0),
            (0.0, ("9.1", "0", "159", "10"), 9.25, 159.0),
            (-180.0, ("-26.85", "0", "78", "10"), -26.75, 78.0),
            (0.0, ("-65.65", "0", "81", "10"), -65.5, 81.0),
        ],
    )
    @pytest.mark.parametrize("kind", [np.float32, Decimal])
    def test_latlon_bbox_float32_and_decimal_edges_keep_the_east_edge_column(
        self, sync_lon, bbox, first, last, kind
    ):
        """GFS 0.25 layout. East edges typed to whole degrees, west to two decimals."""
        grid = self._latlon_grid(-90.0, sync_lon, 0.25, 1440, 721)
        # (13.04 and the like are not on a 0.25 column, so ``first`` is the next one.)
        win = grid.window_from_bbox(tuple(kind(v) for v in bbox))
        lons = grid.subset(win).calculate_coords()["lon"][1]
        assert lons[0] % 360 == pytest.approx(first % 360)
        assert lons[-1] % 360 == pytest.approx(last % 360)

    def test_latlon_bbox_float32_edges_on_a_one_degree_grid(self):
        """Real GDAS 1: float32 (13.04, -70, 60, 70) is 47 columns, 14 to 60, as floats."""
        grid = self._latlon_grid(-90.0, 0.0, 1.0, 360, 181)
        box = tuple(np.array([13.04, -70.0, 60.0, 70.0], dtype=np.float32))
        win = grid.window_from_bbox(box)
        assert (win.x_start, win.x_stop) == (14, 61)

    def test_latlon_bbox_of_infinite_width_holds_every_column(self):
        """An infinite edge is a box 360 degrees wide or more; no warning (#138 review)."""
        grid = self._latlon_grid(-90.0, 0.0, 1.0, 360, 181)
        inf = float("inf")
        for bbox in [(-inf, -90, inf, 90), (-inf, 0, 10, 10), (10, 0, inf, 10)]:
            win = grid.window_from_bbox(bbox)
            assert (win.x_start, win.x_stop) == (0, 360), bbox

    def test_latlon_bbox_with_a_nan_or_unmatched_infinite_edge_raises(self):
        grid = self._latlon_grid(-90.0, 0.0, 1.0, 360, 181)
        inf, nan = float("inf"), float("nan")
        for bbox in [
            (10, 0, nan, 10),
            (nan, 0, 10, 10),
            (inf, 0, -inf, 10),
            (inf, 0, inf, 10),
        ]:
            with pytest.raises(ValueError, match="longitudes"):
                grid.window_from_bbox(bbox)

    @pytest.mark.parametrize("source", [(0.0, 359.0), (-180.0, 179.0)])
    def test_latlon_subset_of_whole_grid_is_unchanged(self, source):
        grid = self._global_grid(*source)
        assert grid.subset(grid.full_window()) is grid


def _crs_reference_window(grid: Grid, crs, bbox, n: int = 801) -> GridWindow | None:
    """
    The window of the part of *bbox*, in *crs*, on *grid*, by brute force, or None.

    An *n* by *n* lattice of the box's points, those inside the grid's cells
    rounded with NINT. The lattice falls inside the box's true extremes, so the
    window can be a cell smaller than the exact one, never larger.
    """
    west, south, east, north = bbox
    bx, by = np.meshgrid(np.linspace(west, east, n), np.linspace(south, north, n))
    to_lonlat = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = to_lonlat.transform(bx.ravel(), by.ravel())
    x, y = grid.fractional_indices(np.asarray(lon), np.asarray(lat))
    if grid.is_latlon:
        # fractional_indices counts east of the first column, in [0, 360): put
        # the points west of the grid west of it, not far east.
        turn = 360.0 / grid.projection.tangent_lon
        x = np.where(x > grid.nx - 0.5 + (turn - grid.nx) / 2, x - turn, x)
    on = np.isfinite(x) & np.isfinite(y)
    on &= (x >= -0.5) & (x <= grid.nx - 0.5) & (y >= -0.5) & (y <= grid.ny - 0.5)
    x, y = x[on], y[on]
    if not x.size:
        return None
    return GridWindow(
        x_start=max(0, _nint(x.min() + 1.0) - 1),
        x_stop=min(grid.nx, _nint(x.max() + 1.0)),
        y_start=max(0, _nint(y.min() + 1.0) - 1),
        y_stop=min(grid.ny, _nint(y.max() + 1.0)),
    )


def _latlon_grid(sync_lat, sync_lon, spacing, nx, ny) -> Grid:
    """A lat/lon grid from its first cell, as HYSPLIT writes one."""
    return TestGrid._latlon_grid(sync_lat, sync_lon, spacing, nx, ny)


class TestWindowFromBboxCrs:
    """``Grid.window_from_bbox(bbox, bbox_crs=...)``: a box in another CRS (#21)."""

    @pytest.mark.parametrize(
        ("grid", "crs", "bbox"),
        [
            pytest.param(
                Grid(projection=_HRRR, nx=1799, ny=1059),
                "EPSG:32612",
                (380_000, 4_400_000, 460_000, 4_560_000),
                id="hrrr-utm12",
            ),
            pytest.param(
                Grid(projection=_HRRR, nx=1799, ny=1059),
                "EPSG:4269",
                (-113.5, 39.5, -110.5, 42.0),
                id="hrrr-nad83",
            ),
            pytest.param(
                Grid(projection=_HRRR, nx=1799, ny=1059),
                "EPSG:5070",
                (-2_500_000, 2_500_000, -1_000_000, 4_000_000),
                id="hrrr-albers-past-the-north-west-corner",
            ),
            pytest.param(
                Grid(projection=_NAMS_AK, nx=387, ny=276),
                "EPSG:3338",
                (-600_000, 600_000, 400_000, 1_600_000),
                id="polar-stereographic-alaska-albers",
            ),
            pytest.param(
                Grid(projection=_NAMS_HI, nx=321, ny=225),
                "EPSG:32604",
                (500_000, 2_200_000, 700_000, 2_400_000),
                id="mercator-utm4",
            ),
            pytest.param(
                _latlon_grid(30.0, -120.0, 0.25, 100, 60),
                "EPSG:32611",
                (200_000, 3_500_000, 400_000, 3_700_000),
                id="latlon-over-the-west-border",
            ),
            pytest.param(
                _latlon_grid(-90.0, 0.0, 1.0, 360, 181),
                "EPSG:32612",
                (300_000, 4_500_000, 700_000, 4_900_000),
                id="global-latlon-utm12",
            ),
        ],
    )
    def test_window_matches_a_dense_lattice_of_the_box(self, grid, crs, bbox):
        """The window holds every lattice point of the box on the grid, a cell at most more."""
        window = grid.window_from_bbox(bbox, bbox_crs=crs)
        reference = _crs_reference_window(grid, crs, bbox)
        assert reference is not None
        assert (
            window.x_start <= reference.x_start and window.y_start <= reference.y_start
        )
        assert window.x_stop >= reference.x_stop and window.y_stop >= reference.y_stop
        _assert_close_windows(window, reference)

    def test_hrrr_utm_box(self):
        """A UTM 12N box over the Salt Lake Valley on the HRRR grid."""
        grid = Grid(projection=_HRRR, nx=1799, ny=1059)
        window = grid.window_from_bbox(
            (380_000, 4_400_000, 460_000, 4_560_000), bbox_crs="EPSG:32612"
        )
        assert window == GridWindow(x_start=476, x_stop=511, y_start=606, y_stop=663)

    def test_grid_crs_is_meters_from_the_first_cell(self):
        """With the grid's own CRS, cell (i, j) is at (i, j) * grid_size meters."""
        grid = Grid(projection=_HRRR, nx=1799, ny=1059)
        step = _HRRR.grid_size * 1000.0
        bbox = (10 * step, 20 * step, 13 * step, 22 * step)
        window = grid.window_from_bbox(bbox, bbox_crs=grid.crs)
        assert window == GridWindow(x_start=10, x_stop=14, y_start=20, y_stop=23)

    @pytest.mark.parametrize("crs", ["EPSG:4326", "OGC:CRS84", 4326, pyproj.CRS(4326)])
    @pytest.mark.parametrize(
        ("grid", "bbox"),
        [
            pytest.param(
                Grid(projection=_HRRR, nx=1799, ny=1059),
                (-113.5, 39.5, -110.5, 42.0),
                id="hrrr",
            ),
            pytest.param(
                _latlon_grid(-90.0, 0.0, 1.0, 360, 181),
                (170.0, 0.0, -170.0, 10.0),
                id="global-across-the-dateline",
            ),
        ],
    )
    def test_epsg_4326_is_the_default(self, grid, bbox, crs):
        """EPSG:4326 in either axis order keeps the degree rules (dateline included)."""
        assert grid.window_from_bbox(bbox, bbox_crs=crs) == grid.window_from_bbox(bbox)

    def test_box_larger_than_the_grid_holds_the_grid(self):
        grid = Grid(projection=_HRRR, nx=1799, ny=1059)
        window = grid.window_from_bbox((-5e6, -2e6, 5e6, 5e6), bbox_crs="EPSG:5070")
        assert window == grid.full_window()

    def test_box_across_a_global_grids_seam_gets_every_column(self):
        """A global lat/lon grid is cut at its seam (here 0 E): no wrapped window."""
        grid = _latlon_grid(-90.0, 0.0, 1.0, 360, 181)
        # UTM 31N from about 1.7 W to 5.4 E at 40-44 N.
        window = grid.window_from_bbox(
            (100_000, 4_500_000, 700_000, 4_900_000), bbox_crs="EPSG:32631"
        )
        assert (window.x_start, window.x_stop) == (0, 360)
        assert (window.y_start, window.y_stop) == (131, 135)

    @pytest.mark.parametrize(
        ("bbox", "match"),
        [
            ((460_000, 4_400_000, 380_000, 4_560_000), "west must be"),
            ((380_000, 4_560_000, 460_000, 4_400_000), "south must be"),
            ((380_000, 4_400_000, float("nan"), 4_560_000), "finite"),
            ((380_000, 4_400_000, float("inf"), 4_560_000), "finite"),
            ((380_000, 1_000_000, 460_000, 1_100_000), "does not intersect"),
        ],
    )
    def test_bad_boxes_raise(self, bbox, match):
        grid = Grid(projection=_HRRR, nx=1799, ny=1059)
        with pytest.raises(ValueError, match=match):
            grid.window_from_bbox(bbox, bbox_crs="EPSG:32612")

    def test_unknown_crs_raises(self):
        grid = Grid(projection=_HRRR, nx=1799, ny=1059)
        with pytest.raises(pyproj.exceptions.CRSError):
            grid.window_from_bbox((0, 0, 1, 1), bbox_crs="EPSG:0")


class TestVerticalAxis:
    """Tests for VerticalAxis class."""

    def test_sigma_coordinate(self):
        """Test sigma vertical coordinate."""
        levels = [1.0, 0.9, 0.8, 0.7]
        axis = SigmaAxis(levels=levels)
        assert axis.coord_system == "sigma"
        np.testing.assert_array_equal(axis.levels, levels)

    def test_pressure_coordinate(self):
        """Test pressure vertical coordinate."""
        levels = [1000, 925, 850, 700, 500]
        axis = PressureAxis(levels=levels)
        assert axis.coord_system == "pressure"
        np.testing.assert_array_equal(axis.levels, levels)

    def test_terrain_coordinate(self):
        """Test terrain vertical coordinate."""
        levels = [0.0, 0.25, 0.5, 0.75, 1.0]
        axis = TerrainAxis(levels=levels)
        assert axis.coord_system == "terrain"
        np.testing.assert_array_equal(axis.levels, levels)

    def test_hybrid_coordinate(self):
        """Test hybrid vertical coordinate."""
        levels = [1000.0, 925.5, 850.0]
        axis = HybridAxis(levels=levels)
        assert axis.coord_system == "hybrid"
        np.testing.assert_array_equal(axis.levels, levels)

    def test_unknown_flag_raises(self):
        """Test unknown vertical flag raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported vertical flag 99"):
            VerticalAxis.from_flag(99, levels=[1.0, 0.5])


class TestReprs:
    def _latlon_proj(self):
        return Projection(
            pole_lat=90.0,
            pole_lon=0.0,
            tangent_lat=1.0,
            tangent_lon=1.0,
            grid_size=0.0,
            orientation=0.0,
            cone_angle=0.0,
            sync_x=1.0,
            sync_y=1.0,
            sync_lat=-10.0,
            sync_lon=20.0,
        )

    def _stere_proj(self):
        return Projection(
            pole_lat=90.0,
            pole_lon=0.0,
            tangent_lat=60.0,
            tangent_lon=0.0,
            grid_size=50.0,
            orientation=0.0,
            cone_angle=90.0,
            sync_x=1.0,
            sync_y=1.0,
            sync_lat=60.0,
            sync_lon=0.0,
        )

    def test_projection_repr_latlon(self):
        assert repr(self._latlon_proj()) == "Projection(latlon)"

    def test_projection_repr_stere(self):
        assert repr(self._stere_proj()) == "Projection(stere)"

    def test_grid_repr_latlon(self):
        grid = Grid(projection=self._latlon_proj(), nx=144, ny=73)
        assert repr(grid) == "Grid(latlon, 144\u00d773)"

    def test_grid_repr_projected_includes_spacing(self):
        grid = Grid(projection=self._stere_proj(), nx=100, ny=80)
        assert repr(grid) == "Grid(stere 50km, 100\u00d780)"


class TestWindRotation:
    def _lambert(self) -> Grid:
        proj = Projection(
            pole_lat=90.0,
            pole_lon=0.0,
            tangent_lat=38.5,
            tangent_lon=-97.5,
            grid_size=3.0,
            orientation=0.0,
            cone_angle=38.5,
            sync_x=1.0,
            sync_y=1.0,
            sync_lat=21.138,
            sync_lon=-122.72,
        )
        return Grid(projection=proj, nx=1799, ny=1059)

    def test_lambert_convergence_matches_cone_constant(self):
        grid = self._lambert()
        lon = np.array([-111.9, -97.5, -80.0, -120.0])
        lat = np.array([40.7, 30.0, 45.0, 25.0])
        expected = np.sin(np.radians(38.5)) * (lon + 97.5)
        np.testing.assert_allclose(
            grid.meridian_convergence(lon, lat), expected, atol=1e-6
        )

    def test_lambert_rotation_matches_hrrr_formula(self):
        grid = self._lambert()
        lon = np.array([-111.9, -80.0])
        lat = np.array([40.7, 45.0])
        u = np.array([3.0, -2.0])
        v = np.array([4.0, 1.0])
        angle = np.radians(0.622515 * (lon + 97.5))  # NOAA HRRR FAQ constants
        expected_u = np.cos(angle) * u + np.sin(angle) * v
        expected_v = -np.sin(angle) * u + np.cos(angle) * v
        u_out, v_out = grid.rotate_winds(u, v, lon, lat)
        np.testing.assert_allclose(u_out, expected_u, atol=1e-6)
        np.testing.assert_allclose(v_out, expected_v, atol=1e-6)

    def test_rotation_broadcasts_scalars(self):
        grid = self._lambert()
        u_out, v_out = grid.rotate_winds(1.0, 0.0, [-111.9, -80.0], 40.0)
        assert u_out.shape == (2,)
        np.testing.assert_allclose(np.hypot(u_out, v_out), 1.0, atol=1e-12)

    def test_polar_stereographic_convergence_is_longitude_offset(self):
        proj = Projection(
            pole_lat=90.0,
            pole_lon=0.0,
            tangent_lat=60.0,
            tangent_lon=-105.0,
            grid_size=100.0,
            orientation=0.0,
            cone_angle=90.0,
            sync_x=1.0,
            sync_y=1.0,
            sync_lat=30.0,
            sync_lon=-140.0,
        )
        grid = Grid(projection=proj, nx=50, ny=50)
        lon = np.array([-105.0, -95.0, -125.0])
        lat = np.array([45.0, 50.0, 55.0])
        np.testing.assert_allclose(
            grid.meridian_convergence(lon, lat), lon + 105.0, atol=1e-6
        )

    def test_latlon_grid_is_unchanged(self):
        proj = Projection(
            pole_lat=90.0,
            pole_lon=0.0,
            tangent_lat=1.0,
            tangent_lon=1.0,
            grid_size=0.0,
            orientation=0.0,
            cone_angle=0.0,
            sync_x=1.0,
            sync_y=1.0,
            sync_lat=-10.0,
            sync_lon=20.0,
        )
        grid = Grid(projection=proj, nx=10, ny=10)
        lon = np.array([25.0, 30.0])
        lat = np.array([-5.0, 0.0])
        np.testing.assert_array_equal(grid.meridian_convergence(lon, lat), 0.0)
        u_out, v_out = grid.rotate_winds([1.0, 2.0], [3.0, 4.0], lon, lat)
        np.testing.assert_array_equal(u_out, [1.0, 2.0])
        np.testing.assert_array_equal(v_out, [3.0, 4.0])


class TestRotatedGrids:
    """Grids turned by ``orientation``, as HYSPLIT's stcm1p turns them (#20)."""

    # Each case: the 12 projection fields, the grid size, and grid nodes
    # ``(i, j)`` (0-based) with the latitude and longitude HYSPLIT's
    # stlmbr/stcm1p/cxy2ll give them, set up as metini.f does.
    CASES = {
        # the 2020-02-08 HRRR grid, turned 15 degrees
        "lambert": (
            replace(_HRRR, orientation=15.0),
            (1799, 1059),
            {
                (0, 0): (21.138102, -122.718994),
                (1798, 1058): (33.199802, -60.855217),
                (0, 1058): (49.353226, -123.243607),
                (1798, 0): (10.234965, -76.817383),
            },
        ),
        # a southern polar stereographic grid around its sync point, turned -40
        "south-stereo": (
            Projection(
                pole_lat=-90.0,
                pole_lon=0.0,
                tangent_lat=-60.0,
                tangent_lon=0.0,
                grid_size=20.0,
                orientation=-40.0,
                cone_angle=-90.0,
                sync_x=100.0,
                sync_y=100.0,
                sync_lat=-75.0,
                sync_lon=30.0,
            ),
            (200, 200),
            {
                (0, 0): (-75.316643, 159.415863),
                (199, 199): (-50.246994, 13.852860),
                (0, 199): (-65.457283, -51.356262),
                (199, 0): (-54.904217, 72.598831),
            },
        ),
        # a southern Lambert grid turned 179 degrees: PROJ's inverse alone
        # puts the east of the grid a whole turn of the cone away
        "lambert-turned-179": (
            Projection(
                pole_lat=-90.0,
                pole_lon=0.0,
                tangent_lat=-35.0,
                tangent_lon=140.0,
                grid_size=10.0,
                orientation=179.0,
                cone_angle=-35.0,
                sync_x=50.0,
                sync_y=40.0,
                sync_lat=-35.0,
                sync_lon=140.0,
            ),
            (200, 150),
            {
                (0, 0): (-31.308483, 145.076935),
                (199, 149): (-43.863754, 121.659119),
                (0, 149): (-44.621418, 146.342499),
                (199, 0): (-30.661024, 124.228455),
                (80, 0): (-31.496548, 136.664734),
            },
        ),
    }

    @pytest.mark.parametrize("case", CASES)
    def test_nodes_match_hysplit(self, case):
        """Node positions and their inverse agree with HYSPLIT's map routines."""
        projection, (nx, ny), hysplit = self.CASES[case]
        grid = Grid(projection=projection, nx=nx, ny=ny)
        coords = grid.calculate_coords()
        lat, lon = coords["lat"][1], coords["lon"][1]
        for (i, j), (hlat, hlon) in hysplit.items():
            assert lat[j, i] == pytest.approx(hlat, abs=1e-4)
            assert lon[j, i] == pytest.approx(hlon, abs=1e-4)
            x, y = grid.fractional_indices(hlon, hlat)
            assert float(x) == pytest.approx(i, abs=2e-3)
            assert float(y) == pytest.approx(j, abs=2e-3)

    @pytest.mark.parametrize("case", CASES)
    def test_every_node_maps_back_to_itself(self, case):
        """``fractional_indices`` inverts ``calculate_coords`` on the whole grid."""
        projection, (nx, ny), _ = self.CASES[case]
        grid = Grid(projection=projection, nx=nx, ny=ny)
        coords = grid.calculate_coords()
        x, y = grid.fractional_indices(coords["lon"][1], coords["lat"][1])
        jj, ii = np.mgrid[0:ny, 0:nx]
        np.testing.assert_allclose(x, ii, atol=1e-6)
        np.testing.assert_allclose(y, jj, atol=1e-6)

    @pytest.mark.parametrize("orientation", [15.0, -40.0, 179.0, 350.0])
    @pytest.mark.parametrize("projection", [_HRRR, _NAMS_AK], ids=["lambert", "stereo"])
    def test_y_axis_points_orientation_east_of_north(self, projection, orientation):
        """A wind along the y-axis blows ``orientation`` east of north at ``tangent_lon``."""
        grid = Grid(projection=replace(projection, orientation=orientation), nx=9, ny=9)
        lon, lat = projection.tangent_lon, projection.tangent_lat
        turn = (orientation + 180.0) % 360.0 - 180.0
        assert float(grid.meridian_convergence(lon, lat)) == pytest.approx(turn)
        u, v = grid.rotate_winds(0.0, 1.0, lon, lat)
        bearing = np.degrees(np.arctan2(u, v))
        assert float(bearing) == pytest.approx(turn, abs=1e-6)

    def test_turn_of_a_whole_circle_changes_nothing(self):
        """``orientation`` counts modulo 360."""
        a = Grid(projection=replace(_HRRR, orientation=-10.0), nx=50, ny=40)
        b = Grid(projection=replace(_HRRR, orientation=350.0), nx=50, ny=40)
        for name in ("lon", "lat"):
            np.testing.assert_allclose(
                a.calculate_coords()[name][1], b.calculate_coords()[name][1], atol=1e-9
            )

    def test_latlon_grid_ignores_orientation(self):
        """HYSPLIT reads a lat/lon grid without its orientation, and so does arlmet."""
        fields = dict(
            pole_lat=90.0,
            pole_lon=0.0,
            tangent_lat=0.5,
            tangent_lon=0.5,
            grid_size=0.0,
            cone_angle=0.0,
            sync_x=1.0,
            sync_y=1.0,
            sync_lat=30.0,
            sync_lon=-120.0,
        )
        plain = Grid(projection=Projection(orientation=0.0, **fields), nx=8, ny=6)
        turned = Grid(projection=Projection(orientation=25.0, **fields), nx=8, ny=6)
        for name in ("lon", "lat"):
            np.testing.assert_array_equal(
                plain.calculate_coords()[name][1], turned.calculate_coords()[name][1]
            )

    def test_subset_keeps_source_positions(self):
        """A crop of a rotated grid puts each cell where the source does."""
        grid = Grid(projection=replace(_HRRR, orientation=15.0), nx=300, ny=200)
        window = GridWindow(x_start=40, x_stop=140, y_start=30, y_stop=110)
        sub = grid.subset(window)
        expected, coords = grid.calculate_coords(), sub.calculate_coords()
        for name in ("lon", "lat"):
            np.testing.assert_allclose(
                coords[name][1],
                expected[name][1][window.y_slice, window.x_slice],
                atol=1e-9,
            )


def _lambert_projection(**overrides) -> Projection:
    params = {
        "pole_lat": 90.0,
        "pole_lon": 0.0,
        "tangent_lat": 38.5,
        "tangent_lon": -97.5,
        "grid_size": 3.0,
        "orientation": 0.0,
        "cone_angle": 38.5,
        "sync_x": 1.0,
        "sync_y": 1.0,
        "sync_lat": 21.138,
        "sync_lon": -122.72,
    }
    params.update(overrides)
    return Projection(**params)


class TestImmutability:
    def test_projection_is_frozen(self):
        proj = _lambert_projection()
        with pytest.raises(FrozenInstanceError):
            proj.grid_size = 12.0  # type: ignore[misc]

    def test_projection_params_are_fresh_copies(self):
        proj = _lambert_projection()
        params = proj.params
        params["proj"] = "merc"
        assert proj.params["proj"] == "lcc"

    def test_projection_equality_and_hash_follow_fields(self):
        a = _lambert_projection()
        b = _lambert_projection()
        assert a == b
        assert hash(a) == hash(b)
        assert a != replace(a, sync_lat=22.0)

    def test_grid_is_frozen(self):
        grid = Grid(projection=_lambert_projection(), nx=50, ny=40)
        for name, value in (("nx", 10), ("ny", 10), ("projection", None)):
            with pytest.raises(FrozenInstanceError):
                setattr(grid, name, value)

    def test_grid_hash_is_stable_after_cached_properties(self):
        grid = Grid(projection=_lambert_projection(), nx=50, ny=40)
        before = hash(grid)
        _ = grid.crs, grid.origin
        assert hash(grid) == before
        assert grid == Grid(projection=_lambert_projection(), nx=50, ny=40)
        assert {grid: 1}[Grid(projection=_lambert_projection(), nx=50, ny=40)] == 1

    def test_replace_builds_new_grid_with_fresh_crs(self):
        grid = Grid(projection=_lambert_projection(), nx=50, ny=40)
        crs = grid.crs
        moved = replace(grid, projection=replace(grid.projection, sync_lat=25.0))
        assert moved.crs != crs
        assert grid.crs == crs

    def test_grid_rejects_empty_shape(self):
        with pytest.raises(ValueError, match="positive"):
            Grid(projection=_lambert_projection(), nx=0, ny=10)

    def test_grid_window_is_exported(self):
        import arlmet

        assert arlmet.GridWindow is GridWindow
