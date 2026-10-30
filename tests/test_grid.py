"""Tests for arlmet.grid module."""

from dataclasses import FrozenInstanceError, replace

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

    def test_rotated_grid_raises_error(self):
        """Test that rotated grids raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
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

    @pytest.mark.parametrize("source", [(0.0, 359.0), (-180.0, 179.0)])
    def test_latlon_subset_of_whole_grid_is_unchanged(self, source):
        grid = self._global_grid(*source)
        assert grid.subset(grid.full_window()) is grid


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
