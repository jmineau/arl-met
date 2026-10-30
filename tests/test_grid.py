"""Tests for arlmet.grid module."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pyproj
import pytest

from arlmet.grid import Grid, GridWindow, Projection, wrap_lons
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

    def test_window_from_bbox_projected_uses_corner_rounding(self):
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
            sync_lat=30.0,
            sync_lon=-125.0,
        )
        grid = Grid(projection=proj, nx=800, ny=600)
        bbox = (-124.0, 31.0, -120.0, 35.0)

        window = grid.window_from_bbox(bbox)
        x_sw, y_sw = grid.fractional_indices(bbox[0], bbox[1])
        x_ne, y_ne = grid.fractional_indices(bbox[2], bbox[3])

        def nint(value):
            if value >= 0.0:
                return int(np.floor(value + 0.5))
            return int(np.ceil(value - 0.5))

        expected = GridWindow(
            x_start=min(nint(float(x_sw) + 1.0), nint(float(x_ne) + 1.0)) - 1,
            x_stop=max(nint(float(x_sw) + 1.0), nint(float(x_ne) + 1.0)),
            y_start=min(nint(float(y_sw) + 1.0), nint(float(y_ne) + 1.0)) - 1,
            y_stop=max(nint(float(y_sw) + 1.0), nint(float(y_ne) + 1.0)),
        )

        assert window == expected

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
