"""Tests for arlmet.grid module."""

import numpy as np
import pytest

from arlmet.grid import Grid, GridWindow, Projection, wrap_lons
from arlmet.vertical import VerticalAxis


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
        assert "lon" in grid.coords
        assert "lat" in grid.coords
        assert len(grid.coords["lon"]) == 360
        assert len(grid.coords["lat"]) == 180

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
        assert "x" in grid.coords
        assert "y" in grid.coords
        assert "lon" in grid.coords
        assert "lat" in grid.coords

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
        np.testing.assert_allclose(subset.coords["lon"][0], 23.0)
        np.testing.assert_allclose(subset.coords["lat"][0], -7.0)


class TestVerticalAxis:
    """Tests for VerticalAxis class."""

    def test_sigma_coordinate(self):
        """Test sigma vertical coordinate."""
        levels = [1.0, 0.9, 0.8, 0.7]
        axis = VerticalAxis(flag=1, levels=levels)
        assert axis.coord_system == "sigma"
        np.testing.assert_array_equal(axis.levels, levels)

    def test_pressure_coordinate(self):
        """Test pressure vertical coordinate."""
        levels = [1000, 925, 850, 700, 500]
        axis = VerticalAxis(flag=2, levels=levels)
        assert axis.coord_system == "pressure"
        np.testing.assert_array_equal(axis.levels, levels)

    def test_terrain_coordinate(self):
        """Test terrain vertical coordinate."""
        levels = [0.0, 0.25, 0.5, 0.75, 1.0]
        axis = VerticalAxis(flag=3, levels=levels)
        assert axis.coord_system == "terrain"
        np.testing.assert_array_equal(axis.levels, levels)

    def test_hybrid_coordinate(self):
        """Test hybrid vertical coordinate."""
        levels = [1000.0, 925.5, 850.0]
        axis = VerticalAxis(flag=4, levels=levels)
        assert axis.coord_system == "hybrid"
        np.testing.assert_array_equal(axis.levels, levels)

    def test_unknown_coordinate(self):
        """Test unknown vertical coordinate."""
        levels = [1.0, 0.5]
        axis = VerticalAxis(flag=99, levels=levels)
        assert axis.coord_system == "unknown"

    def test_sigma_coordinate_calculates_native_coords(self):
        """Test sigma coordinate returns native sigma fractions as level coord."""
        axis = VerticalAxis(flag=1, levels=[1.0, 0.5], offset=0.0)
        coords = axis.calculate_coords()
        assert set(coords.keys()) == {"level"}
        np.testing.assert_allclose(coords["level"], [1.0, 0.5])


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
