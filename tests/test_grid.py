"""Tests for arlmet.grid module."""

import numpy as np
import pytest

from arlmet.grid import Projection, Grid, VerticalAxis, Grid3D, wrap_lons


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
            reserved=0.0,
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
            reserved=0.0,
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
            reserved=0.0,
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
            reserved=0.0,
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
                reserved=0.0,
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
            reserved=0.0,
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
            reserved=0.0,
        )
        grid = Grid(projection=proj, nx=100, ny=100)

        assert grid.is_latlon is False
        assert grid.dims == ("y", "x")
        assert "x" in grid.coords
        assert "y" in grid.coords
        assert "lon" in grid.coords
        assert "lat" in grid.coords


class TestVerticalAxis:
    """Tests for VerticalAxis class."""

    def test_sigma_coordinate(self):
        """Test sigma vertical coordinate."""
        levels = [1.0, 0.9, 0.8, 0.7]
        axis = VerticalAxis(flag=1, levels=levels)
        assert axis.coord_system == "sigma"
        assert axis.heights == levels

    def test_pressure_coordinate(self):
        """Test pressure vertical coordinate."""
        levels = [1000, 925, 850, 700, 500]
        axis = VerticalAxis(flag=2, levels=levels)
        assert axis.coord_system == "pressure"
        assert axis.heights == levels

    def test_terrain_coordinate(self):
        """Test terrain vertical coordinate."""
        levels = [0.0, 0.25, 0.5, 0.75, 1.0]
        axis = VerticalAxis(flag=3, levels=levels)
        assert axis.coord_system == "terrain"
        assert axis.heights == levels

    def test_hybrid_coordinate(self):
        """Test hybrid vertical coordinate."""
        levels = [1000.0, 925.5, 850.0]
        axis = VerticalAxis(flag=4, levels=levels)
        assert axis.coord_system == "hybrid"
        assert axis.heights == levels

    def test_unknown_coordinate(self):
        """Test unknown vertical coordinate."""
        levels = [1.0, 0.5]
        axis = VerticalAxis(flag=99, levels=levels)
        assert axis.coord_system == "unknown"

    def test_get_coord_attrs_sigma(self):
        """Test coordinate attributes for sigma levels."""
        levels = [1.0, 0.8, 0.5]
        axis = VerticalAxis(flag=1, levels=levels)
        attrs = axis.get_coord_attrs()

        assert isinstance(attrs, dict)
        assert attrs["units"] == "1"
        assert attrs["standard_name"] == "atmosphere_sigma_coordinate"
        assert attrs["positive"] == "down"

    def test_get_coord_attrs_pressure(self):
        """Test coordinate attributes for pressure levels."""
        levels = [1000.0, 925.0, 850.0]
        axis = VerticalAxis(flag=2, levels=levels)
        attrs = axis.get_coord_attrs()

        assert isinstance(attrs, dict)
        assert attrs["units"] == "hPa"
        assert attrs["standard_name"] == "air_pressure"
        assert attrs["positive"] == "down"

    def test_get_coord_attrs_terrain(self):
        """Test coordinate attributes for terrain following levels."""
        levels = [0.0, 100.0, 500.0]
        axis = VerticalAxis(flag=3, levels=levels)
        attrs = axis.get_coord_attrs()

        assert isinstance(attrs, dict)
        assert attrs["units"] == "m"
        assert attrs["standard_name"] == "height"
        assert attrs["positive"] == "up"


class TestGridCoordAttrs:
    """Tests for Grid coordinate attributes."""

    def test_latlon_grid_coord_attrs(self):
        """Test coordinate attributes for lat-lon grids."""
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
            sync_lon=-180.0,
            reserved=0.0,
        )
        grid = Grid(projection=proj, nx=360, ny=180)
        attrs = grid.get_coord_attrs()

        assert "lon" in attrs
        assert "lat" in attrs
        assert attrs["lon"]["units"] == "degrees_east"
        assert attrs["lat"]["units"] == "degrees_north"
        assert attrs["lon"]["standard_name"] == "longitude"
        assert attrs["lat"]["standard_name"] == "latitude"

    def test_projected_grid_coord_attrs(self):
        """Test coordinate attributes for projected grids."""
        proj = Projection(
            pole_lat=90.0,
            pole_lon=-95.0,
            tangent_lat=45.0,
            tangent_lon=-95.0,
            grid_size=12.0,
            orientation=0.0,
            cone_angle=45.0,
            sync_x=1.0,
            sync_y=1.0,
            sync_lat=40.0,
            sync_lon=-100.0,
            reserved=0.0,
        )
        grid = Grid(projection=proj, nx=100, ny=100)
        attrs = grid.get_coord_attrs()

        assert "x" in attrs
        assert "y" in attrs
        assert "lon" in attrs
        assert "lat" in attrs
        assert attrs["x"]["units"] == "m"
        assert attrs["y"]["units"] == "m"
        assert attrs["x"]["standard_name"] == "projection_x_coordinate"
        assert attrs["y"]["standard_name"] == "projection_y_coordinate"


class TestGrid3D:
    """Tests for Grid3D class."""

    def test_grid3d_initialization(self):
        """Test 3D grid initialization."""
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
            reserved=0.0,
        )
        vertical_axis = VerticalAxis(flag=2, levels=[1000, 925, 850])
        grid = Grid3D(proj=proj, nx=10, ny=10, vertical_axis=vertical_axis)

        assert grid.nx == 10
        assert grid.ny == 10
        assert grid.vertical_axis.coord_system == "pressure"
        assert len(grid.vertical_axis.levels) == 3

    def test_grid3d_coord_attrs(self):
        """Test coordinate attributes for 3D grid."""
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
            reserved=0.0,
        )
        vertical_axis = VerticalAxis(flag=2, levels=[1000, 925, 850])
        grid = Grid3D(projection=proj, nx=10, ny=10, vertical_axis=vertical_axis)

        attrs = grid.get_coord_attrs()

        assert "level" in attrs
        assert "time" in attrs
        assert "forecast" in attrs
        assert "lon" in attrs
        assert "lat" in attrs

        # Check level attrs
        assert attrs["level"]["units"] == "hPa"
        assert attrs["level"]["standard_name"] == "air_pressure"

        # Check time attrs
        assert attrs["time"]["standard_name"] == "time"

        # Check forecast attrs
        assert attrs["forecast"]["units"] == "hours"
