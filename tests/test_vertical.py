"""Tests for VerticalAxis and Grid3D coordinate helpers."""

import numpy as np
import pytest

from arlmet.grid import Projection
from arlmet.vertical import Grid3D, VerticalAxis


class TestCalculateCoords:
    def test_pressure_axis_returns_native_hpa_values(self):
        ax = VerticalAxis(flag=2, levels=[1000.0, 900.0, 850.0])
        coords = ax.calculate_coords()
        assert set(coords.keys()) == {"level"}
        np.testing.assert_allclose(coords["level"], [1000.0, 900.0, 850.0])

    def test_sigma_axis_returns_native_sigma_fractions(self):
        ax = VerticalAxis(flag=1, levels=[1.0, 0.9, 0.8], offset=100.0)
        coords = ax.calculate_coords()
        np.testing.assert_allclose(coords["level"], [1.0, 0.9, 0.8])

    def test_terrain_axis_returns_native_agl_heights(self):
        ax = VerticalAxis(flag=3, levels=[0.0, 100.0, 500.0])
        coords = ax.calculate_coords()
        np.testing.assert_allclose(coords["level"], [0.0, 100.0, 500.0])

    def test_returns_copy_not_reference(self):
        ax = VerticalAxis(flag=2, levels=[1000.0, 900.0])
        c1 = ax.calculate_coords()
        c2 = ax.calculate_coords()
        c1["level"][0] = 9999.0
        np.testing.assert_allclose(c2["level"][0], 1000.0)

    def test_coord_system_unknown_and_levels_property_returns_copy(self):
        ax = VerticalAxis(flag=99, levels=[1.0, 2.0])
        levels = ax.levels

        assert ax.coord_system == "unknown"
        levels[0] = -1.0
        np.testing.assert_allclose(ax.levels, [1.0, 2.0])


class TestVerticalAxisEquality:
    def test_equality_hash_and_non_axis_comparison(self):
        left = VerticalAxis(flag=2, levels=[1000.0, 900.0], offset=5.0)
        right = VerticalAxis(flag=2, levels=[1000.0, 900.0], offset=5.0)
        different = VerticalAxis(flag=1, levels=[1.0, 0.9], offset=0.0)

        assert left == right
        assert left != different
        assert left != object()
        assert hash(left) == hash(right)


class TestSigmaToPressure:
    def test_sigma_flag1_single_point(self):
        # p = p_top + (sp - p_top) * sigma
        ax = VerticalAxis(flag=1, levels=[1.0, 0.9, 0.8], offset=100.0)
        sp = np.array([1000.0])
        result = ax.sigma_to_pressure(sp, [0, 1, 2])
        expected = 100.0 + (1000.0 - 100.0) * np.array([1.0, 0.9, 0.8])
        np.testing.assert_allclose(result, expected[None, :])

    def test_sigma_flag1_multiple_points(self):
        ax = VerticalAxis(flag=1, levels=[1.0, 0.9], offset=100.0)
        sp = np.array([1000.0, 900.0])
        result = ax.sigma_to_pressure(sp, [0, 1])
        assert result.shape == (2, 2)
        # point 0: sigma=1 → 1000; sigma=0.9 → 100 + 900*0.9 = 910
        np.testing.assert_allclose(result[0], [1000.0, 910.0])
        # point 1: sigma=1 → 900; sigma=0.9 → 100 + 800*0.9 = 820
        np.testing.assert_allclose(result[1], [900.0, 820.0])

    def test_sigma_subset_of_levels(self):
        ax = VerticalAxis(flag=1, levels=[1.0, 0.9, 0.8], offset=0.0)
        sp = np.array([1013.0])
        result = ax.sigma_to_pressure(sp, [1, 2])
        expected = np.array([1013.0 * 0.9, 1013.0 * 0.8])
        np.testing.assert_allclose(result, expected[None, :])

    def test_hybrid_flag4_surface_level(self):
        # hybrid: heights encoded as floor_pressure + sigma_fraction
        # level 0 (first) always returns surface pressure
        ax = VerticalAxis(flag=4, levels=[0.995, 975.5, 950.3])
        sp = np.array([1010.0])
        result = ax.sigma_to_pressure(sp, [0, 1, 2])
        assert result.shape == (1, 3)
        np.testing.assert_allclose(result[0, 0], sp[0])  # first level = surface

    def test_invalid_flag_raises(self):
        ax = VerticalAxis(flag=2, levels=[1000.0, 900.0])
        with pytest.raises(ValueError, match="flag=1.*flag=4"):
            ax.sigma_to_pressure(np.array([1013.0]), [0, 1])

    def test_returns_2d_array(self):
        ax = VerticalAxis(flag=1, levels=[1.0, 0.9], offset=0.0)
        result = ax.sigma_to_pressure(np.array([1000.0, 900.0, 800.0]), [0, 1])
        assert result.ndim == 2
        assert result.shape == (3, 2)


class TestGrid3D:
    def make_projection(self):
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

    def test_requires_projection_and_vertical_axis(self):
        axis = VerticalAxis(flag=2, levels=[1000.0, 900.0])

        with pytest.raises(TypeError, match="requires `projection` or `proj`"):
            Grid3D(vertical_axis=axis)

        with pytest.raises(TypeError, match="requires a `vertical_axis`"):
            Grid3D(projection=self.make_projection(), nx=2, ny=2)

    def test_dims_and_coords_include_vertical_axis(self):
        axis = VerticalAxis(flag=2, levels=[1000.0, 900.0])
        grid = Grid3D(proj=self.make_projection(), nx=2, ny=3, vertical_axis=axis)

        assert grid.dims == ("level", "lat", "lon")
        coords = grid.calculate_coords()
        assert coords["level"][0] == "level"
        np.testing.assert_allclose(coords["level"][1], [1000.0, 900.0])
