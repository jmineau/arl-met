"""Tests for VerticalAxis and coordinate helpers."""

import numpy as np
import pytest

from arlmet.vertical import (
    HybridAxis,
    PressureAxis,
    SigmaAxis,
    TerrainAxis,
    VerticalAxis,
)


class TestFromFlag:
    def test_returns_sigma_axis(self):
        ax = VerticalAxis.from_flag(1, levels=[1.0, 0.9], offset=100.0)
        assert isinstance(ax, SigmaAxis)
        assert ax.flag == 1
        assert ax.coord_system == "sigma"

    def test_returns_pressure_axis(self):
        ax = VerticalAxis.from_flag(2, levels=[1000.0, 900.0])
        assert isinstance(ax, PressureAxis)
        assert ax.flag == 2
        assert ax.coord_system == "pressure"

    def test_returns_terrain_axis(self):
        ax = VerticalAxis.from_flag(3, levels=[0.0, 100.0])
        assert isinstance(ax, TerrainAxis)
        assert ax.flag == 3
        assert ax.coord_system == "terrain"

    def test_returns_hybrid_axis(self):
        ax = VerticalAxis.from_flag(4, levels=[0.995, 0.9])
        assert isinstance(ax, HybridAxis)
        assert ax.flag == 4
        assert ax.coord_system == "hybrid"

    def test_unknown_flag_raises(self):
        with pytest.raises(ValueError, match="Unsupported vertical flag 99"):
            VerticalAxis.from_flag(99, levels=[1.0])


class TestCalculateCoords:
    def test_pressure_axis_returns_native_hpa_values(self):
        ax = PressureAxis(levels=[1000.0, 900.0, 850.0])
        coords = ax.calculate_coords()
        assert set(coords.keys()) == {"level"}
        np.testing.assert_allclose(coords["level"], [1000.0, 900.0, 850.0])

    def test_sigma_axis_returns_native_sigma_fractions(self):
        ax = SigmaAxis(levels=[1.0, 0.9, 0.8], offset=100.0)
        coords = ax.calculate_coords()
        np.testing.assert_allclose(coords["level"], [1.0, 0.9, 0.8])

    def test_terrain_axis_returns_native_agl_heights(self):
        ax = TerrainAxis(levels=[0.0, 100.0, 500.0])
        coords = ax.calculate_coords()
        np.testing.assert_allclose(coords["level"], [0.0, 100.0, 500.0])

    def test_returns_copy_not_reference(self):
        ax = PressureAxis(levels=[1000.0, 900.0])
        c1 = ax.calculate_coords()
        c2 = ax.calculate_coords()
        c1["level"][0] = 9999.0
        np.testing.assert_allclose(c2["level"][0], 1000.0)

    def test_levels_property_returns_copy(self):
        ax = PressureAxis(levels=[1000.0, 900.0])
        levels = ax.levels
        levels[0] = -1.0
        np.testing.assert_allclose(ax.levels, [1000.0, 900.0])


class TestVerticalAxisEquality:
    def test_equality_hash_and_non_axis_comparison(self):
        left = PressureAxis(levels=[1000.0, 900.0], offset=5.0)
        right = PressureAxis(levels=[1000.0, 900.0], offset=5.0)
        different = SigmaAxis(levels=[1.0, 0.9], offset=0.0)

        assert left == right
        assert left != different
        assert left != object()
        assert hash(left) == hash(right)


class TestToPressure:
    def test_sigma_single_point(self):
        # p = p_top + (sp - p_top) * sigma
        ax = SigmaAxis(levels=[1.0, 0.9, 0.8], offset=100.0)
        sp = np.array([1000.0])
        result = ax.to_pressure(surface_pressure=sp)
        expected = 100.0 + (1000.0 - 100.0) * np.array([1.0, 0.9, 0.8])
        np.testing.assert_allclose(result, expected[None, :])

    def test_sigma_multiple_points(self):
        ax = SigmaAxis(levels=[1.0, 0.9], offset=100.0)
        sp = np.array([1000.0, 900.0])
        result = ax.to_pressure(surface_pressure=sp)
        assert result.shape == (2, 2)
        # point 0: sigma=1 → 1000; sigma=0.9 → 100 + 900*0.9 = 910
        np.testing.assert_allclose(result[0], [1000.0, 910.0])
        # point 1: sigma=1 → 900; sigma=0.9 → 100 + 800*0.9 = 820
        np.testing.assert_allclose(result[1], [900.0, 820.0])

    def test_pressure_returns_stored_levels(self):
        ax = PressureAxis(levels=[1000.0, 900.0, 850.0])
        result = ax.to_pressure()
        np.testing.assert_allclose(result, [1000.0, 900.0, 850.0])

    def test_hybrid_surface_level(self):
        # hybrid: heights encoded as floor_pressure + sigma_fraction
        # level 0 (first) always returns surface pressure
        ax = HybridAxis(levels=[0.995, 975.5, 950.3])
        sp = np.array([1010.0])
        result = ax.to_pressure(surface_pressure=sp)
        assert result.shape == (1, 3)
        np.testing.assert_allclose(result[0, 0], sp[0])  # first level = surface

    def test_terrain_raises(self):
        ax = TerrainAxis(levels=[0.0, 100.0, 500.0])
        with pytest.raises(ValueError, match="Terrain-following"):
            ax.to_pressure()

    def test_sigma_returns_2d_array(self):
        ax = SigmaAxis(levels=[1.0, 0.9], offset=0.0)
        result = ax.to_pressure(surface_pressure=np.array([1000.0, 900.0, 800.0]))
        assert result.ndim == 2
        assert result.shape == (3, 2)


class TestToHeightAgl:
    def test_pressure_uses_hgts(self):
        ax = PressureAxis(levels=[1000.0, 900.0])
        hgts = np.array([[200.0, 1200.0], [300.0, 1300.0]])
        terrain = np.array([[200.0], [300.0]])
        result = ax.to_height_agl(hgts=hgts, terrain=terrain)
        np.testing.assert_allclose(result, [[0.0, 1000.0], [0.0, 1000.0]])

    def test_terrain_returns_stored_levels(self):
        ax = TerrainAxis(levels=[0.0, 100.0, 500.0])
        result = ax.to_height_agl()
        np.testing.assert_allclose(result, [0.0, 100.0, 500.0])


# ---------------------------------------------------------------------------
# Module-level xarray vertical helpers
# ---------------------------------------------------------------------------


import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

import arlmet  # noqa: E402
from arlmet import File  # noqa: E402
from arlmet.grid import Grid, Projection  # noqa: E402
from arlmet.vertical import R_D, G  # noqa: E402
from arlmet.xarray._vertical import _hypsometric_z_agl  # noqa: E402


def _make_latlon_grid(nx: int = 20, ny: int = 20) -> Grid:
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
        sync_lat=40.0,
        sync_lon=-110.0,
    )
    return Grid(projection=proj, nx=nx, ny=ny)


def _write_pressure_file(
    path,
    *,
    prss_val,
    temp_val,
    shgt_val,
    pressure_levels,
    with_hgts: bool = False,
    hgts_values=None,
):
    """Write a minimal pressure-coordinate ARL file."""
    grid = _make_latlon_grid()
    vaxis = PressureAxis(levels=[0.0] + list(pressure_levels))
    ny, nx = grid.ny, grid.nx
    prss = np.full((ny, nx), prss_val, dtype=np.float32)
    shgt = np.full((ny, nx), shgt_val, dtype=np.float32)

    with File(path, mode="w", source="TEST", grid=grid, vertical_axis=vaxis) as f:
        rs = f.create_recordset(pd.Timestamp("2024-01-01"))
        rs.create_datarecord("PRSS", level=0, forecast=0, data=prss)
        rs.create_datarecord("SHGT", level=0, forecast=0, data=shgt)
        for k, _p_lev in enumerate(pressure_levels, start=1):
            temp = np.full((ny, nx), temp_val, dtype=np.float32)
            rs.create_datarecord("TEMP", level=k, forecast=0, data=temp)
            if with_hgts and hgts_values is not None:
                hgts = np.full((ny, nx), hgts_values[k - 1], dtype=np.float32)
                rs.create_datarecord("HGTS", level=k, forecast=0, data=hgts)


def _write_sigma_file(path, *, prss_val, temp_val, shgt_val, sigma_levels, offset=0.0):
    """Write a minimal sigma-coordinate ARL file (no HGTS)."""
    grid = _make_latlon_grid()
    vaxis = SigmaAxis(levels=[0.0] + list(sigma_levels), offset=offset)
    ny, nx = grid.ny, grid.nx
    prss = np.full((ny, nx), prss_val, dtype=np.float32)
    shgt = np.full((ny, nx), shgt_val, dtype=np.float32)

    with File(path, mode="w", source="TEST", grid=grid, vertical_axis=vaxis) as f:
        rs = f.create_recordset(pd.Timestamp("2024-01-01"))
        rs.create_datarecord("PRSS", level=0, forecast=0, data=prss)
        rs.create_datarecord("SHGT", level=0, forecast=0, data=shgt)
        for k in range(1, len(sigma_levels) + 1):
            temp = np.full((ny, nx), temp_val, dtype=np.float32)
            rs.create_datarecord("TEMP", level=k, forecast=0, data=temp)


def _write_terrain_file(
    path, *, height_levels, with_pres: bool = False, with_hgts: bool = False
):
    """Write a minimal terrain-following (flag=3) ARL file."""
    grid = _make_latlon_grid()
    vaxis = TerrainAxis(levels=[0.0] + list(height_levels))
    ny, nx = grid.ny, grid.nx
    prss = np.full((ny, nx), 1013.0, dtype=np.float32)

    with File(path, mode="w", source="TEST", grid=grid, vertical_axis=vaxis) as f:
        rs = f.create_recordset(pd.Timestamp("2024-01-01"))
        rs.create_datarecord("PRSS", level=0, forecast=0, data=prss)
        for k, h in enumerate(height_levels, start=1):
            data = np.full((ny, nx), float(k) * 10.0, dtype=np.float32)
            rs.create_datarecord("UWND", level=k, forecast=0, data=data)
            if with_pres:
                p_data = np.full((ny, nx), 1000.0 - float(k) * 50.0, dtype=np.float32)
                rs.create_datarecord("PRES", level=k, forecast=0, data=p_data)
            if with_hgts:
                h_data = np.full((ny, nx), h, dtype=np.float32)
                rs.create_datarecord("HGTS", level=k, forecast=0, data=h_data)


class TestHypsometricZagl:
    """Unit tests for _hypsometric_z_agl with synthetic DataArrays."""

    def _make_inputs(self, p_lev, p_sfc, t_lev):
        """Build minimal 1D DataArrays for testing."""
        p = xr.DataArray(np.array(p_lev, dtype=float), dims=["level"])
        prss = xr.DataArray(np.array([p_sfc], dtype=float), dims=["x"])
        temp = xr.DataArray(
            np.array([[t] for t in t_lev], dtype=float).reshape(len(t_lev), 1),
            dims=["level", "x"],
        )
        return p, prss, temp

    def test_single_level_positive_dz(self):
        # p_sfc=1013, p_lev=1000, T=290 K
        # dz ≈ (R_D/G) * 290 * ln(1013/1000)
        p, prss, temp = self._make_inputs([1000.0], 1013.0, [290.0])
        z = _hypsometric_z_agl(p, prss, temp)
        expected = (R_D / G) * 290.0 * np.log(1013.0 / 1000.0)
        np.testing.assert_allclose(float(z.values[0, 0]), expected, rtol=1e-5)

    def test_two_levels_cumulative(self):
        # p = [1000, 850], T = [290, 290], PRSS = 1013
        p, prss, temp = self._make_inputs([1000.0, 850.0], 1013.0, [290.0, 290.0])
        z = _hypsometric_z_agl(p, prss, temp)
        dz0 = (R_D / G) * 290.0 * np.log(1013.0 / 1000.0)
        dz1 = (R_D / G) * 290.0 * np.log(1000.0 / 850.0)  # T_mean = (290+290)/2 = 290
        np.testing.assert_allclose(float(z.values[0, 0]), dz0, rtol=1e-4)
        np.testing.assert_allclose(float(z.values[1, 0]), dz0 + dz1, rtol=1e-4)

    def test_output_dims_match_temp(self):
        p, prss, temp = self._make_inputs([1000.0, 850.0], 1013.0, [290.0, 280.0])
        z = _hypsometric_z_agl(p, prss, temp)
        assert z.dims == temp.dims
        assert z.shape == temp.shape


class TestPressureHelper:
    def test_flag2_returns_pressure_coord(self, tmp_path):
        path = tmp_path / "test.arl"
        _write_pressure_file(
            path,
            prss_val=1013.0,
            temp_val=280.0,
            shgt_val=0.0,
            pressure_levels=[1000.0, 850.0],
        )
        ds = arlmet.open_dataset(path)
        p = arlmet.pressure(ds)
        assert "level" in p.dims
        np.testing.assert_allclose(p.values, [1000.0, 850.0])

    def test_flag2_returns_dataarray(self, tmp_path):
        path = tmp_path / "test.arl"
        _write_pressure_file(
            path,
            prss_val=1013.0,
            temp_val=280.0,
            shgt_val=0.0,
            pressure_levels=[1000.0, 850.0],
        )
        ds = arlmet.open_dataset(path)
        p = arlmet.pressure(ds)
        assert isinstance(p, xr.DataArray)

    def test_flag3_raises_not_implemented(self, tmp_path):
        path = tmp_path / "test.arl"
        _write_terrain_file(path, height_levels=[100.0, 500.0])
        ds = arlmet.open_dataset(path)
        with pytest.raises(ValueError, match="PRES"):
            arlmet.pressure(ds)

    def test_flag3_returns_pres_field_when_present(self, tmp_path):
        path = tmp_path / "test.arl"
        _write_terrain_file(path, height_levels=[100.0, 500.0], with_pres=True)
        ds = arlmet.open_dataset(path)
        p = arlmet.pressure(ds)
        assert isinstance(p, xr.DataArray)
        assert "level" in p.dims


class TestZAglHelper:
    def test_flag3_returns_height_coord(self, tmp_path):
        path = tmp_path / "test.arl"
        _write_terrain_file(path, height_levels=[100.0, 500.0])
        ds = arlmet.open_dataset(path)
        z = arlmet.z_agl(ds)
        assert "level" in z.dims
        np.testing.assert_allclose(z.values, [100.0, 500.0])

    def test_flag3_hgts_field_used_when_present(self, tmp_path):
        """flag=3 with HGTS still returns the height coord (not HGTS)."""
        path = tmp_path / "test.arl"
        _write_terrain_file(path, height_levels=[100.0, 500.0], with_hgts=True)
        ds = arlmet.open_dataset(path)
        z = arlmet.z_agl(ds)
        # Terrain-following uses stored height coord, not HGTS
        np.testing.assert_allclose(z.values, [100.0, 500.0])

    def test_flag2_uses_hgts(self, tmp_path):
        """flag=2 requires HGTS for z_agl."""
        path = tmp_path / "test.arl"
        _write_pressure_file(
            path,
            prss_val=1013.0,
            temp_val=280.0,
            shgt_val=100.0,
            pressure_levels=[1000.0, 850.0],
            with_hgts=True,
            hgts_values=[100.0, 1500.0],
        )
        ds = arlmet.open_dataset(path)
        z = arlmet.z_agl(ds)
        # HGTS - SHGT: [100-100, 1500-100] = [0, 1400]
        np.testing.assert_allclose(
            z.isel(time=0, lat=0, lon=0).values,
            [0.0, 1400.0],
        )

    def test_flag2_missing_hgts_raises(self, tmp_path):
        """flag=2 without HGTS raises ValueError."""
        path = tmp_path / "test.arl"
        _write_pressure_file(
            path,
            prss_val=1013.0,
            temp_val=280.0,
            shgt_val=0.0,
            pressure_levels=[1000.0, 850.0],
        )
        ds = arlmet.open_dataset(path)
        with pytest.raises(ValueError, match="HGTS"):
            arlmet.z_agl(ds)

    def test_sigma_hypsometric_increases_with_altitude(self, tmp_path):
        """flag=1 uses hypsometric integration (no HGTS needed)."""
        path = tmp_path / "test.arl"
        _write_sigma_file(
            path,
            prss_val=1013.0,
            temp_val=280.0,
            shgt_val=0.0,
            sigma_levels=[0.9, 0.8, 0.7],
            offset=0.0,
        )
        ds = arlmet.open_dataset(path)
        z = arlmet.z_agl(ds)
        z_vals = z.isel(time=0, lat=0, lon=0).values
        assert z_vals[0] < z_vals[1] < z_vals[2]

    def test_sigma_missing_temp_raises(self, tmp_path):
        path = tmp_path / "test.arl"
        _write_sigma_file(
            path,
            prss_val=1013.0,
            temp_val=280.0,
            shgt_val=0.0,
            sigma_levels=[0.9, 0.8],
        )
        ds = arlmet.open_dataset(path).drop_vars("TEMP")
        with pytest.raises(ValueError, match="TEMP"):
            arlmet.z_agl(ds)


class TestVerticalAxisRepr:
    def test_repr_pressure(self):
        ax = PressureAxis(levels=[1000.0, 850.0, 700.0])
        assert repr(ax) == "PressureAxis(n=3)"

    def test_repr_sigma(self):
        ax = SigmaAxis(levels=[1.0, 0.9])
        assert repr(ax) == "SigmaAxis(n=2)"

    def test_len(self):
        ax = PressureAxis(levels=[1000.0, 925.0, 850.0, 700.0])
        assert len(ax) == 4


class TestZMslHelper:
    def test_z_msl_equals_z_agl_plus_shgt(self, tmp_path):
        path = tmp_path / "test.arl"
        shgt_val = 150.0
        _write_pressure_file(
            path,
            prss_val=1013.0,
            temp_val=280.0,
            shgt_val=shgt_val,
            pressure_levels=[1000.0, 850.0],
            with_hgts=True,
            hgts_values=[150.0, 1500.0],
        )
        ds = arlmet.open_dataset(path)
        z = arlmet.z_agl(ds)
        z_msl_expected = z + ds["SHGT"]
        np.testing.assert_allclose(
            arlmet.z_msl(ds).values, z_msl_expected.values, rtol=1e-5
        )

    def test_z_msl_missing_shgt_raises(self, tmp_path):
        path = tmp_path / "test.arl"
        _write_pressure_file(
            path,
            prss_val=1013.0,
            temp_val=280.0,
            shgt_val=0.0,
            pressure_levels=[1000.0],
            with_hgts=True,
            hgts_values=[100.0],
        )
        ds = arlmet.open_dataset(path).drop_vars("SHGT")
        with pytest.raises(ValueError, match="SHGT"):
            arlmet.z_msl(ds)
