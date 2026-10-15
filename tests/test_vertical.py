"""Tests for VerticalAxis and coordinate helpers."""

import pickle
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from arlmet.vertical import (
    HybridAxis,
    PressureAxis,
    SigmaAxis,
    TerrainAxis,
    VerticalAxis,
    WrfAxis,
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

    def test_returns_wrf_axis(self):
        ax = VerticalAxis.from_flag(5, levels=[1.0, 0.995, 0.98], offset=50.0)
        assert isinstance(ax, WrfAxis)
        assert ax.flag == 5
        assert ax.coord_system == "wrf"
        assert ax.offset == 50.0

    def test_unknown_flag_raises(self):
        with pytest.raises(ValueError, match="Unsupported vertical flag 99"):
            VerticalAxis.from_flag(99, levels=[1.0])


class TestImmutability:
    def test_levels_are_read_only(self):
        ax = PressureAxis(levels=[1000.0, 900.0])
        with pytest.raises(ValueError, match="read-only"):
            ax.levels[0] = -1.0
        np.testing.assert_allclose(ax.levels, [1000.0, 900.0])

    def test_levels_are_copied_from_input(self):
        source = np.array([1000.0, 900.0])
        ax = PressureAxis(levels=source)
        source[0] = -1.0
        np.testing.assert_allclose(ax.levels, [1000.0, 900.0])

    @pytest.mark.parametrize("name", ["levels", "offset", "new_attr"])
    def test_attribute_assignment_raises(self, name):
        ax = SigmaAxis(levels=[1.0, 0.9], offset=5.0)
        with pytest.raises(FrozenInstanceError):
            setattr(ax, name, 1.0)
        with pytest.raises(FrozenInstanceError):
            delattr(ax, name)

    def test_hash_is_stable_and_usable_as_key(self):
        ax = HybridAxis(levels=[0.995, 975.5])
        first = hash(ax)
        _ = ax.to_pressure(surface_pressure=[1000.0])
        assert hash(ax) == first
        assert {ax: "x"}[HybridAxis(levels=[0.995, 975.5])] == "x"

    def test_levels_must_be_1d(self):
        with pytest.raises(ValueError, match="1-D"):
            PressureAxis(levels=[[1000.0, 900.0]])

    def test_pickle_roundtrip(self):
        ax = SigmaAxis(levels=[1.0, 0.9], offset=5.0)
        assert pickle.loads(pickle.dumps(ax)) == ax


class TestMissingInputs:
    @pytest.mark.parametrize("axis_cls", [SigmaAxis, HybridAxis])
    def test_sigma_hybrid_pressure_requires_surface_pressure(self, axis_cls):
        ax = axis_cls(levels=[0.995, 0.9])
        with pytest.raises(ValueError, match="surface_pressure="):
            ax.to_pressure()

    @pytest.mark.parametrize("axis_cls", [SigmaAxis, HybridAxis])
    def test_sigma_hybrid_height_requires_temperature(self, axis_cls):
        ax = axis_cls(levels=[0.995, 0.9])
        with pytest.raises(ValueError, match="surface_pressure="):
            ax.to_height_agl(temperature=[[280.0, 275.0]])
        with pytest.raises(ValueError, match="temperature="):
            ax.to_height_agl(surface_pressure=[1000.0])

    def test_pressure_height_requires_hgts_and_terrain(self):
        ax = PressureAxis(levels=[1000.0, 900.0])
        with pytest.raises(ValueError, match="hgts="):
            ax.to_height_agl(terrain=[0.0])
        with pytest.raises(ValueError, match="terrain="):
            ax.to_height_agl(hgts=[100.0, 1000.0])

    def test_wrf_height_requires_surface_pressure_temperature_and_pres(self):
        ax = WrfAxis(levels=[1.0, 0.995, 0.98], offset=50.0)
        inputs = {
            "surface_pressure": 850.0,
            "temperature": [285.0, 284.0],
            "pressure": [845.0, 832.0],
        }
        for name in inputs:
            given = {k: v for k, v in inputs.items() if k != name}
            with pytest.raises(ValueError, match=f"{name}="):
                ax.to_height_agl(**given)

    def test_sigma_height_matches_hypsometric(self):
        ax = SigmaAxis(levels=[1.0, 0.9])
        z = ax.to_height_agl(surface_pressure=[1000.0], temperature=[[280.0, 280.0]])
        assert z.shape == (1, 2)
        np.testing.assert_allclose(z[0, 0], 0.0, atol=1e-9)
        assert z[0, 1] > 0.0

    def test_options_are_keyword_only(self):
        with pytest.raises(TypeError):
            SigmaAxis(levels=[1.0]).to_pressure([1000.0])  # type: ignore[misc]


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

    def test_wrf_raises_and_points_to_pres(self):
        # Pressure on WRF hybrid levels needs WRF's hybrid coefficients, which
        # arw2arl writes to hybrid_wrfvcoords.txt, not to the ARL file.
        ax = WrfAxis(levels=[1.0, 0.995, 0.98], offset=50.0)
        with pytest.raises(ValueError, match="PRES"):
            ax.to_pressure(surface_pressure=[850.0])

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

    def test_wrf_hypsometric_from_pres_by_hand(self):
        # dz = (R_D / G) * T * ln(p_below / p_above), R_D / G = 287.05 / 9.80665
        # = 29.27095 m/K. Layer 1 (PRSS 850 -> PRES 820) uses T(level 1) = 279;
        # each layer above uses the mean T of its two levels.
        #   z1 = 29.27095 * 279.0 * ln(850 / 820) = 293.4422
        #   z2 = z1 + 29.27095 * 278.5 * ln(820 / 790) = 597.2772
        #   z3 = z2 + 29.27095 * 277.5 * ln(790 / 760) = 911.7431
        ax = WrfAxis(levels=[1.0, 0.99, 0.95, 0.9], offset=50.0)
        z = ax.to_height_agl(
            surface_pressure=850.0,
            temperature=np.array([279.0, 278.0, 277.0]),
            pressure=np.array([820.0, 790.0, 760.0]),
        )
        np.testing.assert_allclose(z, [293.4422, 597.2772, 911.7431], atol=1e-3)

    def test_wrf_height_per_point(self):
        # Two points, levels on the last axis; the class docstring's example
        # is the first row: 29.27095 * 285 * ln(850 / 845) = 49.2168 and
        # 49.2168 + 29.27095 * 284.5 * ln(845 / 832) = 178.3292.
        ax = WrfAxis(levels=[1.0, 0.995, 0.98], offset=50.0)
        z = ax.to_height_agl(
            surface_pressure=np.array([850.0, 850.0]),
            temperature=np.array([[285.0, 284.0], [279.0, 278.0]]),
            pressure=np.array([[845.0, 832.0], [820.0, 790.0]]),
        )
        np.testing.assert_allclose(
            z, [[49.2168, 178.3292], [293.4422, 597.2772]], atol=1e-3
        )

    def test_sigma_uses_pres_when_given_by_hand(self):
        # HYSPLIT's prfsig with PRES: level 0 is the ground (PRSS, its temperature);
        #   z1 = 29.27095 * (290 + 288) / 2 * ln(1000 / 988) = 102.1257
        #   z2 = z1 + 29.27095 * (288 + 286) / 2 * ln(988 / 945) = 475.9409
        # The nominal sigma pressures (990, 950) would give 85.0 and 431.5.
        ax = SigmaAxis(levels=[1.0, 0.99, 0.95])
        z = ax.to_height_agl(
            surface_pressure=1000.0,
            temperature=np.array([290.0, 288.0, 286.0]),
            pressure=np.array([np.nan, 988.0, 945.0]),
        )
        np.testing.assert_allclose(z, [0.0, 102.1257, 475.9409], atol=1e-3)

    def test_hybrid_ignores_pres(self):
        # HYSPLIT's prfecm takes a hybrid level's pressure from the levels only.
        ax = HybridAxis(levels=[1.0, 0.99, 0.95])
        args = {"surface_pressure": 1000.0, "temperature": [290.0, 288.0, 286.0]}
        np.testing.assert_array_equal(
            ax.to_height_agl(**args),
            ax.to_height_agl(**args, pressure=[np.nan, 988.0, 945.0]),
        )

    def test_wrf_ignores_hgts(self):
        # Never a fallback across flags: HGTS does not enter a WRF height.
        ax = WrfAxis(levels=[1.0, 0.99], offset=50.0)
        args = {"surface_pressure": 850.0, "temperature": [279.0], "pressure": [820.0]}
        np.testing.assert_array_equal(
            ax.to_height_agl(**args),
            ax.to_height_agl(**args, hgts=[5000.0], terrain=[1000.0]),
        )


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


def test_hypsometric_z_agl_surface_temperature_by_hand():
    # The first layer takes the mean of the ground and level-1 temperatures:
    #   z1 = 29.27095 * (290 + 288) / 2 * ln(1000 / 990) = 85.0189
    #   z2 = z1 + 29.27095 * (288 + 286) / 2 * ln(990 / 950) = 431.4912
    # Without it the first layer uses 288 alone: z1 = 84.7247.
    from arlmet.vertical import hypsometric_z_agl

    z = hypsometric_z_agl(
        [990.0, 950.0], 1000.0, [288.0, 286.0], surface_temperature=290.0
    )
    np.testing.assert_allclose(z, [85.0189, 431.4912], atol=1e-3)
    z_dry = hypsometric_z_agl([990.0, 950.0], 1000.0, [288.0, 286.0])
    np.testing.assert_allclose(z_dry[0], 84.7247, atol=1e-3)


def _write_column_file(path, *, axis, pres=True):
    """A sigma or hybrid file shaped like HRRR's: T02M, PRES, TEMP, SPHU (levels 1-3)."""
    grid = _make_latlon_grid()
    ny, nx = grid.ny, grid.nx

    def field(value):
        return np.full((ny, nx), value, dtype=np.float32)

    with File(path, mode="w", source="TEST", grid=grid, vertical_axis=axis) as f:
        rs = f.create_recordset(pd.Timestamp("2024-01-01"))
        rs.create_datarecord("PRSS", level=0, forecast=0, data=field(850.0))
        rs.create_datarecord("SHGT", level=0, forecast=0, data=field(1300.0))
        rs.create_datarecord("T02M", level=0, forecast=0, data=field(285.0))
        for k, (p, t, q) in enumerate(
            zip(
                [838.0, 800.0, 760.0],
                [283.0, 280.0, 277.0],
                [0.008, 0.006, 0.004],
                strict=True,
            ),
            start=1,
        ):
            if pres:
                rs.create_datarecord("PRES", level=k, forecast=0, data=field(p))
            rs.create_datarecord("TEMP", level=k, forecast=0, data=field(t))
            rs.create_datarecord("SPHU", level=k, forecast=0, data=field(q))


def _column_by_hand(pressures):
    """Heights of levels 1-3 from the ground (850 hPa, T02M 285 K), virtual temperature."""
    tv = [285.0 * (1 + 0.61 * 0.008)] + [
        t * (1 + 0.61 * q)
        for t, q in zip([283.0, 280.0, 277.0], [0.008, 0.006, 0.004], strict=True)
    ]
    p = [850.0, *pressures]
    z, out = 0.0, []
    for k in range(1, 4):
        z += (R_D / G) * 0.5 * (tv[k - 1] + tv[k]) * np.log(p[k - 1] / p[k])
        out.append(z)
    return np.array(out)


class TestColumnHelpers:
    """pressure() and z_agl() on sigma/hybrid files, as HYSPLIT's prfsig/prfecm (#66)."""

    def test_sigma_pressure_is_the_pres_field(self, tmp_path):
        path = tmp_path / "sigma.arl"
        _write_column_file(path, axis=SigmaAxis(levels=[1.0, 0.99, 0.95, 0.9]))
        ds = arlmet.open_dataset(path)
        p = arlmet.pressure(ds).isel(time=0, lat=0, lon=0).values
        np.testing.assert_allclose(p, [838.0, 800.0, 760.0])

    def test_sigma_z_agl_from_pres_with_virtual_temperature(self, tmp_path):
        path = tmp_path / "sigma.arl"
        _write_column_file(path, axis=SigmaAxis(levels=[1.0, 0.99, 0.95, 0.9]))
        ds = arlmet.open_dataset(path)
        z = arlmet.z_agl(ds)
        assert z.dims == ds["TEMP"].dims
        np.testing.assert_allclose(
            z.isel(time=0, lat=0, lon=0).values,
            _column_by_hand([838.0, 800.0, 760.0]),
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            arlmet.z_msl(ds).isel(time=0, lat=0, lon=0).values,
            _column_by_hand([838.0, 800.0, 760.0]) + 1300.0,
            rtol=1e-6,
        )

    def test_sigma_without_pres_uses_nominal_pressure(self, tmp_path):
        path = tmp_path / "sigma.arl"
        _write_column_file(
            path, axis=SigmaAxis(levels=[1.0, 0.99, 0.95, 0.9]), pres=False
        )
        ds = arlmet.open_dataset(path)
        nominal = [850.0 * s for s in (0.99, 0.95, 0.9)]
        np.testing.assert_allclose(
            arlmet.pressure(ds).isel(time=0, lat=0, lon=0).values, nominal, rtol=1e-6
        )
        np.testing.assert_allclose(
            arlmet.z_agl(ds).isel(time=0, lat=0, lon=0).values,
            _column_by_hand(nominal),
            rtol=1e-6,
        )

    def test_hybrid_ignores_pres(self, tmp_path):
        path = tmp_path / "hybrid.arl"
        _write_column_file(path, axis=HybridAxis(levels=[1.0, 0.99, 0.95, 0.9]))
        ds = arlmet.open_dataset(path)
        nominal = [850.0 * s for s in (0.99, 0.95, 0.9)]
        np.testing.assert_allclose(
            arlmet.pressure(ds).isel(time=0, lat=0, lon=0).values, nominal, rtol=1e-6
        )
        np.testing.assert_allclose(
            arlmet.z_agl(ds).isel(time=0, lat=0, lon=0).values,
            _column_by_hand(nominal),
            rtol=1e-6,
        )


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


# ---------------------------------------------------------------------------
# WRF (flag 5) files: written by arlmet, as HYSPLIT's arw2arl lays them out
# ---------------------------------------------------------------------------

# arw2arl's index: level 0 is the surface with height 1.0, then WRF's half-level
# eta (ZNU) bottom up; the reserved projection field holds P_TOP in hPa.
_WRF_LEVELS = [1.0, 0.99, 0.95, 0.9]
_WRF_PTOP = 50.0
_WRF_PRES = [820.0, 790.0, 760.0]
_WRF_TEMP = [279.0, 278.0, 277.0]
# Hand-worked heights for PRSS 850 hPa (see test_wrf_hypsometric_from_pres_by_hand)
_WRF_Z = [293.4422, 597.2772, 911.7431]


def _write_wrf_file(path, *, with_pres: bool = True):
    """Write a minimal WRF (flag=5) ARL file: PRSS 850 hPa, SHGT 1300 m."""
    grid = _make_latlon_grid()
    vaxis = WrfAxis(levels=_WRF_LEVELS, offset=_WRF_PTOP)
    ny, nx = grid.ny, grid.nx

    def full(value):
        return np.full((ny, nx), value, dtype=np.float32)

    with File(path, mode="w", source="AWRF", grid=grid, vertical_axis=vaxis) as f:
        rs = f.create_recordset(pd.Timestamp("2024-01-01"))
        rs.create_datarecord("SHGT", level=0, forecast=0, data=full(1300.0))
        rs.create_datarecord("PRSS", level=0, forecast=0, data=full(850.0))
        for k, (p, t) in enumerate(zip(_WRF_PRES, _WRF_TEMP, strict=True), start=1):
            if with_pres:
                rs.create_datarecord("PRES", level=k, forecast=0, data=full(p))
            rs.create_datarecord("TEMP", level=k, forecast=0, data=full(t))


class TestWrfFile:
    def test_index_record_stores_flag_5_and_model_top(self, tmp_path):
        path = tmp_path / "wrf.arl"
        _write_wrf_file(path)
        raw = path.read_bytes()
        # The index record's fixed part follows the 50-byte header: the
        # vertical flag is its bytes 102:104 (I2), as HYSPLIT's metset reads it.
        assert raw[14:18] == b"INDX"
        assert raw[50 + 102 : 50 + 104] == b" 5"
        with File(path) as f:
            assert f.vertical_axis == WrfAxis(levels=_WRF_LEVELS, offset=_WRF_PTOP)
            np.testing.assert_allclose(f.vertical_axis.levels, _WRF_LEVELS)

    def test_open_dataset_has_eta_coordinate(self, tmp_path):
        path = tmp_path / "wrf.arl"
        _write_wrf_file(path)
        ds = arlmet.open_dataset(path)
        assert ds.attrs["vertical_flag"] == 5
        np.testing.assert_allclose(ds.coords["eta"].values, _WRF_LEVELS[1:])
        assert ds.coords["eta"].attrs["surface"] == 1.0
        assert ds.coords["eta"].attrs["offset"] == _WRF_PTOP
        assert ds.arl.vertical_axis == WrfAxis(levels=_WRF_LEVELS, offset=_WRF_PTOP)

    def test_write_dataset_round_trip(self, tmp_path):
        src = tmp_path / "wrf.arl"
        dest = tmp_path / "wrf_copy.arl"
        _write_wrf_file(src)
        ds = arlmet.open_dataset(src)
        arlmet.write_dataset(ds, dest)
        with File(src) as a, File(dest) as b:
            assert b.vertical_axis == a.vertical_axis
        back = arlmet.open_dataset(dest)
        np.testing.assert_array_equal(back["PRES"].values, ds["PRES"].values)
        np.testing.assert_array_equal(back["TEMP"].values, ds["TEMP"].values)

    def test_extract_subset_keeps_flag_and_model_top(self, tmp_path):
        src = tmp_path / "wrf.arl"
        dest = tmp_path / "wrf_sub.arl"
        _write_wrf_file(src)
        arlmet.extract_subset(src, dest, levels=[0, 2])
        with File(dest) as f:
            assert f.vertical_axis == WrfAxis(levels=[1.0, 0.95], offset=_WRF_PTOP)
            np.testing.assert_allclose(f[0][(1, "PRES")].read(), 790.0)

    def test_pressure_is_the_pres_field(self, tmp_path):
        path = tmp_path / "wrf.arl"
        _write_wrf_file(path)
        ds = arlmet.open_dataset(path)
        p = arlmet.pressure(ds)
        np.testing.assert_allclose(p.isel(time=0, lat=0, lon=0).values, _WRF_PRES)

    def test_pressure_without_pres_raises(self, tmp_path):
        path = tmp_path / "wrf.arl"
        _write_wrf_file(path, with_pres=False)
        ds = arlmet.open_dataset(path)
        with pytest.raises(ValueError, match="PRES"):
            arlmet.pressure(ds)
        with pytest.raises(ValueError, match="PRES"):
            arlmet.z_agl(ds)

    def test_z_agl_and_z_msl_by_hand(self, tmp_path):
        path = tmp_path / "wrf.arl"
        _write_wrf_file(path)
        ds = arlmet.open_dataset(path)
        z = arlmet.z_agl(ds).isel(time=0, lat=0, lon=0).values
        np.testing.assert_allclose(z, _WRF_Z, atol=1e-2)
        z_msl = arlmet.z_msl(ds).isel(time=0, lat=0, lon=0).values
        np.testing.assert_allclose(z_msl, np.add(_WRF_Z, 1300.0), atol=1e-2)

    def test_sample_points_in_pressure_agl_and_msl(self, tmp_path):
        path = tmp_path / "wrf.arl"
        _write_wrf_file(path)
        # A quarter of the way from level 1 to level 2: TEMP 279 -> 278 and
        # PRES 820 -> 790 are linear in the target coordinate.
        z_quarter = _WRF_Z[0] + 0.25 * (_WRF_Z[1] - _WRF_Z[0])
        point = {"lon": [-105.0], "lat": [45.0], "time": [pd.Timestamp("2024-01-01")]}
        with File(path) as f:
            by_p = f.sample_points({**point, "z": [812.5]}, ["TEMP"], z_kind="pressure")
            agl = f.sample_points(
                {**point, "z": [z_quarter]}, ["TEMP", "pressure"], z_kind="agl"
            )
            msl = f.sample_points(
                {**point, "z": [z_quarter + 1300.0]}, ["TEMP", "pressure"], z_kind="msl"
            )
        np.testing.assert_allclose(by_p["TEMP"].iloc[0], 278.75, atol=1e-4)
        for result in (agl, msl):
            np.testing.assert_allclose(result["TEMP"].iloc[0], 278.75, atol=1e-3)
            np.testing.assert_allclose(result["pressure"].iloc[0], 812.5, atol=1e-2)

    def test_sample_points_below_first_level_uses_the_surface(self, tmp_path):
        path = tmp_path / "wrf.arl"
        _write_wrf_file(path)
        # The virtual pressure profile starts at PRSS (850 hPa) at 0 m AGL.
        point = {"lon": [-105.0], "lat": [45.0], "z": [0.5 * _WRF_Z[0]]}
        with File(path) as f:
            agl = f.sample_points(point, ["pressure"], z_kind="agl")
        np.testing.assert_allclose(agl["pressure"].iloc[0], 835.0, atol=1e-2)

    def test_sample_points_without_pres_raises(self, tmp_path):
        path = tmp_path / "wrf.arl"
        _write_wrf_file(path, with_pres=False)
        point = {"lon": [-105.0], "lat": [45.0], "z": [500.0]}
        with File(path) as f:
            for z_kind in ("pressure", "agl"):
                with pytest.raises(ValueError, match="PRES"):
                    f.sample_points(point, ["TEMP"], z_kind=z_kind)
