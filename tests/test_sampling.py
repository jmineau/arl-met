"""Tests for point-sampling helpers."""

import numpy as np
import pandas as pd
import pytest

from arlmet import File, sample_points
from arlmet.grid import Grid, Projection
from arlmet.vertical import VerticalAxis


def make_test_grid(nx: int = 20, ny: int = 20) -> Grid:
    projection = Projection(
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
    return Grid(projection=projection, nx=nx, ny=ny)


_DZ = 9.88  # m/hPa used to build synthetic HGTS for pressure-level test fixtures


def write_sampling_file(path, *, time: pd.Timestamp, temp_offset: float = 0.0):
    grid = make_test_grid()
    vertical_axis = VerticalAxis(flag=2, levels=[1000.0, 900.0, 800.0])
    yy, xx = np.meshgrid(np.arange(grid.ny), np.arange(grid.nx), indexing="ij")
    base = 10.0 * yy + xx

    terrain = 100.0 + base
    surface_pressure = 1000.0 + 2.0 * yy + xx
    temp0 = 280.0 + temp_offset + base
    temp1 = 290.0 + temp_offset + base
    temp2 = 300.0 + temp_offset + base
    hgts0 = terrain + np.maximum((surface_pressure - 1000.0) * _DZ, 0.0)
    hgts1 = terrain + (surface_pressure - 900.0) * _DZ
    hgts2 = terrain + (surface_pressure - 800.0) * _DZ

    with File(
        path, mode="w", source="TEST", grid=grid, vertical_axis=vertical_axis
    ) as arl:
        rs = arl.create_recordset(time)
        rs.create_datarecord(
            "SHGT", level=0, forecast=0, data=terrain.astype(np.float32)
        )
        rs.create_datarecord(
            "PRSS", level=0, forecast=0, data=surface_pressure.astype(np.float32)
        )
        for level_index, (temp, hgts) in enumerate(
            zip([temp0, temp1, temp2], [hgts0, hgts1, hgts2], strict=True)
        ):
            rs.create_datarecord(
                "TEMP", level=level_index, forecast=0, data=temp.astype(np.float32)
            )
            rs.create_datarecord(
                "HGTS", level=level_index, forecast=0, data=hgts.astype(np.float32)
            )

    return {
        "grid": grid,
        "vertical_axis": vertical_axis,
        "terrain": terrain.astype(np.float32),
        "surface_pressure": surface_pressure.astype(np.float32),
        "temp0": temp0.astype(np.float32),
        "temp1": temp1.astype(np.float32),
        "temp2": temp2.astype(np.float32),
    }


_Z_PHI1 = [17.98, 14.73, 13.09, 11.98, 11.15, 10.52, 10.04, 9.75, 9.88]
_Z_PHI2 = [31.37, 27.02, 24.59, 22.92, 21.65, 20.66, 19.83, 19.13, 18.51]


def _level_dz(pressure_level: float) -> float:
    """Scale height (m/hPa) for a given pressure level, used to build test HGTS."""
    if pressure_level >= 100.0:
        return _Z_PHI1[min(int(pressure_level // 100), len(_Z_PHI1) - 1)]
    return _Z_PHI2[min(int(pressure_level // 10), len(_Z_PHI2) - 1)]


def write_deep_sampling_file(path, *, time: pd.Timestamp, temp_offset: float = 0.0):
    grid = make_test_grid(nx=30, ny=30)
    levels = [
        1000.0,
        900.0,
        800.0,
        700.0,
        600.0,
        500.0,
        400.0,
        300.0,
        200.0,
        100.0,
        50.0,
        10.0,
        1.0,
    ]
    vertical_axis = VerticalAxis(flag=2, levels=levels)
    yy, xx = np.meshgrid(np.arange(grid.ny), np.arange(grid.nx), indexing="ij")
    base = 0.1 * yy + 0.1 * xx

    terrain = 100.0 + base
    surface_pressure = 1000.0 + yy + xx

    with File(
        path, mode="w", source="TEST", grid=grid, vertical_axis=vertical_axis
    ) as arl:
        rs = arl.create_recordset(time)
        rs.create_datarecord(
            "SHGT", level=0, forecast=0, data=terrain.astype(np.float32)
        )
        rs.create_datarecord(
            "PRSS", level=0, forecast=0, data=surface_pressure.astype(np.float32)
        )
        for level_index, pressure_level in enumerate(levels):
            temp = 280.0 + temp_offset + 5.0 * level_index + base
            rs.create_datarecord(
                "TEMP", level=level_index, forecast=0, data=temp.astype(np.float32)
            )
            agl = np.maximum(
                (surface_pressure - pressure_level) * _level_dz(pressure_level), 0.0
            )
            hgts = (terrain + agl).astype(np.float32)
            rs.create_datarecord("HGTS", level=level_index, forecast=0, data=hgts)

    return {
        "grid": grid,
        "vertical_axis": vertical_axis,
        "terrain": terrain.astype(np.float32),
        "surface_pressure": surface_pressure.astype(np.float32),
    }


class TestPointSampling:
    def test_terrain_reads_full_surface_slice(self, tmp_path):
        path = tmp_path / "terrain.arl"
        time = pd.Timestamp("2024-07-18 00:00")
        sample = write_sampling_file(path, time=time)

        with File(path) as arl:
            result = arl.terrain(time)

        np.testing.assert_allclose(result, sample["terrain"])

    def test_file_sample_points_interpolates_native_and_pressure_queries(
        self, tmp_path
    ):
        path = tmp_path / "points.arl"
        time = pd.Timestamp("2024-07-18 00:00")
        write_sampling_file(path, time=time)

        points = pd.DataFrame(
            {
                "lon": [20.5, 20.5],
                "lat": [-9.5, -9.5],
                "z": [0.5, 950.0],
            }
        )

        with File(path) as arl:
            native = arl.sample_points(
                points.iloc[[0]], ["TEMP", "PRSS"], time=time, z_kind="native"
            )
            pressure_points = arl.sample_points(
                points.iloc[[1]],
                ["TEMP", "pressure"],
                time=time,
                z_kind="pressure",
            )

        np.testing.assert_allclose(native["TEMP"].iloc[0], 290.5)
        np.testing.assert_allclose(native["PRSS"].iloc[0], 1001.5)
        np.testing.assert_allclose(pressure_points["TEMP"].iloc[0], 290.5)
        np.testing.assert_allclose(pressure_points["pressure"].iloc[0], 950.0)

    def test_file_sample_points_supports_agl_and_msl_queries(self, tmp_path):
        path = tmp_path / "points.arl"
        time = pd.Timestamp("2024-07-18 00:00")
        write_sampling_file(path, time=time)

        surface_pressure = 1001.5
        terrain = 105.5
        z0 = (surface_pressure - 1000.0) * _DZ
        z1 = (surface_pressure - 900.0) * _DZ
        z_agl_mid = 0.5 * (z0 + z1)
        z_msl_mid = z_agl_mid + terrain

        points = pd.DataFrame(
            {
                "lon": [20.5, 20.5],
                "lat": [-9.5, -9.5],
                "z": [z_agl_mid, z_msl_mid],
            }
        )

        with File(path) as arl:
            agl = arl.sample_points(
                points.iloc[[0]],
                ["TEMP", "pressure"],
                time=time,
                z_kind="agl",
            )
            msl = arl.sample_points(
                points.iloc[[1]],
                ["TEMP", "pressure"],
                time=time,
                z_kind="msl",
            )

        np.testing.assert_allclose(agl["TEMP"].iloc[0], 290.5, atol=0.01)
        np.testing.assert_allclose(agl["pressure"].iloc[0], 950.0, atol=0.1)
        np.testing.assert_allclose(msl["TEMP"].iloc[0], 290.5, atol=0.01)
        np.testing.assert_allclose(msl["pressure"].iloc[0], 950.0, atol=0.1)

    def test_module_sample_points_dispatches_across_multiple_sources_by_time(
        self, tmp_path
    ):
        path0 = tmp_path / "points0.arl"
        path1 = tmp_path / "points1.arl"
        time0 = pd.Timestamp("2024-07-18 00:00")
        time1 = pd.Timestamp("2024-07-18 03:00")
        write_sampling_file(path0, time=time0, temp_offset=0.0)
        write_sampling_file(path1, time=time1, temp_offset=20.0)

        points = pd.DataFrame(
            {
                "time": [time0, time1],
                "lon": [20.5, 20.5],
                "lat": [-9.5, -9.5],
                "z": [950.0, 950.0],
            }
        )

        with File(path0) as f0, File(path1) as f1:
            result = sample_points([f0, f1], points, ["TEMP"], z_kind="pressure")

        np.testing.assert_allclose(result["TEMP"].to_numpy(), [290.5, 310.5])

    def test_file_sample_points_supports_simple_slant_path(self, tmp_path):
        path = tmp_path / "slant.arl"
        time = pd.Timestamp("2024-07-18 00:00")
        write_deep_sampling_file(path, time=time)

        points = pd.DataFrame(
            {
                "lon": np.linspace(20.0, 20.1, 1000, dtype=float),
                "lat": np.linspace(-10.0, -9.9, 1000, dtype=float),
                "z": np.linspace(0.0, 20000.0, 1000, dtype=float),
            }
        )

        with File(path) as arl:
            sampled = arl.sample_points(
                points,
                ["TEMP", "pressure"],
                time=time,
                z_kind="agl",
            )

        assert len(sampled) == 1000
        np.testing.assert_allclose(sampled["pressure"].iloc[0], 1000.0, atol=1e-5)
        np.testing.assert_allclose(sampled["TEMP"].iloc[0], 280.0, atol=1e-5)

        finite_pressure = sampled["pressure"].dropna().to_numpy()
        finite_temp = sampled["TEMP"].dropna().to_numpy()
        assert finite_pressure.size > 100
        assert finite_temp.size == finite_pressure.size
        assert np.all(np.diff(finite_pressure) <= 1e-5)
        assert finite_temp[-1] > finite_temp[0]


# --- sigma (flag=1), hybrid (flag=4), and terrain (flag=3) fixtures ---


def write_sigma_sampling_file(path, *, time: pd.Timestamp):
    """
    Sigma (flag=1) file with offset=0 (p_top=0 hPa).
    sigma=[1.0, 0.9, 0.8], PRSS=1000 → pressure levels [1000, 900, 800] hPa.
    HGTS=[100, 1100, 2100] m → AGL=[0, 1000, 2000] m, terrain=100 m.
    TEMP at each level: 280, 290, 300 K (uniform grid).
    """
    grid = make_test_grid()
    vertical_axis = VerticalAxis(flag=1, levels=[1.0, 0.9, 0.8], offset=0.0)
    terrain_data = np.full((grid.ny, grid.nx), 100.0, dtype=np.float32)
    surface_pressure = np.full((grid.ny, grid.nx), 1000.0, dtype=np.float32)

    with File(
        path, mode="w", source="TEST", grid=grid, vertical_axis=vertical_axis
    ) as arl:
        rs = arl.create_recordset(time)
        rs.create_datarecord("SHGT", level=0, forecast=0, data=terrain_data)
        rs.create_datarecord("PRSS", level=0, forecast=0, data=surface_pressure)
        for level_index, (temp_val, hgts_val) in enumerate(
            zip([280.0, 290.0, 300.0], [100.0, 1100.0, 2100.0], strict=True)
        ):
            temp = np.full((grid.ny, grid.nx), temp_val, dtype=np.float32)
            hgts = np.full((grid.ny, grid.nx), hgts_val, dtype=np.float32)
            rs.create_datarecord("TEMP", level=level_index, forecast=0, data=temp)
            rs.create_datarecord("HGTS", level=level_index, forecast=0, data=hgts)

    return {
        "grid": grid,
        "vertical_axis": vertical_axis,
        "terrain": terrain_data,
        "surface_pressure": surface_pressure,
    }


def write_terrain_sampling_file(path, *, time: pd.Timestamp):
    """
    Terrain-following (flag=3) file.
    Level heights (AGL): [0, 100, 500] m.
    TEMP: 280, 290, 300 K at each level (uniform grid).
    terrain (SHGT) = 200 m.
    No PRSS — not needed for flag=3 AGL queries.
    """
    grid = make_test_grid()
    vertical_axis = VerticalAxis(flag=3, levels=[0.0, 100.0, 500.0])
    terrain_data = np.full((grid.ny, grid.nx), 200.0, dtype=np.float32)

    with File(
        path, mode="w", source="TEST", grid=grid, vertical_axis=vertical_axis
    ) as arl:
        rs = arl.create_recordset(time)
        rs.create_datarecord("SHGT", level=0, forecast=0, data=terrain_data)
        for level_index, temp_val in enumerate([280.0, 290.0, 300.0]):
            temp = np.full((grid.ny, grid.nx), temp_val, dtype=np.float32)
            rs.create_datarecord("TEMP", level=level_index, forecast=0, data=temp)

    return {
        "grid": grid,
        "vertical_axis": vertical_axis,
        "terrain": terrain_data,
    }


class TestSigmaSampling:
    """
    Sigma (flag=1) file: sigma=[1.0, 0.9, 0.8], PRSS=1000 hPa, p_top=0.
    Effective pressure levels: [1000, 900, 800] hPa.
    HGTS=[100, 1100, 2100] m → AGL=[0, 1000, 2000] m. Terrain=100 m.
    TEMP: [280, 290, 300] K.
    """

    def test_native_z_kind(self, tmp_path):
        path = tmp_path / "sigma.arl"
        time = pd.Timestamp("2024-07-18 00:00")
        write_sigma_sampling_file(path, time=time)

        # midpoint between levels 0 and 1 (z_kind="native" uses integer level indices)
        points = pd.DataFrame({"lon": [20.0], "lat": [-10.0], "z": [0.5]})
        with File(path) as arl:
            result = arl.sample_points(points, ["TEMP"], time=time, z_kind="native")

        np.testing.assert_allclose(result["TEMP"].iloc[0], 285.0, atol=1e-4)

    def test_pressure_z_kind(self, tmp_path):
        path = tmp_path / "sigma.arl"
        time = pd.Timestamp("2024-07-18 00:00")
        write_sigma_sampling_file(path, time=time)

        # 950 hPa is midpoint between 1000 and 900 hPa → TEMP = 285
        points = pd.DataFrame({"lon": [20.0], "lat": [-10.0], "z": [950.0]})
        with File(path) as arl:
            result = arl.sample_points(
                points, ["TEMP", "pressure"], time=time, z_kind="pressure"
            )

        np.testing.assert_allclose(result["TEMP"].iloc[0], 285.0, atol=1e-4)
        np.testing.assert_allclose(result["pressure"].iloc[0], 950.0, atol=1e-4)

    def test_agl_z_kind(self, tmp_path):
        path = tmp_path / "sigma.arl"
        time = pd.Timestamp("2024-07-18 00:00")
        write_sigma_sampling_file(path, time=time)

        # AGL midpoint between levels 0 (0m) and 1 (1000m) = 500m → TEMP = 285
        points = pd.DataFrame({"lon": [20.0], "lat": [-10.0], "z": [500.0]})
        with File(path) as arl:
            result = arl.sample_points(
                points, ["TEMP", "pressure"], time=time, z_kind="agl"
            )

        np.testing.assert_allclose(result["TEMP"].iloc[0], 285.0, atol=1e-4)
        np.testing.assert_allclose(result["pressure"].iloc[0], 950.0, atol=1e-4)

    def test_msl_z_kind(self, tmp_path):
        path = tmp_path / "sigma.arl"
        time = pd.Timestamp("2024-07-18 00:00")
        write_sigma_sampling_file(path, time=time)

        # terrain=100m; MSL midpoint = 500m AGL + 100m = 600m → TEMP = 285
        points = pd.DataFrame({"lon": [20.0], "lat": [-10.0], "z": [600.0]})
        with File(path) as arl:
            result = arl.sample_points(
                points, ["TEMP", "pressure"], time=time, z_kind="msl"
            )

        np.testing.assert_allclose(result["TEMP"].iloc[0], 285.0, atol=1e-4)
        np.testing.assert_allclose(result["pressure"].iloc[0], 950.0, atol=1e-4)


def write_hybrid_sampling_file(path, *, time: pd.Timestamp):
    """
    Hybrid (flag=4) file with pure-sigma levels (floor_p=0 for all).
    levels=[0.995, 0.9, 0.8], PRSS=1000 hPa.
    Pressure: level 0 → 1000 hPa (surface override), level 1 → 900 hPa, level 2 → 800 hPa.
    HGTS=[100, 1100, 2100] m → AGL=[0, 1000, 2000] m, terrain=100 m.
    TEMP: 280, 290, 300 K (uniform grid).
    """
    grid = make_test_grid()
    vertical_axis = VerticalAxis(flag=4, levels=[0.995, 0.9, 0.8])
    terrain_data = np.full((grid.ny, grid.nx), 100.0, dtype=np.float32)
    surface_pressure = np.full((grid.ny, grid.nx), 1000.0, dtype=np.float32)

    with File(
        path, mode="w", source="TEST", grid=grid, vertical_axis=vertical_axis
    ) as arl:
        rs = arl.create_recordset(time)
        rs.create_datarecord("SHGT", level=0, forecast=0, data=terrain_data)
        rs.create_datarecord("PRSS", level=0, forecast=0, data=surface_pressure)
        for level_index, (temp_val, hgts_val) in enumerate(
            zip([280.0, 290.0, 300.0], [100.0, 1100.0, 2100.0], strict=True)
        ):
            temp = np.full((grid.ny, grid.nx), temp_val, dtype=np.float32)
            hgts = np.full((grid.ny, grid.nx), hgts_val, dtype=np.float32)
            rs.create_datarecord("TEMP", level=level_index, forecast=0, data=temp)
            rs.create_datarecord("HGTS", level=level_index, forecast=0, data=hgts)

    return {
        "grid": grid,
        "vertical_axis": vertical_axis,
        "terrain": terrain_data,
        "surface_pressure": surface_pressure,
    }


class TestHybridSampling:
    """
    Hybrid (flag=4) file: levels=[0.995, 0.9, 0.8], PRSS=1000 hPa.
    Effective pressure levels: [1000, 900, 800] hPa (floor_p=0 → pure sigma).
    HGTS=[100, 1100, 2100] m → AGL=[0, 1000, 2000] m. Terrain=100 m.
    TEMP: [280, 290, 300] K.
    """

    def test_native_z_kind(self, tmp_path):
        path = tmp_path / "hybrid.arl"
        time = pd.Timestamp("2024-07-18 00:00")
        write_hybrid_sampling_file(path, time=time)

        # midpoint between levels 0 and 1 in native index space
        points = pd.DataFrame({"lon": [20.0], "lat": [-10.0], "z": [0.5]})
        with File(path) as arl:
            result = arl.sample_points(points, ["TEMP"], time=time, z_kind="native")

        np.testing.assert_allclose(result["TEMP"].iloc[0], 285.0, atol=1e-4)

    def test_pressure_z_kind(self, tmp_path):
        path = tmp_path / "hybrid.arl"
        time = pd.Timestamp("2024-07-18 00:00")
        write_hybrid_sampling_file(path, time=time)

        # 950 hPa is midpoint between 1000 and 900 hPa → TEMP = 285
        points = pd.DataFrame({"lon": [20.0], "lat": [-10.0], "z": [950.0]})
        with File(path) as arl:
            result = arl.sample_points(
                points, ["TEMP", "pressure"], time=time, z_kind="pressure"
            )

        np.testing.assert_allclose(result["TEMP"].iloc[0], 285.0, atol=1e-4)
        np.testing.assert_allclose(result["pressure"].iloc[0], 950.0, atol=1e-4)

    def test_agl_z_kind(self, tmp_path):
        path = tmp_path / "hybrid.arl"
        time = pd.Timestamp("2024-07-18 00:00")
        write_hybrid_sampling_file(path, time=time)

        # 500m AGL = midpoint between levels 0 (0m) and 1 (1000m) → TEMP = 285
        points = pd.DataFrame({"lon": [20.0], "lat": [-10.0], "z": [500.0]})
        with File(path) as arl:
            result = arl.sample_points(
                points, ["TEMP", "pressure"], time=time, z_kind="agl"
            )

        np.testing.assert_allclose(result["TEMP"].iloc[0], 285.0, atol=1e-4)
        np.testing.assert_allclose(result["pressure"].iloc[0], 950.0, atol=1e-4)

    def test_msl_z_kind(self, tmp_path):
        path = tmp_path / "hybrid.arl"
        time = pd.Timestamp("2024-07-18 00:00")
        write_hybrid_sampling_file(path, time=time)

        # terrain=100m; 600m MSL = 500m AGL → midpoint between levels 0 and 1
        points = pd.DataFrame({"lon": [20.0], "lat": [-10.0], "z": [600.0]})
        with File(path) as arl:
            result = arl.sample_points(
                points, ["TEMP", "pressure"], time=time, z_kind="msl"
            )

        np.testing.assert_allclose(result["TEMP"].iloc[0], 285.0, atol=1e-4)
        np.testing.assert_allclose(result["pressure"].iloc[0], 950.0, atol=1e-4)


def write_sampling_file_with_hgts(path, *, time: pd.Timestamp):
    """
    Pressure-level (flag=2) file with HGTS records.
    levels=[1000, 900, 800] hPa, terrain=200m, PRSS=1000 hPa.
    HGTS=[200, 1200, 2200]m → AGL=[0, 1000, 2000]m.
    TEMP: 280, 290, 300 K at each level.
    """
    grid = make_test_grid()
    vertical_axis = VerticalAxis(flag=2, levels=[1000.0, 900.0, 800.0])
    terrain_data = np.full((grid.ny, grid.nx), 200.0, dtype=np.float32)
    surface_pressure = np.full((grid.ny, grid.nx), 1000.0, dtype=np.float32)
    hgts_values = [200.0, 1200.0, 2200.0]

    with File(
        path, mode="w", source="TEST", grid=grid, vertical_axis=vertical_axis
    ) as arl:
        rs = arl.create_recordset(time)
        rs.create_datarecord("SHGT", level=0, forecast=0, data=terrain_data)
        rs.create_datarecord("PRSS", level=0, forecast=0, data=surface_pressure)
        for level_index, (temp_val, hgts_val) in enumerate(
            zip([280.0, 290.0, 300.0], hgts_values, strict=True)
        ):
            temp = np.full((grid.ny, grid.nx), temp_val, dtype=np.float32)
            hgts = np.full((grid.ny, grid.nx), hgts_val, dtype=np.float32)
            rs.create_datarecord("TEMP", level=level_index, forecast=0, data=temp)
            rs.create_datarecord("HGTS", level=level_index, forecast=0, data=hgts)

    return {
        "grid": grid,
        "vertical_axis": vertical_axis,
        "terrain": terrain_data,
        "hgts": np.array(hgts_values, dtype=np.float32),
        "agl": np.array(hgts_values, dtype=np.float32) - 200.0,
    }


class TestHgtsBasedSampling:
    """
    Pressure-level (flag=2) file with distinct HGTS values.
    HGTS=[200,1200,2200]m, terrain=200m → AGL=[0,1000,2000]m.
    TEMP=[280,290,300]K. Pressure=[1000,900,800]hPa.
    """

    def test_flag2_agl(self, tmp_path):
        path = tmp_path / "hgts.arl"
        time = pd.Timestamp("2024-07-18 00:00")
        write_sampling_file_with_hgts(path, time=time)

        # 500m AGL = midpoint of [0, 1000] → TEMP=285, pressure=950
        points = pd.DataFrame({"lon": [20.0], "lat": [-10.0], "z": [500.0]})
        with File(path) as arl:
            result = arl.sample_points(
                points, ["TEMP", "pressure"], time=time, z_kind="agl"
            )

        np.testing.assert_allclose(result["TEMP"].iloc[0], 285.0, atol=1e-4)
        np.testing.assert_allclose(result["pressure"].iloc[0], 950.0, atol=1e-4)

    def test_flag2_msl_uses_hgts_directly(self, tmp_path):
        path = tmp_path / "hgts.arl"
        time = pd.Timestamp("2024-07-18 00:00")
        write_sampling_file_with_hgts(path, time=time)

        # 700m MSL = midpoint of HGTS [200, 1200] → TEMP=285
        points = pd.DataFrame({"lon": [20.0], "lat": [-10.0], "z": [700.0]})
        with File(path) as arl:
            result = arl.sample_points(
                points, ["TEMP", "pressure"], time=time, z_kind="msl"
            )

        np.testing.assert_allclose(result["TEMP"].iloc[0], 285.0, atol=1e-4)
        np.testing.assert_allclose(result["pressure"].iloc[0], 950.0, atol=1e-4)

    def test_flag2_pressure_virtual_variable_agl(self, tmp_path):
        path = tmp_path / "hgts.arl"
        time = pd.Timestamp("2024-07-18 00:00")
        write_sampling_file_with_hgts(path, time=time)

        # 1000m AGL = exact level 1 in HGTS space → pressure = 900 hPa
        points = pd.DataFrame({"lon": [20.0], "lat": [-10.0], "z": [1000.0]})
        with File(path) as arl:
            result = arl.sample_points(points, ["pressure"], time=time, z_kind="agl")

        np.testing.assert_allclose(result["pressure"].iloc[0], 900.0, atol=1e-4)


class TestTerrainFollowingSampling:
    """
    Terrain-following (flag=3) file.
    Level AGL heights: [0, 100, 500] m.
    TEMP: [280, 290, 300] K. Terrain: 200 m.
    """

    def test_native_z_kind(self, tmp_path):
        path = tmp_path / "terrain.arl"
        time = pd.Timestamp("2024-07-18 00:00")
        write_terrain_sampling_file(path, time=time)

        points = pd.DataFrame({"lon": [20.0], "lat": [-10.0], "z": [0.5]})
        with File(path) as arl:
            result = arl.sample_points(points, ["TEMP"], time=time, z_kind="native")

        np.testing.assert_allclose(result["TEMP"].iloc[0], 285.0, atol=1e-4)

    def test_agl_z_kind_no_prss_needed(self, tmp_path):
        path = tmp_path / "terrain.arl"
        time = pd.Timestamp("2024-07-18 00:00")
        write_terrain_sampling_file(path, time=time)

        # 50m AGL is midpoint between level 0 (0m) and level 1 (100m) → TEMP = 285
        points = pd.DataFrame({"lon": [20.0], "lat": [-10.0], "z": [50.0]})
        with File(path) as arl:
            result = arl.sample_points(points, ["TEMP"], time=time, z_kind="agl")

        np.testing.assert_allclose(result["TEMP"].iloc[0], 285.0, atol=1e-4)

    def test_msl_z_kind(self, tmp_path):
        path = tmp_path / "terrain.arl"
        time = pd.Timestamp("2024-07-18 00:00")
        write_terrain_sampling_file(path, time=time)

        # terrain = 200m; 250m MSL = 50m AGL → midpoint between levels 0 and 1
        points = pd.DataFrame({"lon": [20.0], "lat": [-10.0], "z": [250.0]})
        with File(path) as arl:
            result = arl.sample_points(points, ["TEMP"], time=time, z_kind="msl")

        np.testing.assert_allclose(result["TEMP"].iloc[0], 285.0, atol=1e-4)

    def test_pressure_z_kind_raises(self, tmp_path):
        path = tmp_path / "terrain.arl"
        time = pd.Timestamp("2024-07-18 00:00")
        write_terrain_sampling_file(path, time=time)

        points = pd.DataFrame({"lon": [20.0], "lat": [-10.0], "z": [500.0]})
        with File(path) as arl, pytest.raises(ValueError, match="terrain-following"):
            arl.sample_points(points, ["TEMP"], time=time, z_kind="pressure")


class TestHorizontalInterpolation:
    """Verify bilinear weights and nearest-neighbour snapping."""

    def _write_gradient_file(self, path, time):
        """Single-level file; TEMP[y, x] = x + 10.0 * y. Single level avoids vertical interp."""
        grid = make_test_grid()
        vaxis = VerticalAxis(flag=2, levels=[1000.0])
        yy, xx = np.meshgrid(np.arange(grid.ny), np.arange(grid.nx), indexing="ij")
        temp = (xx + 10.0 * yy).astype(np.float32)
        with File(path, mode="w", source="TEST", grid=grid, vertical_axis=vaxis) as arl:
            rs = arl.create_recordset(time)
            rs.create_datarecord("TEMP", level=0, forecast=0, data=temp)
        return grid, temp

    def test_bilinear_weights_match_manual_calculation(self, tmp_path):
        path = tmp_path / "bilinear.arl"
        time = pd.Timestamp("2024-07-18")
        grid, temp = self._write_gradient_file(path, time)

        test_lon, test_lat = 20.5, -9.5
        x_frac, y_frac = grid.fractional_indices(
            np.array([test_lon]), np.array([test_lat])
        )
        x0 = int(np.floor(x_frac[0]))
        y0 = int(np.floor(y_frac[0]))
        wx = float(x_frac[0] - x0)
        wy = float(y_frac[0] - y0)
        expected = (
            (1 - wx) * (1 - wy) * float(temp[y0, x0])
            + wx * (1 - wy) * float(temp[y0, x0 + 1])
            + (1 - wx) * wy * float(temp[y0 + 1, x0])
            + wx * wy * float(temp[y0 + 1, x0 + 1])
        )

        points = pd.DataFrame({"lon": [test_lon], "lat": [test_lat], "z": [1000.0]})
        with File(path) as arl:
            result = arl.sample_points(points, ["TEMP"], time=time, z_kind="pressure")

        np.testing.assert_allclose(result["TEMP"].iloc[0], expected, atol=1e-3)

    def test_nearest_snaps_to_closest_cell(self, tmp_path):
        path = tmp_path / "nearest.arl"
        time = pd.Timestamp("2024-07-18")
        grid, temp = self._write_gradient_file(path, time)

        test_lon, test_lat = 20.5, -9.5
        x_frac, y_frac = grid.fractional_indices(
            np.array([test_lon]), np.array([test_lat])
        )
        nx_idx = int(np.rint(float(x_frac[0])))
        ny_idx = int(np.rint(float(y_frac[0])))
        expected = float(temp[ny_idx, nx_idx])

        points = pd.DataFrame({"lon": [test_lon], "lat": [test_lat], "z": [1000.0]})
        with File(path) as arl:
            result = arl.sample_points(
                points, ["TEMP"], time=time, z_kind="pressure", method="nearest"
            )

        np.testing.assert_allclose(result["TEMP"].iloc[0], expected, atol=1e-3)

    def test_off_grid_point_returns_nan(self, tmp_path):
        path = tmp_path / "offgrid.arl"
        time = pd.Timestamp("2024-07-18")
        self._write_gradient_file(path, time)

        # lon=0.0 is far outside the test grid (sync_lon=20)
        points = pd.DataFrame({"lon": [0.0], "lat": [-10.0], "z": [1000.0]})
        with File(path) as arl:
            result = arl.sample_points(points, ["TEMP"], time=time, z_kind="pressure")

        assert np.isnan(result["TEMP"].iloc[0])


class TestSamplingErrors:
    """Correct NaN and error behaviour for out-of-range targets and missing fields."""

    def test_z_above_all_levels_native_returns_nan(self, tmp_path):
        path = tmp_path / "range.arl"
        time = pd.Timestamp("2024-07-18")
        write_sampling_file(path, time=time)

        # native level indices are [0, 1, 2]; z=100 is well above the highest
        points = pd.DataFrame({"lon": [20.0], "lat": [-10.0], "z": [100.0]})
        with File(path) as arl:
            result = arl.sample_points(points, ["TEMP"], time=time, z_kind="native")

        assert np.isnan(result["TEMP"].iloc[0])

    def test_z_below_lowest_agl_returns_nan(self, tmp_path):
        path = tmp_path / "range.arl"
        time = pd.Timestamp("2024-07-18")
        write_sampling_file_with_hgts(path, time=time)  # AGL = [0, 1000, 2000] m

        # -100 m AGL is below the lowest level
        points = pd.DataFrame({"lon": [20.0], "lat": [-10.0], "z": [-100.0]})
        with File(path) as arl:
            result = arl.sample_points(points, ["TEMP"], time=time, z_kind="agl")

        assert np.isnan(result["TEMP"].iloc[0])

    def test_missing_hgts_raises_for_agl(self, tmp_path):
        path = tmp_path / "no_hgts.arl"
        time = pd.Timestamp("2024-07-18")
        grid = make_test_grid()
        vaxis = VerticalAxis(flag=2, levels=[1000.0, 900.0, 800.0])
        terrain = np.full((grid.ny, grid.nx), 100.0, dtype=np.float32)
        temp = np.full((grid.ny, grid.nx), 280.0, dtype=np.float32)

        with File(path, mode="w", source="TEST", grid=grid, vertical_axis=vaxis) as arl:
            rs = arl.create_recordset(time)
            rs.create_datarecord("SHGT", level=0, forecast=0, data=terrain)
            for i in range(3):
                rs.create_datarecord("TEMP", level=i, forecast=0, data=temp)
            # intentionally no HGTS records

        points = pd.DataFrame({"lon": [20.0], "lat": [-10.0], "z": [500.0]})
        with File(path) as arl, pytest.raises(ValueError, match="HGTS"):
            arl.sample_points(points, ["TEMP"], time=time, z_kind="agl")

    def test_duplicate_time_across_sources_raises(self, tmp_path):
        path = tmp_path / "dup.arl"
        time = pd.Timestamp("2024-07-18")
        write_sampling_file(path, time=time)

        points = pd.DataFrame(
            {"lon": [20.0], "lat": [-10.0], "z": [950.0], "time": [time]}
        )
        with File(path) as f, pytest.raises(ValueError, match="Multiple sources"):
            sample_points([f, f], points, ["TEMP"], z_kind="pressure")


class TestMultiTimeFile:
    """Sampling a single file that contains multiple recordsets."""

    def test_dispatches_by_time_column(self, tmp_path):
        path = tmp_path / "multi.arl"
        time0 = pd.Timestamp("2024-07-18 00:00")
        time1 = pd.Timestamp("2024-07-18 06:00")

        grid = make_test_grid()
        vaxis = VerticalAxis(flag=2, levels=[1000.0, 900.0, 800.0])

        with File(path, mode="w", source="TEST", grid=grid, vertical_axis=vaxis) as arl:
            for t, base in [(time0, 280.0), (time1, 300.0)]:
                rs = arl.create_recordset(t)
                for i, val in enumerate([base, base + 10.0, base + 20.0]):
                    data = np.full((grid.ny, grid.nx), val, dtype=np.float32)
                    rs.create_datarecord("TEMP", level=i, forecast=0, data=data)

        # 950 hPa sits midway between levels 0 (1000 hPa) and 1 (900 hPa)
        # time0: (280 + 290) / 2 = 285; time1: (300 + 310) / 2 = 305
        points = pd.DataFrame(
            {
                "time": [time0, time1],
                "lon": [20.0, 20.0],
                "lat": [-10.0, -10.0],
                "z": [950.0, 950.0],
            }
        )
        with File(path) as arl:
            result = arl.sample_points(points, ["TEMP"], z_kind="pressure")

        np.testing.assert_allclose(result["TEMP"].to_numpy(), [285.0, 305.0], atol=1e-3)
