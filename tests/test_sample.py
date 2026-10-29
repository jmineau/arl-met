"""Tests for point-sampling helpers."""

import numpy as np
import pandas as pd
import pytest

import arlmet
from arlmet import File, sample_points
from arlmet.grid import Grid, Projection
from arlmet.vertical import (
    HybridAxis,
    PressureAxis,
    SigmaAxis,
    TerrainAxis,
    WrfAxis,
)


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
    vertical_axis = PressureAxis(levels=[1000.0, 900.0, 800.0])
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
    vertical_axis = PressureAxis(levels=levels)
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
        # Level 0 is the ground (0 m AGL, PRSS), as in HYSPLIT's prfprs, whatever
        # its HGTS record says; level 1 is (PRSS - 900) * _DZ above it.
        z1 = (surface_pressure - 900.0) * _DZ
        z_agl_mid = 0.5 * z1
        z_msl_mid = z_agl_mid + terrain
        # Pressure is linear in ln p, as in prfprs: the geometric mean halfway.
        p_mid = np.sqrt(surface_pressure * 900.0)  # 949.39 hPa, not 950.75

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
        np.testing.assert_allclose(agl["pressure"].iloc[0], p_mid, atol=0.01)
        np.testing.assert_allclose(msl["TEMP"].iloc[0], 290.5, atol=0.01)
        np.testing.assert_allclose(msl["pressure"].iloc[0], p_mid, atol=0.01)

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

    def test_module_sample_points_accepts_single_path(self, tmp_path):
        path = tmp_path / "single.arl"
        time = pd.Timestamp("2024-07-18 00:00")
        write_sampling_file(path, time=time, temp_offset=0.0)

        points = pd.DataFrame(
            {"lon": [20.5], "lat": [-9.5], "z": [950.0], "time": [time]}
        )

        # Path is opened and closed internally; no `with` needed.
        result = sample_points(path, points, ["TEMP"], z_kind="pressure")
        np.testing.assert_allclose(result["TEMP"].to_numpy(), [290.5])

    def test_module_sample_points_accepts_sequence_of_paths(self, tmp_path):
        path0 = tmp_path / "p0.arl"
        path1 = tmp_path / "p1.arl"
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

        result = sample_points([path0, path1], points, ["TEMP"], z_kind="pressure")
        np.testing.assert_allclose(result["TEMP"].to_numpy(), [290.5, 310.5])

    def test_module_sample_points_leaves_caller_opened_file_open(self, tmp_path):
        path = tmp_path / "caller.arl"
        time = pd.Timestamp("2024-07-18 00:00")
        write_sampling_file(path, time=time, temp_offset=0.0)

        points = pd.DataFrame(
            {"lon": [20.5], "lat": [-9.5], "z": [950.0], "time": [time]}
        )

        with File(path) as f:
            sample_points(f, points, ["TEMP"], z_kind="pressure")
            # The caller still owns f and it remains usable afterward.
            again = sample_points(f, points, ["TEMP"], z_kind="pressure")
        np.testing.assert_allclose(again["TEMP"].to_numpy(), [290.5])

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
        # ln p in float32 is within a few units in the last place of 1000 hPa.
        np.testing.assert_allclose(sampled["pressure"].iloc[0], 1000.0, rtol=1e-6)
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
    No HGTS and no PRES. Heights derived via hypsometric.
    T02M = 280 K at the ground; TEMP at levels 1 and 2: 290, 300 K (uniform grid).
    terrain (SHGT) = 100 m.
    """
    grid = make_test_grid()
    vertical_axis = SigmaAxis(levels=[1.0, 0.9, 0.8], offset=0.0)
    terrain_data = np.full((grid.ny, grid.nx), 100.0, dtype=np.float32)
    surface_pressure = np.full((grid.ny, grid.nx), 1000.0, dtype=np.float32)

    with File(
        path, mode="w", source="TEST", grid=grid, vertical_axis=vertical_axis
    ) as arl:
        rs = arl.create_recordset(time)
        rs.create_datarecord("SHGT", level=0, forecast=0, data=terrain_data)
        rs.create_datarecord("PRSS", level=0, forecast=0, data=surface_pressure)
        t02m = np.full((grid.ny, grid.nx), 280.0, dtype=np.float32)
        rs.create_datarecord("T02M", level=0, forecast=0, data=t02m)
        for level_index, temp_val in [(1, 290.0), (2, 300.0)]:
            temp = np.full((grid.ny, grid.nx), temp_val, dtype=np.float32)
            rs.create_datarecord("TEMP", level=level_index, forecast=0, data=temp)

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
    vertical_axis = TerrainAxis(levels=[0.0, 100.0, 500.0])
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


_RD_OVER_G = 287.05 / 9.80665  # arl-met's R_d / g, m/K


def _ground_up_agl(pressures, temps):
    """
    Heights AGL by hand, from the ground up: each layer is (R_d/g) * mean T * ln(p_below/p_above).

    *pressures* and *temps* start at the ground (PRSS and T02M) and are dry, so T is
    the virtual temperature.
    """
    z = [0.0]
    for k in range(1, len(pressures)):
        t_mean = 0.5 * (temps[k - 1] + temps[k])
        z.append(z[-1] + _RD_OVER_G * t_mean * np.log(pressures[k - 1] / pressures[k]))
    return np.array(z)


# The sigma and hybrid fixtures: ground 1000 hPa and 280 K, then 900 hPa 290 K, 800 hPa 300 K.
_FIXTURE_AGL = _ground_up_agl([1000.0, 900.0, 800.0], [280.0, 290.0, 300.0])


class TestSigmaSampling:
    """
    Sigma (flag=1) file: sigma=[1.0, 0.9, 0.8], PRSS=1000 hPa, p_top=0.

    Effective pressure levels: [1000, 900, 800] hPa.
    Heights derived via hypsometric integration from PRSS + TEMP.
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

        # Halfway between the ground (T02M 280 K) and level 1 (290 K)
        agl = _FIXTURE_AGL
        z_mid = 0.5 * (agl[0] + agl[1])

        points = pd.DataFrame({"lon": [20.0], "lat": [-10.0], "z": [float(z_mid)]})
        with File(path) as arl:
            result = arl.sample_points(points, ["TEMP"], time=time, z_kind="agl")

        np.testing.assert_allclose(result["TEMP"].iloc[0], 285.0, atol=1e-3)

    def test_msl_z_kind(self, tmp_path):
        path = tmp_path / "sigma.arl"
        time = pd.Timestamp("2024-07-18 00:00")
        write_sigma_sampling_file(path, time=time)

        # terrain=100m; MSL = AGL + terrain
        agl = _FIXTURE_AGL
        z_mid_msl = 0.5 * (agl[0] + agl[1]) + 100.0

        points = pd.DataFrame({"lon": [20.0], "lat": [-10.0], "z": [float(z_mid_msl)]})
        with File(path) as arl:
            result = arl.sample_points(points, ["TEMP"], time=time, z_kind="msl")

        np.testing.assert_allclose(result["TEMP"].iloc[0], 285.0, atol=1e-3)


def write_hybrid_sampling_file(path, *, time: pd.Timestamp):
    """
    Hybrid (flag=4) file with pure-sigma levels (floor_p=0 for all).

    levels=[0.995, 0.9, 0.8], PRSS=1000 hPa.
    Pressure: level 0 → 1000 hPa (surface override), level 1 → 900 hPa, level 2 → 800 hPa.
    No HGTS — matches real hybrid files. Heights derived via hypsometric.
    T02M = 280 K at the ground; TEMP at levels 1 and 2: 290, 300 K (uniform grid).
    terrain (SHGT) = 100 m.
    """
    grid = make_test_grid()
    vertical_axis = HybridAxis(levels=[0.995, 0.9, 0.8])
    terrain_data = np.full((grid.ny, grid.nx), 100.0, dtype=np.float32)
    surface_pressure = np.full((grid.ny, grid.nx), 1000.0, dtype=np.float32)

    with File(
        path, mode="w", source="TEST", grid=grid, vertical_axis=vertical_axis
    ) as arl:
        rs = arl.create_recordset(time)
        rs.create_datarecord("SHGT", level=0, forecast=0, data=terrain_data)
        rs.create_datarecord("PRSS", level=0, forecast=0, data=surface_pressure)
        t02m = np.full((grid.ny, grid.nx), 280.0, dtype=np.float32)
        rs.create_datarecord("T02M", level=0, forecast=0, data=t02m)
        for level_index, temp_val in [(1, 290.0), (2, 300.0)]:
            temp = np.full((grid.ny, grid.nx), temp_val, dtype=np.float32)
            rs.create_datarecord("TEMP", level=level_index, forecast=0, data=temp)

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
    Heights derived via hypsometric integration.
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

        agl = _FIXTURE_AGL
        z_mid = 0.5 * (agl[0] + agl[1])

        points = pd.DataFrame({"lon": [20.0], "lat": [-10.0], "z": [float(z_mid)]})
        with File(path) as arl:
            result = arl.sample_points(points, ["TEMP"], time=time, z_kind="agl")

        np.testing.assert_allclose(result["TEMP"].iloc[0], 285.0, atol=1e-3)

    def test_msl_z_kind(self, tmp_path):
        path = tmp_path / "hybrid.arl"
        time = pd.Timestamp("2024-07-18 00:00")
        write_hybrid_sampling_file(path, time=time)

        agl = _FIXTURE_AGL
        z_mid_msl = 0.5 * (agl[0] + agl[1]) + 100.0  # terrain = 100m

        points = pd.DataFrame({"lon": [20.0], "lat": [-10.0], "z": [float(z_mid_msl)]})
        with File(path) as arl:
            result = arl.sample_points(points, ["TEMP"], time=time, z_kind="msl")

        np.testing.assert_allclose(result["TEMP"].iloc[0], 285.0, atol=1e-3)


def write_sampling_file_with_hgts(path, *, time: pd.Timestamp):
    """
    Pressure-level (flag=2) file with HGTS records.

    levels=[1000, 900, 800] hPa, terrain=200m, PRSS=1000 hPa.
    HGTS=[200, 1200, 2200]m → AGL=[0, 1000, 2000]m.
    TEMP: 280, 290, 300 K at each level.
    """
    grid = make_test_grid()
    vertical_axis = PressureAxis(levels=[1000.0, 900.0, 800.0])
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

        # 500m AGL = midpoint of [0, 1000] → TEMP=285; pressure is linear in
        # ln p, as in HYSPLIT's prfprs: sqrt(1000 * 900) = 948.68, not 950.
        points = pd.DataFrame({"lon": [20.0], "lat": [-10.0], "z": [500.0]})
        with File(path) as arl:
            result = arl.sample_points(
                points, ["TEMP", "pressure"], time=time, z_kind="agl"
            )

        np.testing.assert_allclose(result["TEMP"].iloc[0], 285.0, atol=1e-4)
        np.testing.assert_allclose(
            result["pressure"].iloc[0], np.sqrt(1000.0 * 900.0), rtol=1e-6
        )

    def test_flag2_msl_uses_hgts_directly(self, tmp_path):
        path = tmp_path / "hgts.arl"
        time = pd.Timestamp("2024-07-18 00:00")
        write_sampling_file_with_hgts(path, time=time)

        # 700m MSL = midpoint of HGTS [200, 1200] → TEMP=285, pressure (ln p)
        # the geometric mean of 1000 and 900 hPa
        points = pd.DataFrame({"lon": [20.0], "lat": [-10.0], "z": [700.0]})
        with File(path) as arl:
            result = arl.sample_points(
                points, ["TEMP", "pressure"], time=time, z_kind="msl"
            )

        np.testing.assert_allclose(result["TEMP"].iloc[0], 285.0, atol=1e-4)
        np.testing.assert_allclose(
            result["pressure"].iloc[0], np.sqrt(1000.0 * 900.0), rtol=1e-6
        )

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


class TestFlag2RequiresHgts:
    """Pressure-level (flag=2) files require HGTS for AGL/MSL sampling."""

    def test_missing_hgts_raises_for_agl(self, tmp_path):
        path = tmp_path / "no_hgts.arl"
        time = pd.Timestamp("2024-07-18")
        grid = make_test_grid()
        vaxis = PressureAxis(levels=[1000.0, 900.0, 800.0])
        terrain = np.full((grid.ny, grid.nx), 100.0, dtype=np.float32)
        temp = np.full((grid.ny, grid.nx), 280.0, dtype=np.float32)

        with File(path, mode="w", source="TEST", grid=grid, vertical_axis=vaxis) as arl:
            rs = arl.create_recordset(time)
            rs.create_datarecord("SHGT", level=0, forecast=0, data=terrain)
            for i in range(3):
                rs.create_datarecord("TEMP", level=i, forecast=0, data=temp)

        points = pd.DataFrame({"lon": [20.0], "lat": [-10.0], "z": [500.0]})
        with File(path) as arl, pytest.raises(ValueError, match="HGTS"):
            arl.sample_points(points, ["TEMP"], time=time, z_kind="agl")

    def test_missing_hgts_raises_for_msl(self, tmp_path):
        path = tmp_path / "no_hgts.arl"
        time = pd.Timestamp("2024-07-18")
        grid = make_test_grid()
        vaxis = PressureAxis(levels=[1000.0, 900.0, 800.0])
        terrain = np.full((grid.ny, grid.nx), 100.0, dtype=np.float32)
        temp = np.full((grid.ny, grid.nx), 280.0, dtype=np.float32)

        with File(path, mode="w", source="TEST", grid=grid, vertical_axis=vaxis) as arl:
            rs = arl.create_recordset(time)
            rs.create_datarecord("SHGT", level=0, forecast=0, data=terrain)
            for i in range(3):
                rs.create_datarecord("TEMP", level=i, forecast=0, data=temp)

        points = pd.DataFrame({"lon": [20.0], "lat": [-10.0], "z": [500.0]})
        with File(path) as arl, pytest.raises(ValueError, match="HGTS"):
            arl.sample_points(points, ["TEMP"], time=time, z_kind="msl")


class TestTerrainRequired:
    """A file without SHGT raises the documented ValueError where terrain is needed (#107)."""

    def test_pressure_levels_at_heights(self, tmp_path):
        path = tmp_path / "no_shgt.arl"
        time = pd.Timestamp("2024-07-18")
        grid = make_test_grid()
        vaxis = PressureAxis(levels=[0.0, 1000.0, 900.0])
        field = np.full((grid.ny, grid.nx), 280.0, dtype=np.float32)

        with File(path, mode="w", source="TEST", grid=grid, vertical_axis=vaxis) as arl:
            rs = arl.create_recordset(time)
            rs.create_datarecord("PRSS", level=0, forecast=0, data=field + 700.0)
            for i in (1, 2):
                rs.create_datarecord("HGTS", level=i, forecast=0, data=field * i)
                rs.create_datarecord("TEMP", level=i, forecast=0, data=field)

        points = pd.DataFrame({"lon": [20.0], "lat": [-10.0], "z": [500.0]})
        for z_kind in ("agl", "msl"):
            with File(path) as arl, pytest.raises(ValueError, match="SHGT"):
                arl.sample_points(points, ["TEMP"], time=time, z_kind=z_kind)

    def test_sigma_at_msl(self, tmp_path):
        path = tmp_path / "no_shgt.arl"
        time = pd.Timestamp("2024-07-18")
        grid = make_test_grid()
        vaxis = SigmaAxis(levels=[1.0, 0.99, 0.95])
        field = np.full((grid.ny, grid.nx), 280.0, dtype=np.float32)

        with File(path, mode="w", source="TEST", grid=grid, vertical_axis=vaxis) as arl:
            rs = arl.create_recordset(time)
            rs.create_datarecord("PRSS", level=0, forecast=0, data=field + 570.0)
            for i in (1, 2):
                rs.create_datarecord("TEMP", level=i, forecast=0, data=field)

        points = pd.DataFrame({"lon": [20.0], "lat": [-10.0], "z": [1500.0]})
        with File(path) as arl, pytest.raises(ValueError, match="SHGT"):
            arl.sample_points(points, ["TEMP"], time=time, z_kind="msl")


class TestSigmaHypsometricSampling:
    """Sigma/hybrid files use hypsometric integration for AGL/MSL (no HGTS)."""

    def test_sigma_agl_matches_hypsometric(self, tmp_path):
        path = tmp_path / "sigma.arl"
        time = pd.Timestamp("2024-07-18 00:00")
        write_sigma_sampling_file(path, time=time)

        agl = _FIXTURE_AGL

        # Sample at the second level's AGL height → should return that level's TEMP
        points = pd.DataFrame({"lon": [20.0], "lat": [-10.0], "z": [float(agl[1])]})
        with File(path) as arl:
            result = arl.sample_points(points, ["TEMP"], time=time, z_kind="agl")

        np.testing.assert_allclose(result["TEMP"].iloc[0], 290.0, atol=1e-3)

    def test_sigma_msl_matches_hypsometric_plus_terrain(self, tmp_path):
        path = tmp_path / "sigma.arl"
        time = pd.Timestamp("2024-07-18 00:00")
        write_sigma_sampling_file(path, time=time)

        agl = _FIXTURE_AGL
        # terrain = 100m
        z_msl = float(agl[1]) + 100.0

        points = pd.DataFrame({"lon": [20.0], "lat": [-10.0], "z": [z_msl]})
        with File(path) as arl:
            result = arl.sample_points(points, ["TEMP"], time=time, z_kind="msl")

        np.testing.assert_allclose(result["TEMP"].iloc[0], 290.0, atol=1e-3)


# --- sigma files with PRES, humidity and a 2 m temperature, as HRRR's (HYSPLIT's prfsig) ---

_COLUMN_TIME = pd.Timestamp("2021-01-15 03:00")
# A point on the fixture grid's first cell; every field is uniform.
_COLUMN_POINT = {"lon": 20.0, "lat": -10.0}
_PRSS, _T02M, _SHGT, _U10M = 850.0, 285.0, 1300.0, 2.0
# Levels 1 to 3. PRES differs from the nominal sigma * PRSS (841.5, 807.5, 765).
_PRES = [838.0, 800.0, 760.0]
_TEMP = [283.0, 280.0, 277.0]
_SPHU = [0.008, 0.006, 0.004]
_RELH = [80.0, 60.0, 40.0]
_UWND = [5.0, 6.0, 7.0]
_WWND = [0.1, 0.2, 0.3]


def write_column_file(
    path,
    *,
    axis=None,
    pres=True,
    humidity: str | None = "SPHU",
    t02m=True,
):
    """A sigma file shaped like HRRR's: surface fields at level 0, PRES/TEMP/... on 1-3."""
    grid = make_test_grid()
    axis = axis if axis is not None else SigmaAxis(levels=[1.0, 0.99, 0.95, 0.9])

    def field(value):
        return np.full((grid.ny, grid.nx), value, dtype=np.float32)

    with File(path, mode="w", source="TEST", grid=grid, vertical_axis=axis) as arl:
        rs = arl.create_recordset(_COLUMN_TIME)
        rs.create_datarecord("SHGT", level=0, forecast=0, data=field(_SHGT))
        rs.create_datarecord("PRSS", level=0, forecast=0, data=field(_PRSS))
        if t02m:
            rs.create_datarecord("T02M", level=0, forecast=0, data=field(_T02M))
        rs.create_datarecord("U10M", level=0, forecast=0, data=field(_U10M))
        for k in range(3):
            level = k + 1
            rs.create_datarecord("TEMP", level=level, forecast=0, data=field(_TEMP[k]))
            if pres:
                rs.create_datarecord(
                    "PRES", level=level, forecast=0, data=field(_PRES[k])
                )
            if humidity == "SPHU":
                rs.create_datarecord(
                    "SPHU", level=level, forecast=0, data=field(_SPHU[k])
                )
            elif humidity == "RELH":
                rs.create_datarecord(
                    "RELH", level=level, forecast=0, data=field(_RELH[k])
                )
            rs.create_datarecord("UWND", level=level, forecast=0, data=field(_UWND[k]))
            rs.create_datarecord("WWND", level=level, forecast=0, data=field(_WWND[k]))


def _sample_column(path, variables, z, z_kind="agl"):
    points = pd.DataFrame(
        {**{k: [v] * len(z) for k, v in _COLUMN_POINT.items()}, "z": z}
    )
    return sample_points(path, points, variables, z_kind=z_kind)


def _esat(t):
    """HYSPLIT's CALCESAT, hPa."""
    return np.exp(21.4 - 5351.0 / t)


class TestSigmaColumnAsHysplit:
    """
    Sigma heights as HYSPLIT's prfsig finds them (#66).

    From the ground (PRSS, T02M) through PRES with the virtual temperature;
    below level 1 values come from the ground.
    """

    # By hand: virtual temperatures T (1 + 0.61 q), the ground's with level 1's q.
    TV = [_T02M * (1 + 0.61 * _SPHU[0])] + [
        t * (1 + 0.61 * q) for t, q in zip(_TEMP, _SPHU, strict=True)
    ]
    Z = _ground_up_agl([_PRSS, *_PRES], TV)

    def test_heights_come_from_pres_with_virtual_temperature(self, tmp_path):
        path = tmp_path / "column.arl"
        write_column_file(path)
        # At each level's height, TEMP is that level's value (just below the top one).
        result = _sample_column(
            path, ["TEMP"], [self.Z[1], self.Z[2], self.Z[3] - 1e-6]
        )
        np.testing.assert_allclose(result["TEMP"], _TEMP, atol=1e-3)

    def test_msl_adds_terrain(self, tmp_path):
        path = tmp_path / "column.arl"
        write_column_file(path)
        result = _sample_column(path, ["TEMP"], [self.Z[2] + _SHGT], z_kind="msl")
        np.testing.assert_allclose(result["TEMP"], [_TEMP[1]], atol=1e-3)

    def test_relative_humidity_gives_virtual_temperature(self, tmp_path):
        # HYSPLIT's CALCSP: q = 0.622 * (RH/100 * esat(T)) / p, with the level's PRES;
        # the ground uses its own T and PRSS with level 1's RH.
        path = tmp_path / "column.arl"
        write_column_file(path, humidity="RELH")
        q = [
            0.622 * (rh / 100 * _esat(t)) / p
            for rh, t, p in zip(_RELH, _TEMP, _PRES, strict=True)
        ]
        q0 = 0.622 * (_RELH[0] / 100 * _esat(_T02M)) / _PRSS
        tv = [_T02M * (1 + 0.61 * q0)] + [
            t * (1 + 0.61 * qk) for t, qk in zip(_TEMP, q, strict=True)
        ]
        z = _ground_up_agl([_PRSS, *_PRES], tv)
        result = _sample_column(path, ["TEMP", "pressure"], [z[1], z[2]])
        np.testing.assert_allclose(result["TEMP"], [_TEMP[0], _TEMP[1]], atol=1e-3)
        # The ground's q (8.1 g/kg, not level 1's 7.2) moves every height by 3 cm,
        # which the pressure at a level's height shows (0.003 hPa).
        np.testing.assert_allclose(result["pressure"], _PRES[:2], rtol=1e-6)

    def test_without_humidity_heights_are_dry(self, tmp_path):
        path = tmp_path / "column.arl"
        write_column_file(path, humidity=None)
        z = _ground_up_agl([_PRSS, *_PRES], [_T02M, *_TEMP])
        result = _sample_column(path, ["TEMP"], [z[1], z[2]])
        np.testing.assert_allclose(result["TEMP"], [_TEMP[0], _TEMP[1]], atol=1e-3)

    def test_without_t02m_the_ground_is_level_one_brought_down_adiabatically(
        self, tmp_path
    ):
        # HYSPLIT's PRFSFCT: T_ground = T1 * sigma1 ** -0.286.
        path = tmp_path / "column.arl"
        write_column_file(path, t02m=False)
        t_ground = _TEMP[0] * 0.99**-0.286
        tv = [t_ground * (1 + 0.61 * _SPHU[0])] + self.TV[1:]
        z = _ground_up_agl([_PRSS, *_PRES], tv)
        result = _sample_column(path, ["TEMP"], [0.0, z[1], z[2]])
        np.testing.assert_allclose(
            result["TEMP"], [t_ground, _TEMP[0], _TEMP[1]], atol=1e-3
        )

    def test_below_level_one_values_come_from_the_ground(self, tmp_path):
        # Halfway between the ground and level 1: TEMP from T02M, PRES (as ln p) from
        # PRSS, UWND from U10M, WWND from 0, SPHU stays at level 1's value.
        path = tmp_path / "column.arl"
        write_column_file(path)
        variables = ["TEMP", "PRES", "pressure", "UWND", "WWND", "SPHU"]
        result = _sample_column(path, variables, [0.5 * self.Z[1], 0.0]).iloc[0]
        p_half = np.exp(0.5 * (np.log(_PRSS) + np.log(_PRES[0])))
        np.testing.assert_allclose(result["TEMP"], 0.5 * (_T02M + _TEMP[0]), atol=1e-3)
        np.testing.assert_allclose(result["PRES"], p_half, rtol=1e-6)
        np.testing.assert_allclose(result["pressure"], p_half, rtol=1e-6)
        np.testing.assert_allclose(result["UWND"], 0.5 * (_U10M + _UWND[0]), atol=1e-4)
        np.testing.assert_allclose(result["WWND"], 0.5 * _WWND[0], atol=1e-5)
        np.testing.assert_allclose(result["SPHU"], _SPHU[0], atol=1e-6)

    def test_at_the_ground_values_are_the_surface_fields(self, tmp_path):
        path = tmp_path / "column.arl"
        write_column_file(path)
        result = _sample_column(path, ["TEMP", "PRES", "pressure"], [0.0]).iloc[0]
        np.testing.assert_allclose(
            [result["TEMP"], result["PRES"], result["pressure"]],
            [_T02M, _PRSS, _PRSS],
            rtol=1e-6,
        )

    def test_below_the_ground_is_nan(self, tmp_path):
        path = tmp_path / "column.arl"
        write_column_file(path)
        result = _sample_column(path, ["TEMP", "pressure"], [-1.0]).iloc[0]
        assert np.isnan(result["TEMP"]) and np.isnan(result["pressure"])

    def test_pres_is_interpolated_as_ln_p_between_levels(self, tmp_path):
        path = tmp_path / "column.arl"
        write_column_file(path)
        z_mid = 0.5 * (self.Z[2] + self.Z[3])
        result = _sample_column(path, ["PRES", "TEMP"], [z_mid]).iloc[0]
        np.testing.assert_allclose(
            result["PRES"], np.sqrt(_PRES[1] * _PRES[2]), rtol=1e-6
        )
        np.testing.assert_allclose(
            result["TEMP"], 0.5 * (_TEMP[1] + _TEMP[2]), atol=1e-3
        )

    def test_virtual_pressure_at_heights_is_pres(self, tmp_path):
        # #64/#65: the virtual "pressure" was NaN at AGL/MSL heights on sigma files.
        path = tmp_path / "column.arl"
        write_column_file(path)
        z_mid = 0.5 * (self.Z[1] + self.Z[2])
        expected = [_PRES[0], np.sqrt(_PRES[0] * _PRES[1]), _PRES[2]]
        agl = _sample_column(path, ["pressure"], [self.Z[1], z_mid, self.Z[3] - 1e-6])
        msl = _sample_column(
            path,
            ["pressure"],
            [self.Z[1] + _SHGT, z_mid + _SHGT, self.Z[3] - 1e-6 + _SHGT],
            z_kind="msl",
        )
        np.testing.assert_allclose(agl["pressure"], expected, rtol=1e-6)
        np.testing.assert_allclose(msl["pressure"], expected, rtol=1e-6)

    def test_virtual_pressure_at_native_levels(self, tmp_path):
        # #64: z_kind="native" raised for the virtual pressure on sigma files.
        path = tmp_path / "column.arl"
        write_column_file(path)
        result = _sample_column(path, ["pressure"], [0.0, 1.0, 1.5], z_kind="native")
        np.testing.assert_allclose(
            result["pressure"],
            [_PRSS, _PRES[0], np.sqrt(_PRES[0] * _PRES[1])],
            rtol=1e-6,
        )

    def test_pressure_targets_use_pres(self, tmp_path):
        # Halfway (linear in p) between PRES of levels 1 and 2, TEMP is halfway too.
        path = tmp_path / "column.arl"
        write_column_file(path)
        result = _sample_column(
            path, ["TEMP"], [0.5 * (_PRES[0] + _PRES[1])], z_kind="pressure"
        )
        np.testing.assert_allclose(
            result["TEMP"], [0.5 * (_TEMP[0] + _TEMP[1])], atol=1e-3
        )

    def test_without_pres_heights_use_the_nominal_sigma_pressure(self, tmp_path):
        path = tmp_path / "column.arl"
        write_column_file(path, pres=False)
        nominal = [_PRSS * s for s in (0.99, 0.95, 0.9)]
        z = _ground_up_agl([_PRSS, *nominal], self.TV)
        result = _sample_column(path, ["TEMP", "pressure"], [z[1], z[2]])
        np.testing.assert_allclose(result["TEMP"], [_TEMP[0], _TEMP[1]], atol=1e-3)
        np.testing.assert_allclose(
            result["pressure"], [nominal[0], nominal[1]], rtol=1e-6
        )

    def test_hybrid_ignores_pres(self, tmp_path):
        # HYSPLIT's prfecm takes a hybrid level's pressure from the levels, not PRES.
        path = tmp_path / "hybrid.arl"
        write_column_file(path, axis=HybridAxis(levels=[1.0, 0.99, 0.95, 0.9]))
        nominal = [_PRSS * s for s in (0.99, 0.95, 0.9)]
        z = _ground_up_agl([_PRSS, *nominal], self.TV)
        result = _sample_column(path, ["TEMP", "pressure"], [z[1], z[2]])
        np.testing.assert_allclose(result["TEMP"], [_TEMP[0], _TEMP[1]], atol=1e-3)
        np.testing.assert_allclose(
            result["pressure"], [nominal[0], nominal[1]], rtol=1e-6
        )

    def test_hybrid_ground_temperature_uses_level_one_over_prss(self, tmp_path):
        # prfecm brings level 1 down with its pressure over PRSS (ESG(1)), here
        # (floor 100 hPa + 0.8 * 850) / 850.
        path = tmp_path / "hybrid.arl"
        write_column_file(
            path, axis=HybridAxis(levels=[1.0, 100.8, 100.7, 100.6]), t02m=False
        )
        p1 = 100.0 + 0.8 * _PRSS
        t_ground = _TEMP[0] * (p1 / _PRSS) ** -0.286
        result = _sample_column(path, ["TEMP"], [0.0])
        np.testing.assert_allclose(result["TEMP"], [t_ground], atol=1e-3)


#: A WRF (flag 5) axis: eta levels, model top 50 hPa. Without WRF's hybrid
#: coefficients HYSPLIT's metini sends such a file to prfsig, as a sigma file.
_WRF_AXIS = WrfAxis(levels=[1.0, 0.99, 0.95, 0.9], offset=50.0)


class TestWrfColumnAsHysplit:
    """WRF heights as HYSPLIT's prfsig finds them: the sigma recipe, with PRES."""

    def test_samples_as_a_sigma_file_with_pres(self, tmp_path):
        # The same fields as a sigma file: every value at every height agrees,
        # below level 1 (from the ground values) and between levels (ln p).
        sigma, wrf = tmp_path / "sigma.arl", tmp_path / "wrf.arl"
        write_column_file(sigma)
        write_column_file(wrf, axis=_WRF_AXIS)
        variables = ["TEMP", "PRES", "pressure", "UWND", "WWND", "SPHU"]
        z = TestSigmaColumnAsHysplit.Z
        heights = [0.0, 0.5 * z[1], z[1], 0.5 * (z[2] + z[3]), z[3] - 1e-6]
        for z_kind, offset in (("agl", 0.0), ("msl", _SHGT)):
            targets = [h + offset for h in heights]
            pd.testing.assert_frame_equal(
                _sample_column(wrf, variables, targets, z_kind=z_kind),
                _sample_column(sigma, variables, targets, z_kind=z_kind),
            )
        result = _sample_column(wrf, ["TEMP"], [z[1], z[2], z[3] - 1e-6])
        np.testing.assert_allclose(result["TEMP"], _TEMP, atol=1e-3)

    def test_ground_and_pres_at_pressures_and_native_levels(self, tmp_path):
        path = tmp_path / "wrf.arl"
        write_column_file(path, axis=_WRF_AXIS)
        by_p = _sample_column(path, ["TEMP"], [0.5 * (_PRSS + _PRES[0])], "pressure")
        np.testing.assert_allclose(by_p["TEMP"], [0.5 * (_T02M + _TEMP[0])], atol=1e-3)
        native = _sample_column(path, ["pressure"], [0.0, 0.5, 2.0], z_kind="native")
        np.testing.assert_allclose(
            native["pressure"],
            [_PRSS, np.sqrt(_PRSS * _PRES[0]), _PRES[1]],
            rtol=1e-6,
        )

    def test_without_t02m_the_ground_comes_from_level_one_eta(self, tmp_path):
        # PRFSFCT with PSG(1), the first level's eta: T_ground = T1 * 0.99 ** -0.286.
        path = tmp_path / "wrf.arl"
        write_column_file(path, axis=_WRF_AXIS, t02m=False)
        t_ground = _TEMP[0] * 0.99**-0.286
        tv = [t_ground * (1 + 0.61 * _SPHU[0])] + TestSigmaColumnAsHysplit.TV[1:]
        z = _ground_up_agl([_PRSS, *_PRES], tv)
        result = _sample_column(path, ["TEMP"], [0.0, z[1], z[2]])
        np.testing.assert_allclose(
            result["TEMP"], [t_ground, _TEMP[0], _TEMP[1]], atol=1e-3
        )

    def test_z_agl_matches_sample_points(self, tmp_path):
        path = tmp_path / "wrf.arl"
        write_column_file(path, axis=_WRF_AXIS)
        ds = arlmet.open_dataset(path)
        z = arlmet.z_agl(ds).isel(time=0, lat=0, lon=0).values
        np.testing.assert_allclose(z, TestSigmaColumnAsHysplit.Z[1:], atol=1e-3)


# --- pressure-level files over high ground, as NAM12's over the mountain West (prfprs) ---

# Levels 1 to 5 (hPa) and their HGTS (m MSL). With the ground at _SHGT = 1300 m and
# _PRSS = 850 hPa, 1000 and 950 hPa are below the ground, extrapolated by the model.
_PRS_LEVELS = [1000.0, 950.0, 900.0, 850.0, 800.0]
_PRS_HGTS = [100.0, 560.0, 1010.0, 1460.0, 1950.0]
_PRS_TEMP = [306.0, 302.0, 298.0, 290.0, 286.0]
_PRS_UWND = [-3.0, -2.5, -2.0, 4.0, 6.0]
_PRS_WWND = [0.0, 0.0, 0.0, 0.2, 0.3]
#: Height above ground of the first level above it (850 hPa).
_PRS_Z1 = _PRS_HGTS[3] - _SHGT


def write_pressure_column_file(path, *, t02m=True, u10m=True, shgt=_SHGT):
    """A pressure-level file: surface fields at level 0, HGTS/TEMP/UWND/WWND on 1-5."""
    grid = make_test_grid()
    axis = PressureAxis(levels=[0.0, *_PRS_LEVELS])

    def field(value):
        return np.full((grid.ny, grid.nx), value, dtype=np.float32)

    with File(path, mode="w", source="TEST", grid=grid, vertical_axis=axis) as arl:
        rs = arl.create_recordset(_COLUMN_TIME)
        rs.create_datarecord("SHGT", level=0, forecast=0, data=field(shgt))
        rs.create_datarecord("PRSS", level=0, forecast=0, data=field(_PRSS))
        if t02m:
            rs.create_datarecord("T02M", level=0, forecast=0, data=field(_T02M))
        if u10m:
            rs.create_datarecord("U10M", level=0, forecast=0, data=field(_U10M))
        for k in range(len(_PRS_LEVELS)):
            level = k + 1
            for name, values in (
                ("HGTS", _PRS_HGTS),
                ("TEMP", _PRS_TEMP),
                ("UWND", _PRS_UWND),
                ("WWND", _PRS_WWND),
            ):
                rs.create_datarecord(
                    name, level=level, forecast=0, data=field(values[k])
                )


class TestPressureColumnAsHysplit:
    """
    Pressure-level heights as HYSPLIT's prfprs finds them (#81).

    The profile starts at the ground (0 m AGL; T02M, U10M, PRSS) and skips the
    levels at or below it; between the ground and the first level above it,
    values are linear in height.
    """

    def test_below_the_first_level_values_come_from_the_ground(self, tmp_path):
        path = tmp_path / "prs.arl"
        write_pressure_column_file(path)
        variables = ["TEMP", "UWND", "WWND", "HGTS", "pressure"]
        z = [10.0, 0.5 * _PRS_Z1]
        result = _sample_column(path, variables, z)
        frac = np.array(z) / _PRS_Z1
        np.testing.assert_allclose(
            result["TEMP"], _T02M + frac * (_PRS_TEMP[3] - _T02M), atol=1e-3
        )
        np.testing.assert_allclose(
            result["UWND"], _U10M + frac * (_PRS_UWND[3] - _U10M), atol=1e-4
        )
        np.testing.assert_allclose(result["WWND"], frac * _PRS_WWND[3], atol=1e-5)
        np.testing.assert_allclose(result["HGTS"], _SHGT + np.array(z), atol=1e-2)
        # PRSS and the first level above the ground are both 850 hPa here.
        np.testing.assert_allclose(result["pressure"], [_PRSS, _PRSS], rtol=1e-6)

    def test_pressure_is_interpolated_as_ln_p(self, tmp_path):
        # prfprs: ln p is linear in height between two levels (850 hPa at 160 m
        # AGL, 800 hPa at 650 m), so halfway it is the geometric mean, 824.62
        # hPa, not 825; at the levels it is the stored pressure.
        path = tmp_path / "prs.arl"
        write_pressure_column_file(path)
        z4, z5 = _PRS_HGTS[3] - _SHGT, _PRS_HGTS[4] - _SHGT
        result = _sample_column(path, ["pressure"], [z4, 0.5 * (z4 + z5), z5])
        np.testing.assert_allclose(
            result["pressure"],
            [_PRS_LEVELS[3], np.sqrt(_PRS_LEVELS[3] * _PRS_LEVELS[4]), _PRS_LEVELS[4]],
            rtol=1e-6,
        )
        msl = _sample_column(path, ["pressure"], [0.5 * (z4 + z5) + _SHGT], "msl")
        np.testing.assert_allclose(
            msl["pressure"], [np.sqrt(_PRS_LEVELS[3] * _PRS_LEVELS[4])], rtol=1e-6
        )

    def test_levels_below_the_ground_are_not_used(self, tmp_path):
        # The 1000 and 950 hPa values (306 K, 302 K) never reach the profile.
        path = tmp_path / "prs.arl"
        write_pressure_column_file(path)
        result = _sample_column(path, ["TEMP"], [0.0, 1.0, _PRS_Z1])
        assert result["TEMP"].max() <= _PRS_TEMP[3] + 1e-3
        np.testing.assert_allclose(result["TEMP"].iloc[[0, 2]], [_T02M, _PRS_TEMP[3]])

    def test_msl_starts_at_the_terrain(self, tmp_path):
        path = tmp_path / "prs.arl"
        write_pressure_column_file(path)
        z = [_SHGT, _SHGT + 0.5 * _PRS_Z1, _PRS_HGTS[4]]
        result = _sample_column(path, ["TEMP"], z, z_kind="msl")
        np.testing.assert_allclose(
            result["TEMP"],
            [_T02M, 0.5 * (_T02M + _PRS_TEMP[3]), _PRS_TEMP[4]],
            atol=1e-3,
        )

    def test_without_surface_fields_the_ground_is_level_one(self, tmp_path):
        # PRFSFCT: T_ground = T(1) * (P(1) / PRSS) ** -0.286, with level 1 the file's
        # first level, here below the ground; U at the ground is U(1).
        path = tmp_path / "prs.arl"
        write_pressure_column_file(path, t02m=False, u10m=False)
        result = _sample_column(path, ["TEMP", "UWND"], [0.0]).iloc[0]
        t_ground = _PRS_TEMP[0] * (_PRS_LEVELS[0] / _PRSS) ** -0.286
        np.testing.assert_allclose(result["TEMP"], t_ground, rtol=1e-6)
        np.testing.assert_allclose(result["UWND"], _PRS_UWND[0], atol=1e-4)

    def test_low_ground_below_every_level_is_not_nan(self, tmp_path):
        # The ground under the first level: a point between them used to be NaN.
        path = tmp_path / "prs.arl"
        write_pressure_column_file(path, shgt=0.0)
        result = _sample_column(path, ["TEMP"], [0.5 * _PRS_HGTS[0]])
        np.testing.assert_allclose(
            result["TEMP"], [0.5 * (_T02M + _PRS_TEMP[0])], atol=1e-3
        )

    def test_below_the_ground_is_nan(self, tmp_path):
        path = tmp_path / "prs.arl"
        write_pressure_column_file(path)
        result = _sample_column(path, ["TEMP", "pressure"], [-1.0]).iloc[0]
        assert np.isnan(result["TEMP"]) and np.isnan(result["pressure"])

    def test_pressure_targets_use_the_stored_levels(self, tmp_path):
        # Sampling at pressures is unchanged: the stored levels, none skipped.
        path = tmp_path / "prs.arl"
        write_pressure_column_file(path)
        result = _sample_column(path, ["TEMP"], [975.0], z_kind="pressure")
        np.testing.assert_allclose(result["TEMP"], [304.0], atol=1e-3)


class TestHorizontalInterpolation:
    """Verify bilinear weights and nearest-neighbor snapping."""

    def _write_gradient_file(self, path, time):
        """Single-level file; TEMP[y, x] = x + 10.0 * y. Single level avoids vertical interp."""
        grid = make_test_grid()
        vaxis = PressureAxis(levels=[1000.0])
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
    """Correct NaN and error behavior for out-of-range targets and missing fields."""

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

    def test_duplicate_time_across_files_raises(self, tmp_path):
        path = tmp_path / "dup.arl"
        time = pd.Timestamp("2024-07-18")
        write_sampling_file(path, time=time)

        points = pd.DataFrame(
            {"lon": [20.0], "lat": [-10.0], "z": [950.0], "time": [time]}
        )
        with File(path) as f, pytest.raises(ValueError, match="Multiple input files"):
            sample_points([f, f], points, ["TEMP"], z_kind="pressure")


class TestMultiTimeFile:
    """Sampling a single file that contains multiple recordsets."""

    def test_dispatches_by_time_column(self, tmp_path):
        path = tmp_path / "multi.arl"
        time0 = pd.Timestamp("2024-07-18 00:00")
        time1 = pd.Timestamp("2024-07-18 06:00")

        grid = make_test_grid()
        vaxis = PressureAxis(levels=[1000.0, 900.0, 800.0])

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


# --- result shape, time resolution, and argument validation ---

TIME0 = pd.Timestamp("2024-07-18 00:00")
TIME1 = pd.Timestamp("2024-07-18 06:00")


def write_two_time_file(path):
    """Write uniform TEMP that samples to 285 K (TIME0) and 305 K (TIME1) at 950 hPa."""
    grid = make_test_grid()
    vaxis = PressureAxis(levels=[1000.0, 900.0, 800.0])
    with File(path, mode="w", source="TEST", grid=grid, vertical_axis=vaxis) as arl:
        for t, base in [(TIME0, 280.0), (TIME1, 300.0)]:
            rs = arl.create_recordset(t)
            for i, val in enumerate([base, base + 10.0, base + 20.0]):
                data = np.full((grid.ny, grid.nx), val, dtype=np.float32)
                rs.create_datarecord("TEMP", level=i, forecast=0, data=data)
    return path


class TestSampleResult:
    def test_preserves_caller_columns_and_index(self, tmp_path):
        path = write_two_time_file(tmp_path / "multi.arl")
        points = pd.DataFrame(
            {
                "site": ["a", "b", "c"],
                "time": [TIME1, TIME0, TIME1],
                "lon": [20, 20, 20],  # ints stay ints: the result is a copy
                "lat": [-10.0, -10.0, -10.0],
                "z": [950.0, 950.0, 950.0],
                "obs": [1.0, 2.0, 3.0],
            },
            index=pd.Index([7, 3, 7], name="id"),  # non-default, duplicated
        )

        result = sample_points(path, points, ["TEMP"])

        assert list(result.columns) == [*points.columns, "TEMP"]
        pd.testing.assert_index_equal(result.index, points.index)
        pd.testing.assert_frame_equal(result[points.columns], points)
        np.testing.assert_allclose(result["TEMP"], [305.0, 285.0, 305.0], atol=1e-3)
        assert "TEMP" not in points.columns  # input is not modified

    def test_multi_file_preserves_columns_and_order(self, tmp_path):
        f0, f1 = tmp_path / "t0.arl", tmp_path / "t1.arl"
        write_sampling_file(f0, time=TIME0, temp_offset=0.0)
        write_sampling_file(f1, time=TIME1, temp_offset=10.0)
        points = pd.DataFrame(
            {
                "time": [TIME1, TIME0, TIME1],
                "lon": [20.0, 20.0, 20.0],
                "lat": [-10.0, -10.0, -10.0],
                "z": [1000.0, 1000.0, 1000.0],
                "tag": ["x", "y", "z"],
            },
            index=["p", "q", "r"],
        )

        result = sample_points([f0, f1], points, ["TEMP"])

        assert list(result.index) == ["p", "q", "r"]
        assert list(result["tag"]) == ["x", "y", "z"]
        np.testing.assert_allclose(result["TEMP"], [290.0, 280.0, 290.0], atol=0.1)

    def test_accepts_mapping_points(self, tmp_path):
        path = tmp_path / "met.arl"
        write_sampling_file(path, time=TIME0)
        points = {"lon": [20.0], "lat": [-10.0], "z": [1000.0], "note": ["n"]}

        result = sample_points(path, points, "TEMP")

        assert list(result.columns) == ["lon", "lat", "z", "note", "TEMP"]

    def test_variable_colliding_with_column_raises(self, tmp_path):
        path = tmp_path / "met.arl"
        write_sampling_file(path, time=TIME0)
        points = pd.DataFrame(
            {"lon": [20.0], "lat": [-10.0], "z": [1000.0], "TEMP": [1.0]}
        )

        with pytest.raises(ValueError, match="collide with existing columns"):
            sample_points(path, points, ["TEMP"])


class TestSampleTimeResolution:
    POINTS = {"lon": [20.0], "lat": [-10.0], "z": [950.0]}

    def test_time_column_and_time_argument_is_ambiguous(self, tmp_path):
        path = write_two_time_file(tmp_path / "multi.arl")
        points = pd.DataFrame({**self.POINTS, "time": [TIME0]})

        with pytest.raises(ValueError, match="pass only one"):
            sample_points(path, points, ["TEMP"], time=TIME0)
        with File(path) as arl, pytest.raises(ValueError, match="pass only one"):
            arl.sample_points(points, ["TEMP"], time=TIME0)

    def test_time_argument_applies_to_every_point(self, tmp_path):
        path = write_two_time_file(tmp_path / "multi.arl")

        result = sample_points(path, self.POINTS, ["TEMP"], time="2024-07-18 06:00")

        np.testing.assert_allclose(result["TEMP"], [305.0], atol=1e-3)
        assert "time" not in result.columns

    def test_time_argument_dispatches_across_files(self, tmp_path):
        f0, f1 = tmp_path / "t0.arl", tmp_path / "t1.arl"
        write_sampling_file(f0, time=TIME0, temp_offset=0.0)
        write_sampling_file(f1, time=TIME1, temp_offset=10.0)
        points = {"lon": [20.0], "lat": [-10.0], "z": [1000.0]}

        result = sample_points([f0, f1], points, ["TEMP"], time=TIME1)

        np.testing.assert_allclose(result["TEMP"], [290.0], atol=0.1)

    def test_single_time_input_is_implied(self, tmp_path):
        path = tmp_path / "met.arl"
        write_sampling_file(path, time=TIME0)
        points = {"lon": [20.0], "lat": [-10.0], "z": [1000.0]}

        result = sample_points(path, points, ["TEMP"])

        np.testing.assert_allclose(result["TEMP"], [280.0], atol=0.1)

    def test_no_time_for_multi_time_input_raises(self, tmp_path):
        path = write_two_time_file(tmp_path / "multi.arl")

        with pytest.raises(ValueError, match="needs a time"):
            sample_points(path, self.POINTS, ["TEMP"])

    def test_missing_time_raises_same_error_for_one_or_many_files(self, tmp_path):
        f0, f1 = tmp_path / "t0.arl", tmp_path / "t1.arl"
        write_sampling_file(f0, time=TIME0)
        write_sampling_file(f1, time=TIME1)
        points = pd.DataFrame({**self.POINTS, "time": [pd.Timestamp("2030-01-01")]})

        match = "No input file contains the requested point times: 2030-01-01"
        with pytest.raises(ValueError, match=match):
            sample_points(f0, points, ["TEMP"])
        with File(f0) as arl, pytest.raises(ValueError, match=match):
            arl.sample_points(points, ["TEMP"])
        with pytest.raises(ValueError, match=match):
            sample_points([f0, f1], points, ["TEMP"])

    def test_nat_time_raises(self, tmp_path):
        path = write_two_time_file(tmp_path / "multi.arl")
        points = pd.DataFrame({**self.POINTS, "time": [pd.NaT]})

        with pytest.raises(ValueError, match="NaT"):
            sample_points(path, points, ["TEMP"])


class TestSampleArguments:
    POINTS = {"lon": [20.0], "lat": [-10.0], "z": [950.0]}

    def test_invalid_z_kind_and_method_raise(self, tmp_path):
        path = tmp_path / "met.arl"
        write_sampling_file(path, time=TIME0)

        with pytest.raises(ValueError, match="z_kind"):
            sample_points(path, self.POINTS, ["TEMP"], z_kind="height")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="method"):
            sample_points(path, self.POINTS, ["TEMP"], method="cubic")  # type: ignore[arg-type]
        with File(path) as arl, pytest.raises(ValueError, match="z_kind"):
            arl.sample_points(self.POINTS, ["TEMP"], z_kind="height")  # type: ignore[arg-type]

    def test_files_keyword_and_keyword_only_options(self, tmp_path):
        path = tmp_path / "met.arl"
        write_sampling_file(path, time=TIME0)

        result = sample_points(files=path, points=self.POINTS, variables=["TEMP"])
        assert "TEMP" in result.columns
        with pytest.raises(TypeError):
            sample_points(path, self.POINTS, ["TEMP"], TIME0)  # type: ignore[misc]

    def test_empty_file_sequence_raises(self):
        with pytest.raises(ValueError, match="at least one input file"):
            sample_points([], self.POINTS, ["TEMP"])


# --- earth-relative winds on projected grids ---


def make_lambert_grid(nx: int = 30, ny: int = 30) -> Grid:
    """HRRR-like Lambert conformal grid over the western US."""
    projection = Projection(
        pole_lat=90.0,
        pole_lon=0.0,
        tangent_lat=38.5,
        tangent_lon=-97.5,
        grid_size=30.0,
        orientation=0.0,
        cone_angle=38.5,
        sync_x=1.0,
        sync_y=1.0,
        sync_lat=36.0,
        sync_lon=-118.0,
    )
    return Grid(projection=projection, nx=nx, ny=ny)


def write_wind_file(path, grid: Grid, *, time: pd.Timestamp, u: float, v: float):
    """Single-level file with uniform grid-relative winds on *grid*."""
    vertical_axis = PressureAxis(levels=[1000.0, 850.0])  # level 0 holds surface fields
    shape = (grid.ny, grid.nx)
    with File(
        path, mode="w", source="TEST", grid=grid, vertical_axis=vertical_axis
    ) as arl:
        rs = arl.create_recordset(time)
        for name, value in (("UWND", u), ("VWND", v), ("U10M", u), ("V10M", v)):
            rs.create_datarecord(
                name,
                level=1 if name in {"UWND", "VWND"} else 0,
                forecast=0,
                data=np.full(shape, value, dtype=np.float32),
            )


def _lambert_points(grid: Grid) -> pd.DataFrame:
    coords = grid.calculate_coords()
    lons = coords["lon"][1]  # coords are (dims, array) pairs
    lats = coords["lat"][1]
    rows = [(5, 5), (10, 20), (25, 12)]
    return pd.DataFrame(
        {
            "lon": [float(lons[j, i]) for j, i in rows],
            "lat": [float(lats[j, i]) for j, i in rows],
            "z": [850.0] * len(rows),
        }
    )


def test_sample_points_earth_relative_rotates_lambert_winds(tmp_path):
    grid = make_lambert_grid()
    time = pd.Timestamp("2024-07-01T00:00")
    path = tmp_path / "lcc.arl"
    write_wind_file(path, grid, time=time, u=3.0, v=4.0)
    points = _lambert_points(grid)

    stored = sample_points(path, points, ["UWND", "VWND"])
    rotated = sample_points(path, points, ["UWND", "VWND"], earth_relative=True)

    np.testing.assert_allclose(stored["UWND"], 3.0, rtol=1e-5)
    np.testing.assert_allclose(stored["VWND"], 4.0, rtol=1e-5)

    # HRRR's documented rotation: angle = sin(lat_1) * (lon - lon_0)
    angle = np.radians(np.sin(np.radians(38.5)) * (points["lon"].to_numpy() + 97.5))
    expected_u = np.cos(angle) * 3.0 + np.sin(angle) * 4.0
    expected_v = -np.sin(angle) * 3.0 + np.cos(angle) * 4.0
    np.testing.assert_allclose(rotated["UWND"], expected_u, rtol=1e-5)
    np.testing.assert_allclose(rotated["VWND"], expected_v, rtol=1e-5)
    # Rotation preserves speed.
    np.testing.assert_allclose(
        np.hypot(rotated["UWND"], rotated["VWND"]), 5.0, rtol=1e-5
    )
    # The rotation is not a no-op away from the central meridian.
    assert np.all(np.abs(rotated["UWND"] - 3.0) > 0.1)


def test_sample_points_earth_relative_rotates_10m_winds(tmp_path):
    grid = make_lambert_grid()
    time = pd.Timestamp("2024-07-01T00:00")
    path = tmp_path / "lcc.arl"
    write_wind_file(path, grid, time=time, u=3.0, v=4.0)
    points = _lambert_points(grid)

    upper = sample_points(path, points, ["UWND", "VWND"], earth_relative=True)
    surface = sample_points(path, points, ["U10M", "V10M"], earth_relative=True)
    np.testing.assert_allclose(surface["U10M"], upper["UWND"], rtol=1e-6)
    np.testing.assert_allclose(surface["V10M"], upper["VWND"], rtol=1e-6)


def test_sample_points_earth_relative_requires_both_components(tmp_path):
    grid = make_lambert_grid()
    time = pd.Timestamp("2024-07-01T00:00")
    path = tmp_path / "lcc.arl"
    write_wind_file(path, grid, time=time, u=3.0, v=4.0)
    points = _lambert_points(grid)

    with pytest.raises(ValueError, match="'UWND' was requested without 'VWND'"):
        sample_points(path, points, ["UWND"], earth_relative=True)
    # Without the flag a lone component is fine.
    assert "UWND" in sample_points(path, points, ["UWND"]).columns


def test_sample_points_earth_relative_no_op_on_latlon(tmp_path):
    grid = make_test_grid()
    time = pd.Timestamp("2024-07-01T00:00")
    path = tmp_path / "latlon.arl"
    write_wind_file(path, grid, time=time, u=3.0, v=4.0)
    coords = grid.calculate_coords()
    points = pd.DataFrame(
        {
            "lon": coords["lon"][1][[2, 7]],
            "lat": coords["lat"][1][[3, 9]],
            "z": [850.0, 850.0],
        }
    )

    rotated = sample_points(path, points, ["UWND", "VWND"], earth_relative=True)
    np.testing.assert_allclose(rotated["UWND"], 3.0, rtol=1e-6)
    np.testing.assert_allclose(rotated["VWND"], 4.0, rtol=1e-6)


def test_file_sample_points_passes_earth_relative(tmp_path):
    grid = make_lambert_grid()
    time = pd.Timestamp("2024-07-01T00:00")
    path = tmp_path / "lcc.arl"
    write_wind_file(path, grid, time=time, u=3.0, v=4.0)
    points = _lambert_points(grid)

    with File(path) as met:
        via_method = met.sample_points(points, ["UWND", "VWND"], earth_relative=True)
    via_function = sample_points(path, points, ["UWND", "VWND"], earth_relative=True)
    pd.testing.assert_frame_equal(via_method, via_function)


def write_global_file(path, time):
    """Write a 10-degree global lat/lon file whose TEMP equals the column longitude."""
    projection = Projection(
        pole_lat=90.0,
        pole_lon=0.0,
        tangent_lat=10.0,
        tangent_lon=10.0,
        grid_size=0.0,
        orientation=0.0,
        cone_angle=0.0,
        sync_x=1.0,
        sync_y=1.0,
        sync_lat=-90.0,
        sync_lon=0.0,
    )
    grid = Grid(projection=projection, nx=36, ny=19)
    lons = np.arange(36, dtype=np.float32) * 10.0  # 0, 10, ..., 350
    field = np.broadcast_to(lons, (19, 36)).copy()
    with File(
        path,
        mode="w",
        source="TEST",
        grid=grid,
        vertical_axis=PressureAxis(levels=[0.0, 1000.0]),
    ) as arl:
        rs = arl.create_recordset(time, forecast=0)
        rs.create_datarecord("TEMP", level=1, forecast=0, data=field)
    return path


def test_sample_points_global_grid_western_hemisphere(tmp_path):
    time = pd.Timestamp("2024-01-01 00:00")
    path = write_global_file(tmp_path / "global.arl", time)
    # -90 and 270 are the same meridian; both used to fall off a 0-360 grid.
    points = pd.DataFrame(
        {
            "lon": [-90.0, 270.0, 45.0, -175.0],
            "lat": [30.0, 30.0, 30.0, 30.0],
            "z": [1.0, 1.0, 1.0, 1.0],
            "time": [time] * 4,
        }
    )

    result = sample_points(path, points, ["TEMP"], z_kind="native")

    np.testing.assert_allclose(result["TEMP"], [270.0, 270.0, 45.0, 185.0])


def test_sample_points_global_grid_interpolates_across_seam(tmp_path):
    time = pd.Timestamp("2024-01-01 00:00")
    path = write_global_file(tmp_path / "global.arl", time)
    # 355 E lies between the last column (350) and the first (0 == 360).
    points = pd.DataFrame(
        {"lon": [355.0, -5.0], "lat": [0.0, 0.0], "z": [1.0, 1.0], "time": [time] * 2}
    )

    linear = sample_points(path, points, ["TEMP"], z_kind="native")
    nearest = sample_points(path, points, ["TEMP"], z_kind="native", method="nearest")

    # Halfway between TEMP=350 (x=35) and TEMP=0 (x=0, wrapped).
    np.testing.assert_allclose(linear["TEMP"], [175.0, 175.0])
    assert set(nearest["TEMP"]) <= {350.0, 0.0}


def test_sample_points_regional_latlon_grid_still_masks_outside(tmp_path):
    time = pd.Timestamp("2024-01-01 00:00")
    path = tmp_path / "regional.arl"
    write_sampling_file(path, time=time)
    # make_test_grid spans lon 20-39: 10 E (west of it) and 200 E are outside.
    points = pd.DataFrame(
        {"lon": [10.0, 200.0], "lat": [0.0, 0.0], "z": [1.0, 1.0], "time": [time] * 2}
    )

    result = sample_points(path, points, ["TEMP"], z_kind="native")

    assert result["TEMP"].isna().all()
