"""Tests for point-sampling helpers."""

import numpy as np
import pandas as pd

from arlmet import File, sample_points, terrain
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

    with File(path, mode="w", source="TEST", grid=grid, vertical_axis=vertical_axis) as arl:
        rs = arl.create_recordset(time)
        rs.create_datarecord("SHGT", level=0, forecast=0, data=terrain.astype(np.float32))
        rs.create_datarecord(
            "PRSS",
            level=0,
            forecast=0,
            data=surface_pressure.astype(np.float32),
        )
        rs.create_datarecord("TEMP", level=0, forecast=0, data=temp0.astype(np.float32))
        rs.create_datarecord("TEMP", level=1, forecast=0, data=temp1.astype(np.float32))
        rs.create_datarecord("TEMP", level=2, forecast=0, data=temp2.astype(np.float32))

    return {
        "grid": grid,
        "vertical_axis": vertical_axis,
        "terrain": terrain.astype(np.float32),
        "surface_pressure": surface_pressure.astype(np.float32),
        "temp0": temp0.astype(np.float32),
        "temp1": temp1.astype(np.float32),
        "temp2": temp2.astype(np.float32),
    }


def write_deep_sampling_file(path, *, time: pd.Timestamp, temp_offset: float = 0.0):
    grid = make_test_grid(nx=30, ny=30)
    levels = [1000.0, 900.0, 800.0, 700.0, 600.0, 500.0, 400.0, 300.0, 200.0, 100.0, 50.0, 10.0, 1.0]
    vertical_axis = VerticalAxis(flag=2, levels=levels)
    yy, xx = np.meshgrid(np.arange(grid.ny), np.arange(grid.nx), indexing="ij")
    base = 0.1 * yy + 0.1 * xx

    terrain = 100.0 + base
    surface_pressure = 1000.0 + yy + xx

    with File(path, mode="w", source="TEST", grid=grid, vertical_axis=vertical_axis) as arl:
        rs = arl.create_recordset(time)
        rs.create_datarecord("SHGT", level=0, forecast=0, data=terrain.astype(np.float32))
        rs.create_datarecord(
            "PRSS",
            level=0,
            forecast=0,
            data=surface_pressure.astype(np.float32),
        )
        for level_index, _pressure_level in enumerate(levels):
            temp = 280.0 + temp_offset + 5.0 * level_index + base
            rs.create_datarecord(
                "TEMP",
                level=level_index,
                forecast=0,
                data=temp.astype(np.float32),
            )

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
        np.testing.assert_allclose(terrain(path, time=time), sample["terrain"])

    def test_file_sample_points_interpolates_native_and_pressure_queries(self, tmp_path):
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
            native = arl.sample_points(points.iloc[[0]], ["TEMP", "PRSS"], time=time, z_kind="native")
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
        delta_z = VerticalAxis.Z_PHI1[-1]
        z0 = (surface_pressure - 1000.0) * delta_z
        z1 = (surface_pressure - 900.0) * delta_z
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

        np.testing.assert_allclose(agl["TEMP"].iloc[0], 290.5, atol=1e-5)
        np.testing.assert_allclose(agl["pressure"].iloc[0], 950.0, atol=1e-5)
        np.testing.assert_allclose(msl["TEMP"].iloc[0], 290.5, atol=1e-5)
        np.testing.assert_allclose(msl["pressure"].iloc[0], 950.0, atol=1e-5)

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

        result = sample_points([path0, path1], points, ["TEMP"], z_kind="pressure")

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
