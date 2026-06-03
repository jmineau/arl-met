"""Tests for dataset reading and low-level ARL plumbing."""

import numpy as np
import pandas as pd
import pytest
from xarray.core import indexing

from arlmet import File, open_dataset, write_dataset
from arlmet.grid import Grid, Projection
from arlmet.index import IndexRecord
from arlmet.packing import pack, unpack
from arlmet.vertical import PressureAxis


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


class TestWriter:
    def write_sample_file(self, path):
        grid = make_test_grid()
        vertical_axis = PressureAxis(levels=[0.0, 1000.0])

        prss0 = (
            1000.0
            + np.arange(grid.nx * grid.ny, dtype=np.float32).reshape(grid.ny, grid.nx)
            * 0.01
        )
        temp0 = (
            280.0
            + np.arange(grid.nx * grid.ny, dtype=np.float32).reshape(grid.ny, grid.nx)
            * 0.05
        )
        prss1 = prss0 + 1.0
        temp1 = temp0 + 2.0

        time0 = pd.Timestamp("2024-07-18 00:00")
        time1 = pd.Timestamp("2024-07-18 03:00")

        with File(
            path,
            mode="w",
            source="TEST",
            grid=grid,
            vertical_axis=vertical_axis,
        ) as arl:
            rs0 = arl.create_recordset(time0)
            rs0.create_datarecord("PRSS", level=0, forecast=0, data=prss0)
            rs0.create_datarecord("TEMP", level=1, forecast=0, data=temp0)

            rs1 = arl.create_recordset(time1)
            rs1.create_datarecord("PRSS", level=0, forecast=3, data=prss1)
            rs1.create_datarecord("TEMP", level=1, forecast=3, data=temp1)

        return {
            "grid": grid,
            "vertical_axis": vertical_axis,
            "prss0": prss0,
            "temp0": temp0,
            "prss1": prss1,
            "temp1": temp1,
            "time0": time0,
            "time1": time1,
        }

    def write_surface_only_file(self, path):
        grid = make_test_grid()
        vertical_axis = PressureAxis(levels=[0.0])
        time0 = pd.Timestamp("2024-07-18 00:00")
        time1 = pd.Timestamp("2024-07-18 03:00")
        data0 = np.ones((grid.ny, grid.nx), dtype=np.float32)
        data1 = data0 + 1.0

        with File(
            path,
            mode="w",
            source="TEST",
            grid=grid,
            vertical_axis=vertical_axis,
        ) as arl:
            rs0 = arl.create_recordset(time0, forecast=6)
            rs0.create_datarecord("PRSS", level=0, forecast=6, data=data0)

            rs1 = arl.create_recordset(time1, forecast=6)
            rs1.create_datarecord("PRSS", level=0, forecast=6, data=data1)

    def test_close_writes_index_and_data_records_roundtrip(self, tmp_path):
        path = tmp_path / "roundtrip.arl"
        sample = self.write_sample_file(path)
        grid = sample["grid"]
        prss0 = sample["prss0"]
        temp0 = sample["temp0"]
        prss1 = sample["prss1"]
        temp1 = sample["temp1"]
        time0 = sample["time0"]
        time1 = sample["time1"]

        with open(path, "rb") as handle:
            index0 = IndexRecord.from_position(handle, 0)

        assert index0.source == "TEST"
        assert index0.forecast == 0
        assert index0.vertical_flag == 2
        assert [lvl.height for lvl in index0.levels] == [0.0, 1000.0]
        assert list(index0.levels[0].variables) == ["PRSS"]
        assert list(index0.levels[1].variables) == ["TEMP"]

        reopened = File(path)
        assert reopened.source == "TEST"
        assert reopened.grid == grid
        assert reopened.vertical_axis.flag == 2
        assert reopened.vertical_axis.levels.tolist() == [0.0, 1000.0]
        assert reopened.times == [time0, time1]

        rs0 = reopened[time0]
        rs1 = reopened[time1]
        assert rs0.position == 0
        assert rs1.position == reopened.record_length * 3

        prss0_rt = rs0[(0, "PRSS")]
        temp0_rt = rs0[(1, "TEMP")]
        prss1_rt = rs1[(0, "PRSS")]
        temp1_rt = rs1[(1, "TEMP")]

        prss0_expected = unpack(*self._pack_args(prss0))
        temp0_expected = unpack(*self._pack_args(temp0))
        prss1_expected = unpack(*self._pack_args(prss1))
        temp1_expected = unpack(*self._pack_args(temp1))

        np.testing.assert_allclose(
            np.asarray(prss0_rt), prss0_expected, atol=prss0_rt.header.precision
        )
        np.testing.assert_allclose(
            np.asarray(temp0_rt), temp0_expected, atol=temp0_rt.header.precision
        )
        np.testing.assert_allclose(
            np.asarray(prss1_rt), prss1_expected, atol=prss1_rt.header.precision
        )
        np.testing.assert_allclose(
            np.asarray(temp1_rt), temp1_expected, atol=temp1_rt.header.precision
        )

        assert prss0_rt.verify_checksum()
        assert temp0_rt.verify_checksum()
        assert prss1_rt.verify_checksum()
        assert temp1_rt.verify_checksum()

    def test_open_dataset_preserves_forecast_metadata(self, tmp_path):
        source_path = tmp_path / "source.arl"

        self.write_sample_file(source_path)
        ds = open_dataset(source_path)

        assert ds.attrs["source"] == "TEST"
        assert ds.arl.vertical_axis.flag == 2
        assert ds.arl.vertical_axis.levels.tolist() == [0.0, 1000.0]
        np.testing.assert_array_equal(ds["forecast_hour"].values, [0, 3])
        assert (
            ds["forecast_hour"].attrs["long_name"] == "ARL index record forecast hour"
        )
        assert (
            "Individual variable record forecasts may differ"
            in ds["forecast_hour"].attrs["description"]
        )

    def test_open_dataset_handles_mixed_forecast_hours_within_recordset(self, tmp_path):
        """
        open_dataset must not raise when data records within one time step
        carry different per-record forecast hours (e.g. GDAS weekly files).
        """
        path = tmp_path / "mixed_forecast.arl"
        grid = make_test_grid()
        vertical_axis = PressureAxis(levels=[0.0, 1000.0])
        time0 = pd.Timestamp("2025-09-01 00:00")
        data = np.ones((grid.ny, grid.nx), dtype=np.float32)

        with File(
            path, mode="w", source="TEST", grid=grid, vertical_axis=vertical_axis
        ) as arl:
            rs = arl.create_recordset(time0, forecast=0)
            rs.create_datarecord("PRSS", level=0, forecast=0, data=data)
            rs.create_datarecord("TEMP", level=1, forecast=3, data=data)

        ds = open_dataset(path)

        assert "PRSS" in ds
        assert "TEMP" in ds
        np.testing.assert_array_equal(ds["forecast_hour"].values, [0])
        assert ds.sizes["time"] == 1
        assert ds.sizes["level"] == 1

    def test_write_dataset_roundtrips_simple_dataset(self, tmp_path):
        source_path = tmp_path / "source.arl"
        written_path = tmp_path / "written.arl"

        self.write_sample_file(source_path)
        ds = open_dataset(source_path)

        write_dataset(ds, written_path)
        reopened = open_dataset(written_path)

        np.testing.assert_array_equal(
            reopened["forecast_hour"].values, ds["forecast_hour"].values
        )
        np.testing.assert_allclose(reopened["PRSS"].values, ds["PRSS"].values)
        np.testing.assert_allclose(reopened["TEMP"].values, ds["TEMP"].values)

    def test_write_dataset_defaults_missing_forecast_hours_to_zero(self, tmp_path):
        source_path = tmp_path / "source.arl"
        written_path = tmp_path / "written.arl"

        self.write_sample_file(source_path)
        ds = open_dataset(source_path).drop_vars("forecast_hour")

        write_dataset(ds, written_path)

        with File(written_path) as reopened:
            assert [reopened[time].forecast for time in reopened.times] == [0, 0]

    def test_write_dataset_surface_only_requires_vertical_axis(self, tmp_path):
        source_path = tmp_path / "surface.arl"
        written_path = tmp_path / "written.arl"

        self.write_surface_only_file(source_path)
        ds = open_dataset(source_path)

        with pytest.raises(
            ValueError, match="Surface-only datasets require an explicit vertical_axis"
        ):
            write_dataset(ds, written_path)

        write_dataset(ds, written_path, vertical_axis=PressureAxis(levels=[0.0]))
        reopened = open_dataset(written_path)
        np.testing.assert_allclose(reopened["PRSS"].values, ds["PRSS"].values)

    def test_write_dataset_rejects_missing_upper_level_slices(self, tmp_path):
        source_path = tmp_path / "source.arl"
        written_path = tmp_path / "written.arl"

        self.write_sample_file(source_path)
        ds = open_dataset(source_path)
        ds["TEMP"] = ds["TEMP"].where(ds["level"] != 1)

        with pytest.raises(ValueError, match="contains missing values"):
            write_dataset(ds, written_path)

    def test_write_dataset_generates_diff_from_parent_attrs(self, tmp_path):
        source_path = tmp_path / "source.arl"
        written_path = tmp_path / "written.arl"

        self.write_sample_file(source_path)
        ds = open_dataset(source_path)
        ds["TEMP"].attrs["diff"] = "DIFT"

        write_dataset(ds, written_path)

        with File(written_path) as reopened:
            record = reopened[
                self.write_sample_file.__self__
                if False
                else pd.Timestamp("2024-07-18 00:00")
            ]

        with File(written_path) as reopened:
            time0 = pd.Timestamp("2024-07-18 00:00")
            record = reopened[time0][(1, "TEMP")]
            assert record.diff is not None
            assert record.diff.variable == "DIFT"

        reopened_ds = open_dataset(written_path)
        np.testing.assert_allclose(reopened_ds["TEMP"].values, ds["TEMP"].values)

    def test_write_dataset_rejects_invalid_diff_attr_name(self, tmp_path):
        source_path = tmp_path / "source.arl"
        written_path = tmp_path / "written.arl"

        self.write_sample_file(source_path)
        ds = open_dataset(source_path)
        ds["TEMP"].attrs["diff"] = "WDIFF"

        with pytest.raises(ValueError, match="must start with 'DIF'"):
            write_dataset(ds, written_path)

    def test_file_copy_is_byte_identical(self, tmp_path):
        """Read every record from a written file and rewrite it; the bytes must match."""
        source = tmp_path / "source.arl"
        copy = tmp_path / "copy.arl"

        self.write_sample_file(source)

        with (
            File(source) as src,
            File(
                copy,
                mode="w",
                source=src.source,
                grid=src.grid,
                vertical_axis=src.vertical_axis,
            ) as dst,
        ):
            for time in src.times:
                src_rs = src[time]
                dst_rs = dst.create_recordset(time, forecast=src_rs.forecast)
                for record in src_rs:
                    dst_rs.create_datarecord(
                        variable=record.variable,
                        level=record.level,
                        forecast=record.header.forecast,
                        data=np.asarray(record),
                    )

        assert source.read_bytes() == copy.read_bytes()

    def test_write_dataset_rejects_conflicting_diff_bindings(self, tmp_path):
        source_path = tmp_path / "source.arl"
        written_path = tmp_path / "written.arl"

        self.write_sample_file(source_path)
        ds = open_dataset(source_path)
        ds["PRSS"].attrs["diff"] = "DIFX"
        ds["TEMP"].attrs["diff"] = "DIFX"

        with pytest.raises(ValueError, match="already bound to parent"):
            write_dataset(ds, written_path)

    def test_open_dataset_uses_lazy_variable_arrays(self, tmp_path):
        source_path = tmp_path / "source.arl"

        self.write_sample_file(source_path)
        ds = open_dataset(source_path)

        # TEMP is upper-air — LazilyIndexedArray directly wraps ArlVariableArray
        assert isinstance(ds["TEMP"].variable._data, indexing.LazilyIndexedArray)
        assert ds["TEMP"].variable._data.array.__class__.__name__ == "ArlVariableArray"

        # PRSS is surface — stripped via isel, so navigate through lazy wrappers
        backend = ds["PRSS"].variable._data
        while hasattr(backend, "array"):
            backend = backend.array
        assert backend.__class__.__name__ == "ArlVariableArray"

    def test_open_dataset_sfc_vars_have_no_level_dim(self, tmp_path):
        path = tmp_path / "sample.arl"
        self.write_sample_file(path)
        ds = open_dataset(path)
        assert "level" not in ds["PRSS"].dims  # sfc var
        assert "level" in ds["TEMP"].dims  # upper var
        assert "PRSS" in ds.data_vars
        assert "TEMP" in ds.data_vars

    @staticmethod
    def _pack_args(
        data: np.ndarray,
    ) -> tuple[bytes, int, int, float, int, float, type[np]]:
        packed, precision, exponent, initial_value = pack(data)
        return (
            packed.tobytes(),
            data.shape[1],
            data.shape[0],
            precision,
            exponent,
            initial_value,
            np,
        )
