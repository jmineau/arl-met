"""Tests for ARL write-path plumbing."""

import numpy as np
import pandas as pd
import pytest

from arlmet import File, open_dataset, write_dataset
from arlmet.grid import Grid, Projection
from arlmet.metadata import IndexRecord
from arlmet.packing import pack, unpack
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


class TestWriter:
    def write_sample_file(self, path):
        grid = make_test_grid()
        vertical_axis = VerticalAxis(flag=2, levels=[0.0, 1000.0])

        prss0 = 1000.0 + np.arange(grid.nx * grid.ny, dtype=np.float32).reshape(
            grid.ny, grid.nx
        ) * 0.01
        temp0 = 280.0 + np.arange(grid.nx * grid.ny, dtype=np.float32).reshape(
            grid.ny, grid.nx
        ) * 0.05
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
        assert index0.forecast_hour == 0
        assert index0.vertical_flag == 2
        assert [lvl.height for lvl in index0.levels] == [0.0, 1000.0]
        assert list(index0.levels[0].variables) == ["PRSS"]
        assert list(index0.levels[1].variables) == ["TEMP"]

        reopened = File(path)
        assert reopened.source == "TEST"
        assert reopened.grid == grid
        assert reopened.vertical_axis.flag == 2
        assert reopened.vertical_axis.heights.tolist() == [0.0, 1000.0]
        assert reopened.times == [time0, time1]

        rs0 = reopened[time0]
        rs1 = reopened[time1]
        assert rs0.position == 0
        assert rs1.position == reopened.record_size * 3

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

    def test_write_dataset_roundtrip_from_open_dataset(self, tmp_path):
        source_path = tmp_path / "source.arl"
        written_path = tmp_path / "written.arl"

        sample = self.write_sample_file(source_path)
        ds = open_dataset(source_path, squeeze=False)

        assert ds.attrs["source"] == "TEST"
        assert ds.attrs["arl_vertical_flag"] == 2
        assert ds.attrs["arl_vertical_levels"] == [0.0, 1000.0]
        np.testing.assert_array_equal(ds.coords["forecast"].values, [0, 3])

        write_dataset(ds, written_path)

        original = open_dataset(source_path, squeeze=False)
        roundtripped = open_dataset(written_path, squeeze=False)

        assert roundtripped.attrs["source"] == "TEST"
        assert roundtripped.attrs["arl_nx"] == sample["grid"].nx
        assert roundtripped.attrs["arl_ny"] == sample["grid"].ny
        assert roundtripped.attrs["arl_vertical_levels"] == [0.0, 1000.0]
        np.testing.assert_array_equal(
            roundtripped.coords["forecast"].values, original.coords["forecast"].values
        )
        np.testing.assert_array_equal(
            roundtripped.coords["level"].values, original.coords["level"].values
        )

        for name in ("PRSS", "TEMP"):
            np.testing.assert_allclose(
                np.asarray(roundtripped[name]),
                np.asarray(original[name]),
                atol=1e-6,
            )

    def test_write_dataset_uses_serialized_metadata_attrs(self, tmp_path):
        source_path = tmp_path / "source.arl"
        written_path = tmp_path / "written.arl"

        self.write_sample_file(source_path)
        ds = open_dataset(source_path, squeeze=False)
        del ds.attrs["grid"]
        del ds.attrs["vertical_axis"]

        write_dataset(ds, written_path)

        reopened = open_dataset(written_path, squeeze=False)
        assert reopened.attrs["source"] == "TEST"
        np.testing.assert_array_equal(reopened.coords["forecast"].values, [0, 3])

    def test_write_dataset_requires_forecast_coordinate(self, tmp_path):
        source_path = tmp_path / "source.arl"
        written_path = tmp_path / "written.arl"

        self.write_sample_file(source_path)
        ds = open_dataset(source_path, squeeze=False).drop_vars("forecast")

        with pytest.raises(ValueError, match="forecast"):
            write_dataset(ds, written_path)

    def test_write_dataset_accepts_scalar_time_level_and_omits_all_nan_slices(
        self, tmp_path
    ):
        source_path = tmp_path / "source.arl"
        written_path = tmp_path / "written.arl"

        self.write_sample_file(source_path)
        ds = open_dataset(source_path, squeeze=False).load().isel(time=0, level=0)

        write_dataset(ds, written_path)

        reopened = open_dataset(written_path, squeeze=False)
        assert list(reopened.data_vars) == ["PRSS"]
        assert reopened.sizes["time"] == 1
        assert reopened.sizes["level"] == 1
        np.testing.assert_array_equal(reopened.coords["forecast"].values, [0])

    def test_write_dataset_rejects_partial_nan_slice(self, tmp_path):
        source_path = tmp_path / "source.arl"
        written_path = tmp_path / "written.arl"

        self.write_sample_file(source_path)
        ds = open_dataset(source_path, squeeze=False).load()
        ds["PRSS"].data[0, 0, 0, 0] = np.nan

        with pytest.raises(ValueError, match="partially-missing slice"):
            write_dataset(ds, written_path)

    def test_write_dataset_rejects_malformed_serialized_grid_metadata(self, tmp_path):
        source_path = tmp_path / "source.arl"
        written_path = tmp_path / "written.arl"

        self.write_sample_file(source_path)
        ds = open_dataset(source_path, squeeze=False)
        del ds.attrs["grid"]
        del ds.attrs["vertical_axis"]
        ds.attrs["arl_nx"] = "not-an-int"

        with pytest.raises(ValueError, match="ARL grid metadata"):
            write_dataset(ds, written_path)

    def test_write_dataset_rejects_malformed_serialized_vertical_metadata(
        self, tmp_path
    ):
        source_path = tmp_path / "source.arl"
        written_path = tmp_path / "written.arl"

        self.write_sample_file(source_path)
        ds = open_dataset(source_path, squeeze=False)
        del ds.attrs["grid"]
        del ds.attrs["vertical_axis"]
        ds.attrs["arl_vertical_flag"] = "bad-flag"

        with pytest.raises(ValueError, match="ARL vertical metadata"):
            write_dataset(ds, written_path)

    def test_write_dataset_rejects_dif_variables(self, tmp_path):
        source_path = tmp_path / "source.arl"
        written_path = tmp_path / "written.arl"

        self.write_sample_file(source_path)
        ds = open_dataset(source_path, squeeze=False).load().rename({"PRSS": "DIFW"})

        with pytest.raises(NotImplementedError, match="DIF"):
            write_dataset(ds, written_path)

    @staticmethod
    def _pack_args(data: np.ndarray) -> tuple[bytes, int, int, float, int, float, type[np]]:
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
