"""Tests for ARL write-path plumbing."""

import numpy as np
import pandas as pd

from arlmet.core import File
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
    def test_close_writes_index_and_data_records_roundtrip(self, tmp_path):
        path = tmp_path / "roundtrip.arl"
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
