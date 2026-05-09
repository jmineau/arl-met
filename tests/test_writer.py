"""Tests for ARL write-path plumbing."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core import indexing

from arlmet import File, open_dataset, write_dataset
from arlmet.grid import Grid, Projection
from arlmet.metadata import IndexRecord
from arlmet.packing import pack, unpack
from arlmet.vertical import VerticalAxis
from arlmet.xarray import (
    _assign_dataset_metadata,
    _expand_scalar_dim,
    attach_record_collection_metadata,
    extract_dataset_forecasts,
    extract_dataset_source,
    extract_dataset_vertical_axis,
    normalize_dataset_for_write,
    record_collection_to_xarray,
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


class TestWriter:
    def write_sample_file(self, path):
        grid = make_test_grid()
        vertical_axis = VerticalAxis(flag=2, levels=[0.0, 1000.0])

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

    def test_open_dataset_handles_mixed_forecast_hours_within_recordset(self, tmp_path):
        """open_dataset must not raise when data records within one time step
        carry different per-record forecast hours (e.g. GDAS weekly files)."""
        path = tmp_path / "mixed_forecast.arl"
        grid = make_test_grid()
        vertical_axis = VerticalAxis(flag=2, levels=[0.0, 1000.0])
        time0 = pd.Timestamp("2025-09-01 00:00")
        data = np.ones((grid.ny, grid.nx), dtype=np.float32)

        with File(path, mode="w", source="TEST", grid=grid, vertical_axis=vertical_axis) as arl:
            rs = arl.create_recordset(time0, forecast=0)
            rs.create_datarecord("PRSS", level=0, forecast=0, data=data)
            rs.create_datarecord("TEMP", level=1, forecast=3, data=data)

        ds = open_dataset(path, squeeze=False)

        assert "PRSS" in ds
        assert "TEMP" in ds
        assert ds.coords["forecast"].item() == 0

    def test_open_dataset_uses_lazy_variable_arrays_without_dask(self, tmp_path):
        source_path = tmp_path / "source.arl"

        self.write_sample_file(source_path)
        ds = open_dataset(source_path, squeeze=False)

        assert isinstance(ds["PRSS"].variable._data, indexing.LazilyIndexedArray)
        assert ds["PRSS"].variable._data.array.__class__.__name__ == "ArlVariableArray"

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

    def test_record_collection_to_xarray_can_drop_variables(self, tmp_path):
        source_path = tmp_path / "source.arl"

        self.write_sample_file(source_path)
        with File(source_path) as arl:
            ds = record_collection_to_xarray(arl, drop_variables=["TEMP"], squeeze=False)

        assert list(ds.data_vars) == ["PRSS"]
        np.testing.assert_array_equal(ds.coords["forecast"].values, [0, 3])

    def test_attach_record_collection_metadata_assigns_scalar_forecast_for_recordset(
        self, tmp_path
    ):
        source_path = tmp_path / "source.arl"

        sample = self.write_sample_file(source_path)
        with File(source_path) as arl:
            ds = xr.Dataset(coords=arl.grid.calculate_coords())
            enriched = attach_record_collection_metadata(arl[sample["time0"]], ds)

        assert enriched.coords["forecast"].item() == 0
        assert enriched.attrs["source"] == "TEST"

    def test_assign_dataset_metadata_handles_scalar_forecast(self):
        grid = make_test_grid()
        vertical_axis = VerticalAxis(flag=2, levels=[0.0, 1000.0])
        ds = xr.Dataset(coords=grid.calculate_coords())

        enriched = _assign_dataset_metadata(
            ds,
            source="TEST",
            grid=grid,
            vertical_axis=vertical_axis,
            forecast_by_time={pd.Timestamp("2024-07-18 00:00"): 6},
        )

        assert enriched.coords["forecast"].item() == 6
        assert enriched.attrs["arl_vertical_levels"] == [0.0, 1000.0]

    def test_expand_scalar_dim_rejects_missing_and_non_scalar_coords(self):
        ds = xr.Dataset(coords={"time": pd.Timestamp("2024-07-18 00:00")})
        expanded = _expand_scalar_dim(ds, "time")
        assert expanded.sizes["time"] == 1

        with pytest.raises(ValueError, match="include a 'level' coordinate"):
            _expand_scalar_dim(ds, "level")

        with pytest.raises(ValueError, match="dimension or scalar coordinate"):
            _expand_scalar_dim(xr.Dataset(coords={"time2": ("z", [1, 2])}), "time2")

    def test_normalize_dataset_for_write_requires_dataset(self):
        with pytest.raises(TypeError, match="xarray.Dataset"):
            normalize_dataset_for_write([1, 2, 3])

    def test_extract_dataset_source_supports_fallback_and_validation(self):
        assert extract_dataset_source({"arl_source": "TEST"}) == "TEST"

        with pytest.raises(ValueError, match="source identifier"):
            extract_dataset_source({})

    def test_extract_dataset_vertical_axis_returns_copy(self):
        axis = VerticalAxis(flag=2, levels=[1000.0, 900.0], offset=5.0)

        extracted = extract_dataset_vertical_axis({"vertical_axis": axis})

        assert extracted is not axis
        assert extracted.flag == axis.flag
        np.testing.assert_allclose(extracted.levels, axis.levels)
        assert extracted.offset == axis.offset

    def test_extract_dataset_forecasts_rejects_wrong_dimensions(self):
        ds = xr.Dataset(coords={"time": [0, 1], "level": [0, 1]})
        ds = ds.assign_coords(forecast=(("time", "level"), np.zeros((2, 2), dtype=int)))

        with pytest.raises(ValueError, match="must use only the 'time' dimension"):
            extract_dataset_forecasts(ds)

    def test_write_dataset_rejects_missing_horizontal_dimension(self, tmp_path):
        source_path = tmp_path / "source.arl"
        written_path = tmp_path / "written.arl"

        self.write_sample_file(source_path)
        ds = open_dataset(source_path, squeeze=False).isel(lon=0)

        with pytest.raises(ValueError, match="required horizontal dimension 'lon'"):
            write_dataset(ds, written_path)

    def test_write_dataset_rejects_wrong_horizontal_dimension_size(self, tmp_path):
        source_path = tmp_path / "source.arl"
        written_path = tmp_path / "written.arl"

        self.write_sample_file(source_path)
        ds = open_dataset(source_path, squeeze=False).isel(lon=slice(0, -1))

        with pytest.raises(ValueError, match="dimension 'lon' has size"):
            write_dataset(ds, written_path)

    def test_write_dataset_rejects_duplicate_level_values(self, tmp_path):
        source_path = tmp_path / "source.arl"
        written_path = tmp_path / "written.arl"

        self.write_sample_file(source_path)
        ds = open_dataset(source_path, squeeze=False).assign_coords(level=[0.0, 0.0])

        with pytest.raises(ValueError, match="must not contain duplicate values"):
            write_dataset(ds, written_path)

    def test_write_dataset_rejects_nonfinite_values(self, tmp_path):
        source_path = tmp_path / "source.arl"
        written_path = tmp_path / "written.arl"

        self.write_sample_file(source_path)
        ds = open_dataset(source_path, squeeze=False).load()
        ds["PRSS"].data[0, 0, 0, 0] = np.inf

        with pytest.raises(ValueError, match="non-finite values"):
            write_dataset(ds, written_path)

    def test_write_dataset_rejects_long_variable_names(self, tmp_path):
        source_path = tmp_path / "source.arl"
        written_path = tmp_path / "written.arl"

        self.write_sample_file(source_path)
        ds = open_dataset(source_path, squeeze=False).rename({"PRSS": "PRESS"})

        with pytest.raises(ValueError, match="4 characters or fewer"):
            write_dataset(ds, written_path)

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
