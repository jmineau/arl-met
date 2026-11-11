"""Tests for dataset reading and low-level ARL plumbing."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core import indexing

import arlmet
from arlmet import File, open_dataset, write_dataset
from arlmet.grid import Grid, Projection
from arlmet.packing import pack, unpack
from arlmet.record import IndexRecord
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
        assert [name for name, _ in index0.levels[0].variables] == ["PRSS"]
        assert [name for name, _ in index0.levels[1].variables] == ["TEMP"]

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
        assert "forecast_hour" in ds.coords
        assert "forecast_hour" not in ds.data_vars
        assert ds["forecast_hour"].dims == ("time",)
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
        ``open_dataset`` reads a time step whose records differ in forecast hour.

        GDAS weekly files have them; opening one must not raise.
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

    def test_write_dataset_writes_the_index_forecast_hour_into_every_record(
        self, tmp_path
    ):
        """
        A round trip writes the time step's forecast hour into every record.

        The Dataset keeps only the index record's hour, so TEMP's own hour (3)
        becomes the index record's (0).
        """
        path = tmp_path / "mixed_forecast.arl"
        written_path = tmp_path / "written.arl"
        grid = make_test_grid()
        vertical_axis = PressureAxis(levels=[0.0, 1000.0])
        data = np.ones((grid.ny, grid.nx), dtype=np.float32)

        with File(
            path, mode="w", source="TEST", grid=grid, vertical_axis=vertical_axis
        ) as arl:
            rs = arl.create_recordset(pd.Timestamp("2025-09-01 00:00"), forecast=0)
            rs.create_datarecord("PRSS", level=0, forecast=0, data=data)
            rs.create_datarecord("TEMP", level=1, forecast=3, data=data)

        write_dataset(open_dataset(path), written_path)

        with File(written_path) as reopened:
            rs = reopened[0]
            assert rs.forecast == 0
            assert [record.forecast for record in rs.records] == [0, 0]

    def test_forecast_hour_coordinate_survives_arithmetic(self, tmp_path):
        source_path = tmp_path / "source.arl"
        written_path = tmp_path / "written.arl"

        self.write_sample_file(source_path)
        ds = open_dataset(source_path)
        shifted = ds + 1.0

        np.testing.assert_array_equal(shifted["forecast_hour"].values, [0, 3])
        write_dataset(shifted, written_path)
        with File(written_path) as reopened:
            assert [reopened[time].forecast for time in reopened.times] == [0, 3]

    def test_write_dataset_rejects_forecast_hour_as_a_data_variable(self, tmp_path):
        source_path = tmp_path / "source.arl"
        written_path = tmp_path / "written.arl"

        self.write_sample_file(source_path)
        ds = open_dataset(source_path).reset_coords("forecast_hour")
        assert "forecast_hour" in ds.data_vars

        with pytest.raises(ValueError, match="set_coords"):
            write_dataset(ds, written_path)
        assert not written_path.exists()

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

    def test_write_dataset_converts_aware_times_to_utc_without_warning(self, tmp_path):
        source_path = tmp_path / "source.arl"
        written_path = tmp_path / "written.arl"
        self.write_sample_file(source_path)
        ds = open_dataset(source_path).load()
        utc = pd.DatetimeIndex(ds.time.values)
        ds = ds.assign_coords(time=utc.tz_localize("UTC").tz_convert("America/Denver"))

        write_dataset(ds, written_path)  # warnings are errors in the tests

        assert list(File(written_path).times) == list(utc)

    @pytest.mark.parametrize("dim", ["lat", "lon"])
    def test_write_dataset_rejects_a_descending_coordinate(self, tmp_path, dim):
        """Rows and columns are written in array order, so a flipped Dataset is refused."""
        source_path = tmp_path / "source.arl"
        written_path = tmp_path / "written.arl"

        self.write_sample_file(source_path)
        flipped = open_dataset(source_path).isel({dim: slice(None, None, -1)})

        with pytest.raises(ValueError, match=rf"coordinate '{dim}' must increase"):
            write_dataset(flipped, written_path)
        assert not written_path.exists()

        write_dataset(flipped.sortby(dim), written_path)
        np.testing.assert_allclose(
            open_dataset(written_path)["TEMP"].values,
            open_dataset(source_path)["TEMP"].values,
        )

    @pytest.mark.parametrize("dim", ["lat", "lon"])
    def test_write_dataset_rejects_an_edited_coordinate(self, tmp_path, dim):
        """The grid is written from arl_grid, so a lat or lon that disagrees is refused."""
        source_path = tmp_path / "source.arl"
        written_path = tmp_path / "written.arl"
        self.write_sample_file(source_path)
        ds = open_dataset(source_path).load()

        shifted = ds.assign_coords({dim: ds[dim] + 10})
        with pytest.raises(ValueError, match=rf"coordinate '{dim}' disagrees"):
            write_dataset(shifted, written_path)
        assert not written_path.exists()

        # Irregular spacing is refused too, not written back as regular.
        values = ds[dim].values.copy()
        values[3] += 0.5 * (values[1] - values[0])
        irregular = ds.assign_coords({dim: (dim, values)})
        with pytest.raises(ValueError, match=rf"coordinate '{dim}' disagrees"):
            write_dataset(irregular, written_path)

        # Unedited, and with the coordinate dropped, both write.
        write_dataset(ds, written_path)
        write_dataset(ds.drop_vars(dim), written_path)

    def test_write_dataset_accepts_longitudes_a_turn_apart(self, tmp_path):
        source_path = tmp_path / "source.arl"
        written_path = tmp_path / "written.arl"
        self.write_sample_file(source_path)
        ds = open_dataset(source_path).load()

        write_dataset(ds.assign_coords(lon=ds.lon + 360), written_path)

        np.testing.assert_allclose(open_dataset(written_path).lon, ds.lon)

    def test_write_dataset_refuses_to_write_no_time_steps(self, tmp_path):
        source_path = tmp_path / "source.arl"
        written_path = tmp_path / "written.arl"

        self.write_sample_file(source_path)
        ds = open_dataset(source_path).isel(time=slice(0, 0))

        with pytest.raises(ValueError, match="no time steps"):
            write_dataset(ds, written_path)
        assert not written_path.exists()

    def test_write_dataset_defaults_missing_forecast_hours_to_zero(self, tmp_path):
        source_path = tmp_path / "source.arl"
        written_path = tmp_path / "written.arl"

        self.write_sample_file(source_path)
        ds = open_dataset(source_path).drop_vars("forecast_hour")

        write_dataset(ds, written_path)

        with File(written_path) as reopened:
            assert [reopened[time].forecast for time in reopened.times] == [0, 0]

    @pytest.mark.parametrize("name", ["O3", "T", "TMPXX"])
    def test_variable_names_not_four_characters_are_refused(self, tmp_path, name):
        """A short name would be padded in the file and fail to match when read back."""
        grid = make_test_grid()
        axis = PressureAxis(levels=[0.0, 1000.0])
        field = np.ones((20, 20), "float32")

        with File(
            tmp_path / "a.arl", mode="w", source="TEST", grid=grid, vertical_axis=axis
        ) as arl:
            rs = arl.create_recordset(pd.Timestamp("2024-07-18"), forecast=0)
            rs.create_datarecord("PRSS", level=0, forecast=0, data=field)
            with pytest.raises(ValueError, match="exactly 4 characters"):
                rs.create_datarecord(name, level=1, forecast=0, data=field)
            assert len(rs) == 1  # the refused record was not added

        source = tmp_path / "source.arl"
        self.write_sample_file(source)
        ds = open_dataset(source).load().rename({"TEMP": name})
        out = tmp_path / "b.arl"
        with pytest.raises(ValueError, match="exactly 4 characters"):
            write_dataset(ds, out)
        assert not out.exists()

    def test_masked_cells_are_missing_not_their_fill_value(self, tmp_path):
        """A masked array's mask counts; the 1e20 fill under it must not be packed."""
        grid = make_test_grid()
        axis = PressureAxis(levels=[0.0, 1000.0])
        field = np.ma.masked_array(np.full((20, 20), 280.0), mask=False)
        field.data[2:4, 3:6] = 1e20
        field[2:4, 3:6] = np.ma.masked

        # Some cells masked: refused, naming the field, instead of 1e20 packed.
        with (
            pytest.raises(ValueError, match="TEMP at level 1 has 6 missing"),
            File(
                tmp_path / "a.arl",
                mode="w",
                source="TEST",
                grid=grid,
                vertical_axis=axis,
            ) as arl,
        ):
            rs = arl.create_recordset(pd.Timestamp("2024-07-18"), forecast=0)
            rs.create_datarecord("PRSS", level=0, forecast=0, data=np.ones((20, 20)))
            rs.create_datarecord("TEMP", level=1, forecast=0, data=field)

        # Every cell masked: a missing field, as an all-NaN one is. Assigning
        # through record[:] reads the mask too.
        everything = np.ma.masked_all((20, 20))
        everything.data[:] = 1e20
        path = tmp_path / "b.arl"
        with File(path, mode="w", source="TEST", grid=grid, vertical_axis=axis) as arl:
            rs = arl.create_recordset(pd.Timestamp("2024-07-18"), forecast=0)
            rs.create_datarecord("PRSS", level=0, forecast=0, data=np.ones((20, 20)))
            rs.create_datarecord("TEMP", level=1, forecast=0, data=everything)
            record = rs.create_datarecord("RELH", level=1, forecast=0)
            record[:] = everything
        ds = open_dataset(path)
        assert np.isnan(ds["TEMP"].values).all()
        assert np.isnan(ds["RELH"].values).all()

    def test_create_datarecord_rejects_data_of_the_wrong_shape(self, tmp_path):
        grid = make_test_grid()
        axis = PressureAxis(levels=[0.0, 1000.0])
        path = tmp_path / "short.arl"

        with File(path, mode="w", source="TEST", grid=grid, vertical_axis=axis) as arl:
            rs = arl.create_recordset(pd.Timestamp("2024-07-18"), forecast=0)
            rs.create_datarecord(
                "PRSS", level=0, forecast=0, data=np.ones((20, 20), "float32")
            )
            with pytest.raises(ValueError, match=r"shape \(10, 20\).*\(20, 20\)"):
                rs.create_datarecord(
                    "TEMP", level=1, forecast=0, data=np.ones((10, 20), "float32")
                )
            assert len(rs) == 1  # the refused record was not added
            record = rs.create_datarecord("TEMP", level=1, forecast=0)
            with pytest.raises(ValueError, match="TEMP at level 1"):
                record[:] = np.ones((20, 10), "float32")
            record[:] = np.ones((20, 20), "float32")

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
        ds["TEMP"] = ds["TEMP"].where(ds["lat"] != ds["lat"][0])

        with pytest.raises(ValueError, match="contains missing values"):
            write_dataset(ds, written_path)

    def test_write_dataset_skips_all_nan_slices(self, tmp_path):
        """A variable stored on only some levels round-trips through a Dataset."""
        source_path = tmp_path / "partial.arl"
        written_path = tmp_path / "written.arl"
        grid = make_test_grid()
        vertical_axis = PressureAxis(levels=[0.0, 1000.0, 900.0])
        data = np.ones((grid.ny, grid.nx), dtype=np.float32)
        with File(
            source_path,
            mode="w",
            source="TEST",
            grid=grid,
            vertical_axis=vertical_axis,
        ) as arl:
            rs = arl.create_recordset(pd.Timestamp("2024-07-18 00:00"), forecast=0)
            rs.create_datarecord("TEMP", level=1, forecast=0, data=data)
            rs.create_datarecord("TEMP", level=2, forecast=0, data=2 * data)
            rs.create_datarecord("RELH", level=1, forecast=0, data=3 * data)

        ds = open_dataset(source_path)
        assert np.isnan(ds["RELH"].sel(level=2)).all()
        write_dataset(ds, written_path)

        with File(written_path) as reopened:
            records = reopened[reopened.times[0]].records
            assert [(r.level, r.variable) for r in records] == [
                (1, "RELH"),
                (1, "TEMP"),
                (2, "TEMP"),
            ]
        xr.testing.assert_identical(
            open_dataset(written_path).drop_encoding(), ds.drop_encoding()
        )

    @pytest.mark.parametrize("year", [1899, 1939, 2040, 2045])
    def test_times_outside_1940_to_2039_are_refused(self, tmp_path, year):
        grid = make_test_grid()
        axis = PressureAxis(levels=[0.0])
        time = pd.Timestamp(f"{year}-03-04 05:00")

        with File(
            tmp_path / "year.arl",
            mode="w",
            source="TEST",
            grid=grid,
            vertical_axis=axis,
        ) as arl:
            with pytest.raises(ValueError, match=rf"1940 to 2039; got the year {year}"):
                arl.create_recordset(time)
            with pytest.raises(ValueError, match="1940 to 2039"):
                arl.add_record(
                    time, "PRSS", 0, forecast=0, data=np.ones((20, 20), "float32")
                )

        source_path = tmp_path / "source.arl"
        self.write_surface_only_file(source_path)
        ds = open_dataset(source_path)
        ds = ds.assign_coords(
            time=ds.time.values - ds.time.values[0] + np.datetime64(time)
        )
        with pytest.raises(ValueError, match="1940 to 2039"):
            write_dataset(ds, tmp_path / "written.arl", vertical_axis=axis)
        assert not (tmp_path / "written.arl").exists()

    @pytest.mark.parametrize("year", [1940, 2039])
    def test_the_first_and_last_year_round_trip(self, tmp_path, year):
        grid = make_test_grid()
        axis = PressureAxis(levels=[0.0])
        time = pd.Timestamp(f"{year}-12-31 23:00")
        path = tmp_path / "year.arl"

        with File(path, mode="w", source="TEST", grid=grid, vertical_axis=axis) as arl:
            arl.add_record(
                time, "PRSS", 0, forecast=0, data=np.ones((20, 20), "float32")
            )

        with File(path) as arl:
            assert arl.times == [time]

    def test_write_dataset_compacts_level_subset(self, tmp_path):
        source_path = tmp_path / "levels.arl"
        written_path = tmp_path / "written.arl"
        grid = make_test_grid()
        vertical_axis = PressureAxis(levels=[0.0, 1000.0, 900.0, 800.0])
        data = np.ones((grid.ny, grid.nx), dtype=np.float32)
        with File(
            source_path,
            mode="w",
            source="TEST",
            grid=grid,
            vertical_axis=vertical_axis,
        ) as arl:
            rs = arl.create_recordset(pd.Timestamp("2024-07-18 00:00"), forecast=0)
            rs.create_datarecord("PRSS", level=0, forecast=0, data=data)
            for level in (1, 2, 3):
                rs.create_datarecord("TEMP", level=level, forecast=0, data=level * data)

        ds = open_dataset(source_path, levels=[0, 3])
        assert ds.arl.vertical_axis.levels.tolist() == [0.0, 800.0]
        write_dataset(ds, written_path)

        with File(written_path) as reopened:
            assert reopened.vertical_axis.levels.tolist() == [0.0, 800.0]
        reopened_ds = open_dataset(written_path)
        np.testing.assert_array_equal(reopened_ds["level"].values, [1])
        np.testing.assert_allclose(reopened_ds["TEMP"].values, ds["TEMP"].values)

        # ds.sel(level=...) on a full Dataset compacts the same way.
        full = open_dataset(source_path).sel(level=[2, 3])
        write_dataset(full, tmp_path / "sel.arl")
        with File(tmp_path / "sel.arl") as reopened:
            assert reopened.vertical_axis.levels.tolist() == [0.0, 900.0, 800.0]

    def test_write_dataset_rejects_overwriting_its_source(self, tmp_path):
        source_path = tmp_path / "source.arl"
        self.write_sample_file(source_path)
        size = source_path.stat().st_size
        ds = open_dataset(source_path)

        with pytest.raises(ValueError, match="same file as the input"):
            write_dataset(ds, source_path)
        assert source_path.stat().st_size == size

    def test_write_dataset_refuses_an_axis_that_relabels_the_levels(self, tmp_path):
        """The file's axis must be the one the Dataset's pressure coordinate describes."""
        source_path = tmp_path / "source.arl"
        written_path = tmp_path / "written.arl"
        self.write_sample_file(source_path)
        ds = open_dataset(source_path).load()  # one upper level, 1000 hPa

        with pytest.raises(ValueError, match=r"levels \[0.0, 925.0\].*'pressure'"):
            write_dataset(ds, written_path, vertical_axis=PressureAxis([0.0, 925.0]))
        with pytest.raises(ValueError, match="surface first"):
            write_dataset(ds, written_path, vertical_axis=PressureAxis([500.0, 1000.0]))
        flagged = ds.copy()
        flagged.attrs["vertical_flag"] = 1  # pressure values as a sigma axis
        with pytest.raises(ValueError, match="'pressure' does not match"):
            write_dataset(flagged, written_path)
        assert not written_path.exists()

        # The axis the coordinate describes, given explicitly, writes.
        write_dataset(ds, written_path, vertical_axis=PressureAxis([0.0, 1000.0]))
        assert File(written_path).vertical_axis.levels.tolist() == [0.0, 1000.0]

    def test_write_dataset_checks_explicit_vertical_axis_size(self, tmp_path):
        source_path = tmp_path / "source.arl"
        self.write_sample_file(source_path)
        ds = open_dataset(source_path)

        with pytest.raises(ValueError, match="vertical_axis has 3 levels"):
            write_dataset(
                ds,
                tmp_path / "written.arl",
                vertical_axis=PressureAxis(levels=[0.0, 1000.0, 900.0]),
            )

    def test_write_dataset_generates_diff_from_parent_attrs(self, tmp_path):
        source_path = tmp_path / "source.arl"
        written_path = tmp_path / "written.arl"

        self.write_sample_file(source_path)
        # DIFT corrects the TKE, so write the upper-air field as TKEN.
        ds = open_dataset(source_path).rename({"TEMP": "TKEN"})
        ds["TKEN"].attrs["diff"] = "DIFT"

        write_dataset(ds, written_path)

        with File(written_path) as reopened:
            record = reopened[
                self.write_sample_file.__self__
                if False
                else pd.Timestamp("2024-07-18 00:00")
            ]

        with File(written_path) as reopened:
            time0 = pd.Timestamp("2024-07-18 00:00")
            record = reopened[time0][(1, "TKEN")]
            assert record.diffs
            assert record.diffs[0].variable == "DIFT"

        reopened_ds = open_dataset(written_path)
        np.testing.assert_allclose(reopened_ds["TKEN"].values, ds["TKEN"].values)

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

    def test_write_dataset_rejects_a_diff_the_name_table_does_not_pair(self, tmp_path):
        source_path = tmp_path / "source.arl"
        written_path = tmp_path / "written.arl"

        self.write_sample_file(source_path)
        ds = open_dataset(source_path)
        ds["TEMP"].attrs["diff"] = "DIFT"

        with pytest.raises(ValueError, match="corrects TKEN or TKEU, not TEMP"):
            write_dataset(ds, written_path)
        assert not written_path.exists()

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

    def test_open_dataset_records_the_arlmet_version(self, tmp_path):
        path = tmp_path / "sample.arl"
        self.write_sample_file(path)
        assert open_dataset(path).attrs["arlmet_version"] == arlmet.__version__

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
    ) -> tuple[bytes, int, int, float, int, float]:
        packed, precision, exponent, initial_value = pack(data)
        return (
            packed.tobytes(),
            data.shape[1],
            data.shape[0],
            precision,
            exponent,
            initial_value,
        )


class TestForecastHourAbove99:
    """The label holds two digits of forecast hour, the index record three."""

    HOUR = pd.Timestamp("2025-09-01 00:00")

    @staticmethod
    def write_hour_100_file(path):
        """Write a file as ``pakrec`` does for hour 100: 99 in every label, 100 in the index."""
        grid = make_test_grid()
        data = np.arange(grid.nx * grid.ny, dtype=np.float32).reshape(grid.ny, grid.nx)
        with File(
            path,
            mode="w",
            source="TEST",
            grid=grid,
            vertical_axis=PressureAxis(levels=[0.0, 1000.0]),
        ) as arl:
            rs = arl.create_recordset(TestForecastHourAbove99.HOUR, forecast=99)
            rs.create_datarecord("PRSS", level=0, forecast=99, data=data)
            rs.create_datarecord("TEMP", level=1, forecast=99, data=data + 200.0)
        raw = bytearray(path.read_bytes())
        assert raw[54:57] == b" 99"  # the index record's forecast field
        raw[54:57] = b"100"
        path.write_bytes(raw)

    @staticmethod
    def assert_hour_100(path):
        with File(path) as arl:
            assert arl[0].forecast == 100
            assert [record.forecast for record in arl[0].records] == [99, 99]
        with open(path, "rb") as handle:
            index = IndexRecord.from_position(handle, 0)
        assert index.forecast == 100
        assert index.header.forecast == 99

    def test_the_helper_file_reads_as_hysplit_writes_it(self, tmp_path):
        path = tmp_path / "h100.arl"
        self.write_hour_100_file(path)
        self.assert_hour_100(path)

    def test_file_writer_splits_the_hour_between_label_and_index_record(self, tmp_path):
        path = tmp_path / "written.arl"
        grid = make_test_grid()
        data = np.ones((grid.ny, grid.nx), dtype=np.float32)
        with File(
            path,
            mode="w",
            source="TEST",
            grid=grid,
            vertical_axis=PressureAxis(levels=[0.0, 1000.0]),
        ) as arl:
            rs = arl.create_recordset(self.HOUR, forecast=100)
            rs.create_datarecord("PRSS", level=0, forecast=100, data=data)
            rs.create_datarecord("TEMP", level=1, forecast=100, data=data)
        self.assert_hour_100(path)
        # the label bytes themselves: characters 8-9 of the first data record
        record_length = 50 + grid.nx * grid.ny
        raw = path.read_bytes()
        assert raw[8:10] == b"99"
        assert raw[record_length + 8 : record_length + 10] == b"99"

    def test_extract_subset_levels_keeps_the_hour(self, tmp_path):
        from arlmet import extract_subset

        path, out = tmp_path / "h100.arl", tmp_path / "levels.arl"
        self.write_hour_100_file(path)
        extract_subset(path, out, levels=[0, 1])
        self.assert_hour_100(out)

    def test_extract_subset_bbox_keeps_the_hour(self, tmp_path):
        from arlmet import extract_subset

        path, out = tmp_path / "h100.arl", tmp_path / "crop.arl"
        self.write_hour_100_file(path)
        extract_subset(path, out, bbox=(20.0, -10.0, 36.0, 6.0))
        self.assert_hour_100(out)

    def test_write_dataset_keeps_the_hour(self, tmp_path):
        path, out = tmp_path / "h100.arl", tmp_path / "rt.arl"
        self.write_hour_100_file(path)
        ds = open_dataset(path)
        np.testing.assert_array_equal(ds["forecast_hour"].values, [100])
        write_dataset(ds, out)
        self.assert_hour_100(out)

    def test_hour_above_999_raises(self, tmp_path):
        grid = make_test_grid()
        data = np.ones((grid.ny, grid.nx), dtype=np.float32)
        with (
            pytest.raises(ValueError, match="999"),
            File(
                tmp_path / "bad.arl",
                mode="w",
                source="TEST",
                grid=grid,
                vertical_axis=PressureAxis(levels=[0.0]),
            ) as arl,
        ):
            rs = arl.create_recordset(self.HOUR, forecast=1000)
            rs.create_datarecord("PRSS", level=0, forecast=99, data=data)
