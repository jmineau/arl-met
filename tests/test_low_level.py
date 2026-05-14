"""Tests for low-level record, recordset, file, and xarray helper behavior."""

import io
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from arlmet import File
from arlmet.grid import Grid, GridWindow, Projection
from arlmet.header import Header
from arlmet.index import LvlInfo, VarInfo
from arlmet.packing import unpack
from arlmet.record import DataRecord
from arlmet.recordset import VariableAccessor
from arlmet.vertical import VerticalAxis
from arlmet.xarray._backend import ArlVariableArray, _normalize_backend_indexer


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


def write_single_record_file(path):
    grid = make_test_grid()
    vertical_axis = VerticalAxis(flag=2, levels=[1000.0])
    time = pd.Timestamp("2024-07-18 00:00")
    data = np.arange(grid.nx * grid.ny, dtype=np.float32).reshape(grid.ny, grid.nx)

    with File(
        path,
        mode="w",
        source="TEST",
        grid=grid,
        vertical_axis=vertical_axis,
    ) as arl:
        rs = arl.create_recordset(time)
        rs.create_datarecord("TEMP", level=0, forecast=0, data=data)

    return time, data


def write_terrain_file(path, times: list[pd.Timestamp]):
    grid = make_test_grid()
    vertical_axis = VerticalAxis(flag=2, levels=[0.0])
    terrain = np.arange(grid.nx * grid.ny, dtype=np.float32).reshape(grid.ny, grid.nx)

    with File(
        path,
        mode="w",
        source="TEST",
        grid=grid,
        vertical_axis=vertical_axis,
    ) as arl:
        for forecast, time in enumerate(times):
            rs = arl.create_recordset(time)
            rs.create_datarecord("SHGT", level=0, forecast=forecast, data=terrain)

    return terrain


def write_variable_view_file(path):
    grid = make_test_grid()
    vertical_axis = VerticalAxis(flag=2, levels=[1000.0, 900.0])
    times = [
        pd.Timestamp("2024-07-18 00:00"),
        pd.Timestamp("2024-07-18 03:00"),
    ]

    with File(
        path,
        mode="w",
        source="TEST",
        grid=grid,
        vertical_axis=vertical_axis,
    ) as arl:
        for forecast, time in enumerate(times):
            rs = arl.create_recordset(time)
            for level in range(2):
                base = forecast * 100 + level * 10
                temp = np.full((grid.ny, grid.nx), base + 1, dtype=np.float32)
                uwnd = np.full((grid.ny, grid.nx), base + 2, dtype=np.float32)
                rs.create_datarecord("TEMP", level=level, forecast=forecast, data=temp)
                rs.create_datarecord("UWND", level=level, forecast=forecast, data=uwnd)

    return times


class StubRecord:
    def __init__(self, field: np.ndarray):
        self.field = field

    def read(self, window=None):
        assert window is None
        return self.field


class StubDiff:
    def __init__(self, field: np.ndarray):
        self.field = field

    def read(self, window=None):
        return self.field

    def _load_from_disk(self, driver=None):
        return self.field


class TestDataRecordModeGuards:
    def test_datarecord_constructor_validates_forecast_for_read_and_write(
        self, tmp_path
    ):
        path = tmp_path / "constructor.arl"
        time, _ = write_single_record_file(path)

        grid = make_test_grid()
        vertical_axis = VerticalAxis(flag=2, levels=[1000.0])
        writable = File(
            tmp_path / "write_constructor.arl",
            mode="w",
            source="TEST",
            grid=grid,
            vertical_axis=vertical_axis,
        )

        try:
            rs_write = writable.create_recordset(pd.Timestamp("2024-07-18 00:00"))
            with pytest.raises(ValueError, match="Forecast hour must be specified"):
                DataRecord(rs_write, position=-1, level=0, variable="TEMP")
        finally:
            writable._manager.close()

        with File(path) as arl:
            rs_read = arl[time]
            record = rs_read[(0, "TEMP")]
            with pytest.raises(ValueError, match="should not be specified"):
                DataRecord(
                    rs_read,
                    position=record.position,
                    level=0,
                    variable="TEMP",
                    forecast=0,
                )

    def test_readable_record_rejects_item_assignment(self, tmp_path):
        path = tmp_path / "read_only.arl"
        time, data = write_single_record_file(path)

        with File(path) as arl:
            record = arl[time][(0, "TEMP")]

            with pytest.raises(io.UnsupportedOperation):
                record[:] = np.zeros_like(data)

    def test_writable_record_rejects_read(self, tmp_path):
        path = tmp_path / "write_only.arl"
        grid = make_test_grid()
        vertical_axis = VerticalAxis(flag=2, levels=[1000.0])

        with File(
            path,
            mode="w",
            source="TEST",
            grid=grid,
            vertical_axis=vertical_axis,
        ) as arl:
            rs = arl.create_recordset(pd.Timestamp("2024-07-18 00:00"))
            record = rs.create_datarecord(
                "TEMP",
                level=0,
                forecast=0,
                data=np.zeros((grid.ny, grid.nx), dtype=np.float32),
            )

            with pytest.raises(io.UnsupportedOperation):
                record.read()

    def test_writable_record_requires_full_initialization_before_slice_assignment(
        self, tmp_path
    ):
        path = tmp_path / "uninitialized.arl"
        grid = make_test_grid()
        vertical_axis = VerticalAxis(flag=2, levels=[1000.0])

        arl = File(
            path,
            mode="w",
            source="TEST",
            grid=grid,
            vertical_axis=vertical_axis,
        )

        try:
            rs = arl.create_recordset(pd.Timestamp("2024-07-18 00:00"))
            record = rs.create_datarecord("TEMP", level=0, forecast=0)

            with pytest.raises(ValueError, match="initialized before setting slices"):
                record[0, 0] = 1.0
        finally:
            arl._manager.close()

    def test_writable_record_header_validation_branches(self, tmp_path):
        path = tmp_path / "header_validation.arl"
        grid = make_test_grid()
        vertical_axis = VerticalAxis(flag=2, levels=[1000.0])
        arl = File(
            path, mode="w", source="TEST", grid=grid, vertical_axis=vertical_axis
        )

        try:
            rs = arl.create_recordset(pd.Timestamp("2024-07-18 00:00"))
            record = rs.create_datarecord(
                "TEMP",
                level=0,
                forecast=0,
                data=np.zeros((grid.ny, grid.nx), dtype=np.float32),
            )

            record._header = None
            with pytest.raises(ValueError, match="Header state must be a dictionary"):
                _ = record.header

            record._header = {"precision": 1.0, "exponent": 0, "initial_value": 0.0}
            with pytest.raises(ValueError, match="Forecast hour must be set"):
                _ = record.header

            record._header = {
                "forecast": 0,
                "precision": None,
                "exponent": 0,
                "initial_value": 0.0,
            }
            with pytest.raises(
                ValueError,
                match="Precision, exponent, and initial value must be set",
            ):
                _ = record.header

            record._header = {"forecast": 0}
            record[:] = np.zeros((grid.ny, grid.nx), dtype=np.float32)
            record._create_diff(position=-1, variable="DIFW", forecast=0)
            record._derive_diff_on_pack = True
            assert record.header.variable == "TEMP"
        finally:
            arl._manager.close()

    def test_record_diff_and_checksum_validation_branches(self, tmp_path):
        path = tmp_path / "record_branches.arl"
        time, _ = write_single_record_file(path)

        with File(path) as arl:
            record = arl[time][(0, "TEMP")]

            diff = record._create_diff(
                position=record.position + record.n_bytes, variable="DIFW"
            )
            assert diff.level == record.level
            with pytest.raises(ValueError, match="Difference record already exists"):
                record._create_diff(
                    position=record.position + 2 * record.n_bytes, variable="DIFX"
                )
            with pytest.raises(ValueError, match="must start with 'DIF'"):
                diff._create_diff(
                    position=record.position + 3 * record.n_bytes, variable="TEMP"
                )

            record._checksum = None
            with pytest.raises(ValueError, match="No stored checksum"):
                _ = record.checksum

        grid = make_test_grid()
        vertical_axis = VerticalAxis(flag=2, levels=[1000.0])
        writable = File(
            tmp_path / "record_branches_write.arl",
            mode="w",
            source="TEST",
            grid=grid,
            vertical_axis=vertical_axis,
        )

        try:
            rs = writable.create_recordset(pd.Timestamp("2024-07-18 00:00"))
            record = rs.create_datarecord("TEMP", level=0, forecast=0)

            with pytest.raises(
                ValueError, match="Data must be packed before accessing checksum"
            ):
                _ = record.checksum

            with pytest.raises(ValueError, match="No data to read"):
                _ = record.data

            with pytest.raises(ValueError, match="Data to pack must be a numpy array"):
                record._pack()

            record._unpacked = np.zeros((grid.ny, grid.nx), dtype=np.float32)
            record._header = Header(
                year=2024,
                month=7,
                day=18,
                hour=0,
                forecast=0,
                level=0,
                grid=(grid.nx, grid.ny),
                variable="TEMP",
                exponent=0,
                precision=1.0,
                initial_value=0.0,
            )
            with pytest.raises(ValueError, match="Header must be a dictionary"):
                record._pack()
        finally:
            writable._manager.close()

    def test_write_roundtrip_with_generated_diff_record(self, tmp_path):
        path = tmp_path / "generated_diff.arl"
        grid = make_test_grid()
        vertical_axis = VerticalAxis(flag=2, levels=[1000.0])
        time = pd.Timestamp("2024-07-18 00:00")
        base = (
            0.123
            + np.arange(grid.nx * grid.ny, dtype=np.float32).reshape(grid.ny, grid.nx)
            * 0.0073
        )

        with File(
            path,
            mode="w",
            source="TEST",
            grid=grid,
            vertical_axis=vertical_axis,
        ) as arl:
            rs = arl.create_recordset(time)
            record = rs.create_datarecord(
                "WWND", level=0, forecast=0, data=base, diff="DIFW"
            )
            assert record.diff is not None
            assert record.diff.variable == "DIFW"

        with File(path) as reopened:
            record = reopened[time][(0, "WWND")]
            assert record.diff is not None
            assert record.diff.variable == "DIFW"
            combined = record.read()
            parent_only = unpack(
                packed=record.bytes[Header.N_BYTES :],
                nx=grid.nx,
                ny=grid.ny,
                precision=record.header.precision,
                exponent=record.header.exponent,
                initial_value=record.header.initial_value,
                driver=np,
            )
            np.testing.assert_allclose(combined, base, atol=record.header.precision)
            np.testing.assert_allclose(
                np.asarray(parent_only, dtype=np.float32) + record.diff.read(),
                combined,
                atol=max(record.header.precision, record.diff.header.precision),
            )

    def test_write_rejects_conflicting_diff_bindings(self, tmp_path):
        path = tmp_path / "conflicting_diff.arl"
        grid = make_test_grid()
        vertical_axis = VerticalAxis(flag=2, levels=[1000.0])
        data = np.ones((grid.ny, grid.nx), dtype=np.float32)

        with File(
            path,
            mode="w",
            source="TEST",
            grid=grid,
            vertical_axis=vertical_axis,
        ) as arl:
            rs = arl.create_recordset(pd.Timestamp("2024-07-18 00:00"))
            rs.create_datarecord("WWND", level=0, forecast=0, data=data, diff="DIFW")
            with pytest.raises(ValueError, match="already bound to parent"):
                rs.create_datarecord(
                    "TEMP", level=0, forecast=0, data=data, diff="DIFW"
                )

    def test_write_rejects_invalid_diff_name(self, tmp_path):
        path = tmp_path / "invalid_diff.arl"
        grid = make_test_grid()
        vertical_axis = VerticalAxis(flag=2, levels=[1000.0])
        data = np.ones((grid.ny, grid.nx), dtype=np.float32)

        with File(
            path,
            mode="w",
            source="TEST",
            grid=grid,
            vertical_axis=vertical_axis,
        ) as arl:
            rs = arl.create_recordset(pd.Timestamp("2024-07-18 00:00"))
            with pytest.raises(ValueError, match="must start with 'DIF'"):
                rs.create_datarecord(
                    "WWND", level=0, forecast=0, data=data, diff="WDIFF"
                )

    def test_writable_record_rejects_non_mutable_header_after_pack(
        self, tmp_path, monkeypatch
    ):
        path = tmp_path / "mutable_header.arl"
        grid = make_test_grid()
        vertical_axis = VerticalAxis(flag=2, levels=[1000.0])
        arl = File(
            path, mode="w", source="TEST", grid=grid, vertical_axis=vertical_axis
        )

        try:
            rs = arl.create_recordset(pd.Timestamp("2024-07-18 00:00"))
            record = rs.create_datarecord(
                "TEMP",
                level=0,
                forecast=0,
                data=np.zeros((grid.ny, grid.nx), dtype=np.float32),
            )

            def fake_pack():
                record._header = Header(
                    year=2024,
                    month=7,
                    day=18,
                    hour=0,
                    forecast=0,
                    level=0,
                    grid=(grid.nx, grid.ny),
                    variable="TEMP",
                    exponent=0,
                    precision=1.0,
                    initial_value=0.0,
                )
                return np.zeros(grid.nx * grid.ny, dtype=np.uint8)

            monkeypatch.setattr(record, "_pack", fake_pack)
            record._header = {"forecast": 0}

            with pytest.raises(ValueError, match="must remain mutable"):
                _ = record.header
        finally:
            arl._manager.close()

    def test_read_and_load_from_disk_handle_diff_header_mismatch_and_array_copy(
        self, tmp_path
    ):
        path = tmp_path / "read_behaviors.arl"
        time, data = write_single_record_file(path)

        with File(path) as arl:
            record = arl[time][(0, "TEMP")]
            diff = np.ones_like(data)
            record._diff = StubDiff(diff)

            loaded = record._load_from_disk()
            np.testing.assert_allclose(loaded, data + diff)

            arr = np.array(record, dtype=np.float64, copy=True)
            assert arr.dtype == np.float64
            np.testing.assert_allclose(arr, data + diff)

        with File(path) as arl:
            record = arl[time][(0, "TEMP")]
            assert record.source == "TEST"
            assert record.time == time
            assert record.shape == (20, 20)
            assert record.dtype == np.float32
            first = record.read()
            second = record.read()
            assert second is first
            subset = record.read(window=GridWindow(0, 2, 0, 2))
            np.testing.assert_allclose(subset, data[:2, :2])
            assert record[0, 0] == data[0, 0]
            original_checksum = record.checksum
            record._checksum = original_checksum + 1
            assert record.verify_checksum() is False

        with File(path) as arl:
            record = arl[time][(0, "TEMP")]
            record.variable = "WIND"
            with pytest.raises(ValueError, match="header mismatch"):
                _ = record.header

        with File(path) as arl:
            record = arl[time][(0, "TEMP")]
            _ = record.bytes
            _ = record.header
            original_forecast = record.forecast
            record._invalidate_write_cache()
            assert (
                record._header == record._header
            )  # keep linter happy about branch use
            assert record._bytes is not None  # no-op in read mode
            assert record.forecast == original_forecast

    def test_writable_record_invalidate_cache_and_flush_existing_position(
        self, tmp_path
    ):
        path = tmp_path / "flush_record.arl"
        grid = make_test_grid()
        vertical_axis = VerticalAxis(flag=2, levels=[1000.0])
        arl = File(
            path, mode="w", source="TEST", grid=grid, vertical_axis=vertical_axis
        )

        try:
            rs = arl.create_recordset(pd.Timestamp("2024-07-18 00:00"))
            record = rs.create_datarecord(
                "TEMP",
                level=0,
                forecast=0,
                data=np.zeros((grid.ny, grid.nx), dtype=np.float32),
            )

            _ = record.bytes
            assert isinstance(record._header, Header)
            record._invalidate_write_cache()
            assert record._packed is None
            assert record._bytes is None
            assert record._checksum is None
            assert record._header == {"forecast": 0}

            record[:] = np.ones((grid.ny, grid.nx), dtype=np.float32)
            record._flush()
            first_position = record.position
            assert first_position >= 0
            assert record.mode == "r"
            with pytest.raises(io.UnsupportedOperation):
                record[:] = np.full((grid.ny, grid.nx), 2.0, dtype=np.float32)
        finally:
            arl._manager.close()


class TestRecordCollectionForecasts:
    def test_recordset_derives_index_forecast_from_mixed_record_forecasts(
        self, tmp_path
    ):
        path = tmp_path / "derived_forecast.arl"
        grid = make_test_grid()
        vertical_axis = VerticalAxis(flag=2, levels=[1000.0])
        zeros = np.zeros((grid.ny, grid.nx), dtype=np.float32)

        with File(
            path,
            mode="w",
            source="TEST",
            grid=grid,
            vertical_axis=vertical_axis,
        ) as arl:
            rs = arl.create_recordset(pd.Timestamp("2024-07-18 00:00"))
            rs.create_datarecord("TEMP", level=0, forecast=3, data=zeros)
            rs.create_datarecord("UWND", level=0, forecast=0, data=zeros)

        with File(path) as arl:
            assert arl[pd.Timestamp("2024-07-18 00:00")].forecast == 0

    def test_file_add_record_creates_recordsets_and_derives_file_records(
        self, tmp_path
    ):
        path = tmp_path / "add_record.arl"
        grid = make_test_grid()
        vertical_axis = VerticalAxis(flag=2, levels=[1000.0, 900.0])
        zeros = np.zeros((grid.ny, grid.nx), dtype=np.float32)
        time0 = pd.Timestamp("2024-07-18 00:00")
        time1 = pd.Timestamp("2024-07-18 03:00")

        with File(
            path,
            mode="w",
            source="TEST",
            grid=grid,
            vertical_axis=vertical_axis,
        ) as arl:
            arl.add_record(time0, "TEMP", level=0, forecast=0, data=zeros)
            arl.add_record(time0, "UWND", level=1, forecast=3, data=zeros)
            arl.add_record(time1, "TEMP", level=0, forecast=6, data=zeros)

            assert len(arl.records) == 3
            assert [record.time for record in arl.records] == [time0, time0, time1]
            assert arl[time0].forecast is None

        with File(path) as arl:
            assert [record.time for record in arl.records] == [time0, time0, time1]
            assert arl[time0].forecast == 0
            assert arl[time1].forecast == 6

    def test_file_add_record_respects_explicit_recordset_forecast(self, tmp_path):
        path = tmp_path / "add_record_explicit_forecast.arl"
        grid = make_test_grid()
        vertical_axis = VerticalAxis(flag=2, levels=[1000.0])
        zeros = np.zeros((grid.ny, grid.nx), dtype=np.float32)
        time0 = pd.Timestamp("2024-07-18 00:00")

        with File(
            path,
            mode="w",
            source="TEST",
            grid=grid,
            vertical_axis=vertical_axis,
        ) as arl:
            arl.create_recordset(time0, forecast=9)
            arl.add_record(time0, "TEMP", level=0, forecast=0, data=zeros)

            assert arl[time0].forecast == 9

        with File(path) as arl:
            assert arl[time0].forecast == 9


class TestVariableViewsAndAccessors:
    def test_variable_accessor_reports_names_iter_len_and_repr(self, tmp_path):
        path = tmp_path / "variables.arl"
        write_variable_view_file(path)

        with File(path) as arl:
            names = set(iter(arl.variables))
            assert names == {"TEMP", "UWND"}
            assert len(arl.variables) == 2
            assert repr(arl.variables) in {
                "VariableAccessor(['TEMP', 'UWND'])",
                "VariableAccessor(['UWND', 'TEMP'])",
            }

    def test_variable_accessor_requires_source_records_attribute(self):
        accessor = VariableAccessor(source=object())

        with pytest.raises(TypeError, match="records"):
            _ = accessor._names

    def test_variable_view_exposes_shape_dtype_array_and_slice(self, tmp_path):
        path = tmp_path / "variables.arl"
        write_variable_view_file(path)

        with File(path) as arl:
            view = arl.variables["TEMP"]

            assert view.dtype == np.float32
            assert view.ndim == 4
            assert view.shape == (2, 2, 20, 20)

            array = np.asarray(view)
            assert array.shape == (2, 2, 20, 20)
            np.testing.assert_allclose(
                array[0, 0], 1.0
            )  # time=0, level idx 0 → 1000 hPa
            np.testing.assert_allclose(
                array[0, 1], 11.0
            )  # time=0, level idx 1 → 900 hPa
            np.testing.assert_allclose(view[1, 1, 0, :3], [111.0, 111.0, 111.0])

    def test_recordset_variable_view_mapping_helpers(self, tmp_path):
        path = tmp_path / "variables.arl"
        times = write_variable_view_file(path)

        with File(path) as arl:
            rs = arl[times[0]]
            view = rs.variables["TEMP"]

            assert view.ndim == 4
            assert view.shape == (1, 2, 20, 20)

            assert len(rs) == 4
            assert all(record.recordset is rs for record in rs)

            with pytest.raises(KeyError, match=r"tuple of \(level, variable\)"):
                rs[0]


class TestFileLowLevelBehavior:
    def test_file_validates_mode_and_setter_types(self, tmp_path):
        with pytest.raises(ValueError, match="Mode must be 'r' .* 'w'"):
            File(tmp_path / "bad.arl", mode="a")

        arl = File(tmp_path / "setters.arl", mode="w")
        try:
            arl.source = "TEST"
            assert arl.source == "TEST"

            with pytest.raises(TypeError, match="grid must be a Grid instance"):
                arl.grid = "bad"

            with pytest.raises(TypeError, match="vertical_axis must be a VerticalAxis"):
                arl.vertical_axis = "bad"

            axis = VerticalAxis(flag=2, levels=[1000.0])
            arl.vertical_axis = axis
            assert arl.vertical_axis == axis
        finally:
            arl._manager.close()

    def test_unset_file_metadata_properties_raise_until_configured(self, tmp_path):
        arl = File(tmp_path / "blank.arl", mode="w")

        try:
            with pytest.raises(ValueError, match="Source has not been set"):
                _ = arl.source

            with pytest.raises(ValueError, match="Grid has not been set"):
                _ = arl.grid

            with pytest.raises(ValueError, match="Vertical axis has not been set"):
                _ = arl.vertical_axis
        finally:
            arl._manager.close()

    def test_read_only_file_rejects_write_only_operations(self, tmp_path):
        path = tmp_path / "read_only_file.arl"
        time, _ = write_single_record_file(path)

        with File(path) as arl:
            with pytest.raises(io.UnsupportedOperation):
                arl.create_recordset(time)

            with pytest.raises(io.UnsupportedOperation):
                arl.source = "NEW"

    def test_create_grid_and_recordset_internal_validations(self, tmp_path):
        arl = File(tmp_path / "create_grid.arl", mode="w", source="TEST")
        try:
            grid = arl.create_grid(
                20,
                20,
                90.0,
                0.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                -10.0,
                20.0,
            )
            assert grid == arl.grid

            with pytest.raises(ValueError, match="Grid has already been set"):
                arl.create_grid(
                    20,
                    20,
                    90.0,
                    0.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    -10.0,
                    20.0,
                )

            time = pd.Timestamp("2024-07-18 00:00")
            arl.vertical_axis = VerticalAxis(flag=2, levels=[1000.0])
            arl._create_recordset(position=-1, source=None, grid=None, time=time)

            with pytest.raises(ValueError, match="already exists"):
                arl._create_recordset(position=-1, source=None, grid=None, time=time)

            with pytest.raises(ValueError, match="Source mismatch"):
                arl._create_recordset(
                    position=-1,
                    source="NOPE",
                    grid=None,
                    time=pd.Timestamp("2024-07-18 03:00"),
                )

            with pytest.raises(ValueError, match="Grid mismatch"):
                arl._create_recordset(
                    position=-1,
                    source=None,
                    grid=make_test_grid(nx=21, ny=20),
                    time=pd.Timestamp("2024-07-18 06:00"),
                )
        finally:
            arl._manager.close()

    def test_create_recordset_requires_configured_source_and_grid(self, tmp_path):
        arl = File(tmp_path / "needs_source.arl", mode="w")
        try:
            with pytest.raises(ValueError, match="Source has not been set"):
                arl.create_recordset(pd.Timestamp("2024-07-18 00:00"))
        finally:
            arl._manager.close()

    def test_file_getitem_supports_integer_and_string_lookup(self, tmp_path):
        path = tmp_path / "terrain_lookup.arl"
        times = [
            pd.Timestamp("2024-07-18 00:00"),
            pd.Timestamp("2024-07-18 03:00"),
        ]
        write_terrain_file(path, times)

        with File(path) as arl:
            assert arl[0].time == times[0]
            assert arl[str(times[1])].time == times[1]
            assert len(arl) == 2
            assert list(iter(arl)) == times

    def test_scan_rejects_vertical_axis_mismatch_between_index_records(
        self, tmp_path, monkeypatch
    ):
        path = tmp_path / "scan_vertical.arl"
        path.write_bytes(b"\0" * 2000)
        grid = make_test_grid()
        first = SimpleNamespace(
            source="TEST",
            grid=grid,
            vertical_axis=VerticalAxis(flag=2, levels=[1000.0]),
            time=pd.Timestamp("2024-07-18 00:00"),
            forecast=0,
            levels=[],
        )
        second = SimpleNamespace(
            source="TEST",
            grid=grid,
            vertical_axis=VerticalAxis(flag=2, levels=[900.0]),
            time=pd.Timestamp("2024-07-18 03:00"),
            forecast=0,
            levels=[],
        )
        indices = iter([first, second])

        def fake_from_position(fh, position):
            return next(indices)

        monkeypatch.setattr("arlmet.file.IndexRecord.from_position", fake_from_position)

        with pytest.raises(ValueError, match="Vertical axis mismatch"):
            File(path)

    def test_scan_rejects_diff_record_without_preceding_record(
        self, tmp_path, monkeypatch
    ):
        path = tmp_path / "scan_diff.arl"
        path.write_bytes(b"\0" * 2000)
        grid = make_test_grid()
        index = SimpleNamespace(
            source="TEST",
            grid=grid,
            vertical_axis=VerticalAxis(flag=2, levels=[1000.0]),
            time=pd.Timestamp("2024-07-18 00:00"),
            forecast=0,
            levels=[
                LvlInfo(
                    level=0,
                    height=1000.0,
                    variables=OrderedDict([("DIFW", VarInfo(checksum=0, reserved=""))]),
                )
            ],
        )

        def fake_from_position(fh, position):
            return index

        monkeypatch.setattr("arlmet.file.IndexRecord.from_position", fake_from_position)

        with pytest.raises(ValueError, match="Difference record found"):
            File(path)

    def test_scan_stops_cleanly_on_eof_and_assigns_diff_after_data_record(
        self, tmp_path, monkeypatch
    ):
        eof_path = tmp_path / "scan_eof.arl"
        eof_path.write_bytes(b"\0" * 2000)

        def eof_from_position(fh, position):
            raise EOFError

        monkeypatch.setattr("arlmet.file.IndexRecord.from_position", eof_from_position)
        arl = File(eof_path)
        assert len(arl) == 0
        arl.close()

        diff_path = tmp_path / "scan_diff_after_data.arl"
        diff_path.write_bytes(b"\0" * 2000)
        grid = make_test_grid()
        index = SimpleNamespace(
            source="TEST",
            grid=grid,
            vertical_axis=VerticalAxis(flag=2, levels=[1000.0]),
            time=pd.Timestamp("2024-07-18 00:00"),
            forecast=0,
            levels=[
                LvlInfo(
                    level=0,
                    height=1000.0,
                    variables=OrderedDict(
                        [
                            ("TEMP", VarInfo(checksum=1, reserved="")),
                            ("DIFW", VarInfo(checksum=2, reserved="R")),
                        ]
                    ),
                )
            ],
        )
        called = {"count": 0}

        def diff_from_position(fh, position):
            if called["count"] == 0:
                called["count"] += 1
                return index
            raise EOFError

        monkeypatch.setattr("arlmet.file.IndexRecord.from_position", diff_from_position)
        with File(diff_path) as arl:
            record = arl[pd.Timestamp("2024-07-18 00:00")][(0, "TEMP")]
            assert record._diff is not None
            assert record._diff.variable == "DIFW"


class TestReprs:
    def test_datarecord_repr(self, tmp_path):
        path = tmp_path / "repr.arl"
        time, _ = write_single_record_file(path)
        with File(path) as arl:
            record = arl[time][(0, "TEMP")]
            r = repr(record)
        assert r == "DataRecord('TEMP', level=0, time=2024-07-18 00:00)"

    def test_recordset_repr(self, tmp_path):
        path = tmp_path / "repr.arl"
        time, _ = write_single_record_file(path)
        with File(path) as arl:
            rs = arl[time]
            r = repr(rs)
        assert r == "RecordSet(time=2024-07-18 00:00, forecast=0, n=1)"

    def test_recordset_contains_variable(self, tmp_path):
        path = tmp_path / "repr.arl"
        time, _ = write_single_record_file(path)
        with File(path) as arl:
            rs = arl[time]
            assert "TEMP" in rs
            assert "UWND" not in rs
            assert 42 not in rs  # non-string returns False

    def test_file_repr_read_mode(self, tmp_path):
        path = tmp_path / "repr.arl"
        write_single_record_file(path)
        with File(path) as arl:
            r = repr(arl)
        assert r == "File('repr.arl', mode='r', times=1, grid=20×20, levels=1)"

    def test_file_repr_write_mode_none_grid_and_levels(self, tmp_path):
        path = tmp_path / "repr_w.arl"
        arl = File(path, mode="w")
        try:
            r = repr(arl)
        finally:
            arl._manager.close()
        assert r == "File('repr_w.arl', mode='w', times=0, grid=None, levels=None)"

    def test_file_contains_timestamp(self, tmp_path):
        path = tmp_path / "repr.arl"
        write_single_record_file(path)
        ts = pd.Timestamp("2024-07-18 00:00")
        with File(path) as arl:
            assert ts in arl
            assert pd.Timestamp("2000-01-01") not in arl
            assert "not-a-timestamp" not in arl

    def test_variable_view_repr(self, tmp_path):
        path = tmp_path / "vv.arl"
        write_single_record_file(path)
        with File(path) as arl:
            r = repr(arl.variables["TEMP"])
        assert r == "VariableView('TEMP', shape=(1, 1, 20, 20))"


class TestXarrayBackendHelpers:
    def test_normalize_backend_indexer_supports_scalar_slice_and_array(self):
        scalar_idx, scalar = _normalize_backend_indexer(2, 5)
        np.testing.assert_array_equal(scalar_idx, [2])
        assert scalar is True

        slice_idx, scalar = _normalize_backend_indexer(slice(1, None, 2), 5)
        np.testing.assert_array_equal(slice_idx, [1, 3])
        assert scalar is False

        array_idx, scalar = _normalize_backend_indexer(np.array([3, 1]), 5)
        np.testing.assert_array_equal(array_idx, [3, 1])
        assert scalar is False

    def test_arl_variable_array_handles_missing_records_and_scalar_squeeze(self):
        field = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        array = ArlVariableArray(
            records={(0, 0): StubRecord(field)},
            shape=(2, 2, 2, 3),
        )

        stacked = array._getitem((0, slice(None), slice(None), slice(None)))
        assert stacked.shape == (2, 2, 3)
        np.testing.assert_allclose(stacked[0], field)
        assert np.isnan(stacked[1]).all()

        squeezed = array._getitem((0, 0, 1, slice(None)))
        np.testing.assert_allclose(squeezed, [4.0, 5.0, 6.0])

    def test_arl_variable_array_rejects_non_4d_indexing(self):
        array = ArlVariableArray(records={}, shape=(1, 1, 1, 1))

        with pytest.raises(IndexError, match="expect 4 indexers"):
            array._getitem((slice(None),))
