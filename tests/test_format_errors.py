"""Tests for malformed-file detection: ARLFormatError and ARLFormatWarning."""

import warnings

import numpy as np
import pandas as pd
import pytest

from arlmet import ARLFormatError, ARLFormatWarning, File, extract_subset
from arlmet.grid import Grid, Projection
from arlmet.header import Header
from arlmet.vertical import PressureAxis

TIMES = [pd.Timestamp("2024-07-18 00:00"), pd.Timestamp("2024-07-18 03:00")]
NX = NY = 20
RECORD_LENGTH = Header.N_BYTES + NX * NY  # 450 bytes
STEP_LENGTH = 3 * RECORD_LENGTH  # index record + PRSS + TEMP


def make_test_grid(**overrides) -> Grid:
    params = dict(
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
    params.update(overrides)
    return Grid(projection=Projection(**params), nx=NX, ny=NY)


def field(time_index: int, offset: float) -> np.ndarray:
    base = np.arange(NX * NY, dtype=np.float32).reshape(NY, NX)
    return base + offset + 10.0 * time_index


def write_two_step_file(path, grid: Grid | None = None) -> None:
    """Write two time steps, each an index record plus PRSS and TEMP."""
    with File(
        path,
        mode="w",
        source="TEST",
        grid=grid or make_test_grid(),
        vertical_axis=PressureAxis(levels=[0.0, 1000.0]),
    ) as arl:
        for i, time in enumerate(TIMES):
            rs = arl.create_recordset(time)
            rs.create_datarecord("PRSS", level=0, forecast=i, data=field(i, 1000.0))
            rs.create_datarecord("TEMP", level=1, forecast=i, data=field(i, 280.0))


def repeat_first_step(path, *, alter: bool = False) -> None:
    """Insert a copy of the first time step right after it (HRRR-style)."""
    raw = path.read_bytes()
    first = bytearray(raw[:STEP_LENGTH])
    if alter:
        first[-1] = (first[-1] + 1) % 256  # change one packed byte of TEMP
    path.write_bytes(raw[:STEP_LENGTH] + bytes(first) + raw[STEP_LENGTH:])


class TestHeaderAndIndexParsing:
    def test_corrupt_header_field_raises_format_error(self):
        header = bytearray(Header.N_BYTES)
        header[:] = b"24 7180000" + b" " * 40
        header[2:4] = b"xx"  # month

        with pytest.raises(ARLFormatError, match="cannot parse month from 'xx'"):
            Header.from_bytes(bytes(header))

    def test_short_header_raises_format_error(self):
        with pytest.raises(ARLFormatError, match="exactly 50 bytes"):
            Header.from_bytes(b"too short")

    def test_format_error_is_a_value_error(self):
        assert issubclass(ARLFormatError, ValueError)
        with pytest.raises(ValueError):
            Header.from_bytes(b"too short")

    def test_non_arl_file_raises_format_error_naming_path(self, tmp_path):
        path = tmp_path / "junk.bin"
        path.write_bytes(b"CDF\x01 this is a netCDF file, not ARL" * 10)

        with pytest.raises(ARLFormatError, match="junk.bin.*byte 0"):
            File(path)

    def test_corrupt_index_record_raises_format_error(self, tmp_path):
        path = tmp_path / "corrupt_index.arl"
        write_two_step_file(path)
        raw = bytearray(path.read_bytes())
        # nx field of the second index record's fixed portion
        nx_offset = STEP_LENGTH + Header.N_BYTES + 93
        raw[nx_offset : nx_offset + 3] = b" xx"
        path.write_bytes(bytes(raw))

        with pytest.raises(
            ARLFormatError, match=f"corrupt_index.arl.*byte {STEP_LENGTH}"
        ):
            File(path)

    def test_data_record_where_index_expected_raises_format_error(self, tmp_path):
        path = tmp_path / "no_index.arl"
        write_two_step_file(path)
        # Drop the first index record, so the file starts with a data record.
        path.write_bytes(path.read_bytes()[RECORD_LENGTH:])

        with pytest.raises(ARLFormatError, match="Expected 'INDX' record"):
            File(path)


class TestTruncatedFiles:
    def test_partial_record_raises_format_error(self, tmp_path):
        path = tmp_path / "partial.arl"
        write_two_step_file(path)
        path.write_bytes(path.read_bytes()[:-10])

        with pytest.raises(ARLFormatError, match="not a whole number of 450-byte"):
            File(path)

    def test_missing_data_records_raise_format_error(self, tmp_path):
        path = tmp_path / "missing_records.arl"
        write_two_step_file(path)
        # Cut the last whole data record: the size is still a multiple of the
        # record length, but the second index record declares 2 data records.
        path.write_bytes(path.read_bytes()[:-RECORD_LENGTH])

        with pytest.raises(
            ARLFormatError, match="declares 2 data records, but only 1 remain"
        ):
            File(path)

    def test_complete_file_still_opens(self, tmp_path):
        path = tmp_path / "complete.arl"
        write_two_step_file(path)

        with File(path) as arl:
            assert arl.times == TIMES


class TestDuplicateTimeSteps:
    def test_identical_duplicate_warns_and_is_ignored(self, tmp_path):
        path = tmp_path / "duplicate.arl"
        write_two_step_file(path)
        repeat_first_step(path)

        with pytest.warns(ARLFormatWarning) as record:
            arl = File(path)
        with arl:
            assert len(record) == 1
            message = str(record[0].message)
            assert "duplicate.arl" in message
            assert str(TIMES[0]) in message
            assert "ignored" in message

            assert arl.times == TIMES
            assert len(arl) == 2
            np.testing.assert_allclose(
                arl[TIMES[0]][(1, "TEMP")].read(), field(0, 280.0), atol=0.01
            )
            # The second time step is read from after the repeated copy.
            assert arl[TIMES[1]].position == 2 * STEP_LENGTH
            np.testing.assert_allclose(
                arl[TIMES[1]][(0, "PRSS")].read(), field(1, 1000.0), atol=0.01
            )

    def test_extract_subset_of_duplicate_file_is_clean(self, tmp_path):
        source = tmp_path / "duplicate.arl"
        destination = tmp_path / "clean.arl"
        write_two_step_file(source)
        repeat_first_step(source)

        with pytest.warns(ARLFormatWarning):
            extract_subset(source, destination)

        assert destination.stat().st_size == 2 * STEP_LENGTH
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with File(destination) as clean:
                assert clean.times == TIMES
                np.testing.assert_allclose(
                    clean[TIMES[0]][(1, "TEMP")].read(), field(0, 280.0), atol=0.01
                )

    def test_differing_duplicate_raises_format_error(self, tmp_path):
        path = tmp_path / "conflict.arl"
        write_two_step_file(path)
        repeat_first_step(path, alter=True)

        with pytest.raises(ARLFormatError, match="repeated with different content"):
            File(path)

    def test_write_mode_still_rejects_existing_time(self, tmp_path):
        with File(
            tmp_path / "write.arl",
            mode="w",
            source="TEST",
            grid=make_test_grid(),
            vertical_axis=PressureAxis(levels=[0.0]),
        ) as arl:
            rs = arl.create_recordset(TIMES[0])
            rs.create_datarecord("PRSS", level=0, forecast=0, data=field(0, 1000.0))
            with pytest.raises(ValueError, match="already exists"):
                arl.create_recordset(TIMES[0])


def write_steps(path, steps) -> None:
    """Write one time step per ``(time, variables)``, with PRSS at level 0 and TEMP at level 1."""
    levels = {"PRSS": 0, "TEMP": 1}
    with File(
        path,
        mode="w",
        source="TEST",
        grid=make_test_grid(),
        vertical_axis=PressureAxis(levels=[0.0, 1000.0]),
    ) as arl:
        for i, (time, variables) in enumerate(steps):
            rs = arl.create_recordset(pd.Timestamp(time))
            for name in variables:
                rs.create_datarecord(
                    name, level=levels[name], forecast=i, data=field(i, 1000.0)
                )


#: Byte position of the second time step's TEMP record in a two-step file.
SECOND_TEMP = STEP_LENGTH + 2 * RECORD_LENGTH


class TestCheck:
    def test_whole_file_has_no_problems(self, tmp_path):
        path = tmp_path / "whole.arl"
        write_two_step_file(path)

        with File(path) as arl:
            assert arl.check() == []

    def test_time_step_written_partway(self, tmp_path):
        path = tmp_path / "partway.arl"
        write_steps(
            path,
            [
                ("2024-07-18 00:00", ["PRSS", "TEMP"]),
                ("2024-07-18 03:00", ["PRSS"]),
                ("2024-07-18 06:00", ["PRSS", "TEMP"]),
            ],
        )

        with File(path) as arl:
            assert arl.times[1] == pd.Timestamp("2024-07-18 03:00")
            assert arl.check() == [
                "2024-07-18 03:00 has 1 of the 2 data records the first time step has."
            ]

    def test_missing_time_step(self, tmp_path):
        path = tmp_path / "gap.arl"
        both = ["PRSS", "TEMP"]
        write_steps(
            path,
            [
                ("2024-07-18 00:00", both),
                ("2024-07-18 03:00", both),
                ("2024-07-18 09:00", both),
            ],
        )

        with File(path) as arl:
            assert arl.check() == [
                "2024-07-18 03:00 to 2024-07-18 09:00 is 6 h; the first two "
                "time steps are 3 h apart."
            ]

    def test_quarter_hour_time_steps_are_whole(self, tmp_path):
        path = tmp_path / "quarter.arl"
        both = ["PRSS", "TEMP"]
        write_steps(
            path,
            [
                ("2024-07-18 00:00", both),
                ("2024-07-18 00:15", both),
                ("2024-07-18 00:30", both),
            ],
        )

        with File(path) as arl:
            assert arl.check() == []

    def test_record_header_of_null_bytes(self, tmp_path):
        """A record whose bytes were lost reads as nulls, but the file opens."""
        path = tmp_path / "nulls.arl"
        write_two_step_file(path)
        raw = bytearray(path.read_bytes())
        raw[SECOND_TEMP : SECOND_TEMP + 2] = b"\x00\x00"
        path.write_bytes(bytes(raw))

        with File(path) as arl:
            problems = arl.check()

        assert len(problems) == 1
        assert problems[0].startswith(
            f"2024-07-18 03:00: the TEMP record at level 1 (byte {SECOND_TEMP}) "
            "cannot be read."
        )

    def test_record_from_another_time_step(self, tmp_path):
        path = tmp_path / "copied.arl"
        write_two_step_file(path)
        raw = bytearray(path.read_bytes())
        first_temp = 2 * RECORD_LENGTH
        raw[SECOND_TEMP : SECOND_TEMP + RECORD_LENGTH] = raw[
            first_temp : first_temp + RECORD_LENGTH
        ]
        path.write_bytes(bytes(raw))

        with File(path) as arl:
            assert arl.check() == [
                f"2024-07-18 03:00: the TEMP record at level 1 (byte {SECOND_TEMP}) "
                "is for 2024-07-18 00:00."
            ]

    def test_write_mode_raises(self, tmp_path):
        with (
            File(
                tmp_path / "new.arl",
                mode="w",
                source="TEST",
                grid=make_test_grid(),
                vertical_axis=PressureAxis(levels=[0.0, 1000.0]),
            ) as arl,
            pytest.raises(ValueError),
        ):
            arl.check()


class TestIndexLongitudeWrap:
    def _projected_grid(self, **overrides) -> Grid:
        params = dict(
            pole_lat=90.0,
            pole_lon=0.0,
            tangent_lat=38.5,
            tangent_lon=-97.5,
            grid_size=3.0,
            orientation=0.0,
            cone_angle=38.5,
            sync_x=900.5,  # grid index of a center sync point, > 180
            sync_y=530.0,
            sync_lat=38.5,
            sync_lon=-97.5,
        )
        params.update(overrides)
        return make_test_grid(**params)

    def test_sync_xy_above_180_roundtrip_unchanged(self, tmp_path):
        path = tmp_path / "projected.arl"
        grid = self._projected_grid()
        write_two_step_file(path, grid=grid)

        with File(path) as arl:
            projection = arl.grid.projection
            assert projection.sync_x == 900.5
            assert projection.sync_y == 530.0
            assert arl.grid == grid

    def test_longitudes_in_0_360_are_wrapped(self, tmp_path):
        path = tmp_path / "lon360.arl"
        write_two_step_file(
            path, grid=self._projected_grid(tangent_lon=262.5, sync_lon=237.28)
        )

        with File(path) as arl:
            projection = arl.grid.projection
            assert projection.tangent_lon == pytest.approx(-97.5)
            assert projection.sync_lon == pytest.approx(-122.72)
            assert projection.sync_x == 900.5
