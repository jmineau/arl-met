"""Tests for the cheap header-only probe of an ARL file."""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from arlmet import ARLFormatError, File, Probe, probe
from arlmet._grid import Grid, Projection
from arlmet.record import Header
from arlmet.vertical import PressureAxis

TIMES = pd.date_range("2024-07-18 00:00", periods=3, freq="3h")
NX = NY = 20
RECORD_LENGTH = Header.N_BYTES + NX * NY  # 450 bytes
STEP_LENGTH = 3 * RECORD_LENGTH  # one index record + PRSS + TEMP


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


def write_steps(path, times=TIMES, *, source="TEST", forecast=0) -> None:
    """Write *times* as time steps, each an index record plus PRSS and TEMP."""
    with File(
        path,
        mode="w",
        source=source,
        grid=make_test_grid(),
        vertical_axis=PressureAxis(levels=[0.0, 1000.0]),
    ) as arl:
        for i, time in enumerate(times):
            rs = arl.create_recordset(time, forecast=forecast)
            rs.create_datarecord(
                "PRSS", level=0, forecast=forecast, data=field(i, 1000.0)
            )
            rs.create_datarecord(
                "TEMP", level=1, forecast=forecast, data=field(i, 280.0)
            )


def repeat_first_step(path) -> None:
    """Insert a byte-identical copy of the first time step right after it."""
    raw = path.read_bytes()
    first = raw[:STEP_LENGTH]
    path.write_bytes(first + raw)


class TestProbe:
    def test_reads_the_first_two_index_records(self, tmp_path):
        path = tmp_path / "three_steps.arl"
        write_steps(path)

        result = probe(path)

        assert isinstance(result, Probe)
        assert result.path == path
        assert result.first == TIMES[0]
        assert result.step == pd.Timedelta(hours=3)
        assert result.n_steps == 3
        assert result.grid == make_test_grid()
        assert result.source == "TEST"
        assert result.forecast == 0
        assert result.whole is True

    def test_does_not_read_any_data_record(self, tmp_path):
        path = tmp_path / "three_steps.arl"
        write_steps(path)
        # Zero out both data records (PRSS, TEMP) of every time step, leaving
        # the three index records untouched. A probe that reached into the
        # data records would still not raise (it never unpacks values), but
        # it would have to read past the index record to do so; this proves
        # the answer does not depend on them.
        raw = bytearray(path.read_bytes())
        for step in range(3):
            start = step * STEP_LENGTH + RECORD_LENGTH
            end = step * STEP_LENGTH + STEP_LENGTH
            raw[start:end] = bytes(end - start)
        path.write_bytes(bytes(raw))

        result = probe(path)

        assert result.n_steps == 3
        assert result.whole is True
        assert result.step == pd.Timedelta(hours=3)

    def test_a_file_of_one_time_step_has_no_step(self, tmp_path):
        path = tmp_path / "one_step.arl"
        write_steps(path, TIMES[:1])

        result = probe(path)

        assert result.n_steps == 1
        assert result.step is None
        assert result.whole is True

    def test_a_truncated_file_is_not_whole(self, tmp_path):
        path = tmp_path / "truncated.arl"
        write_steps(path, TIMES[:2])
        # Cut the last data record of the second time step: the size is
        # still a whole number of 450-byte records, but not of 2 full time
        # steps (3 records each).
        path.write_bytes(path.read_bytes()[:-RECORD_LENGTH])

        result = probe(path)

        assert result.n_steps == 1  # only the first time step fits whole
        assert result.whole is False
        assert result.first == TIMES[0]
        assert result.step is None  # fewer than two whole steps to compare

    def test_a_file_cut_mid_record_is_not_whole(self, tmp_path):
        path = tmp_path / "mid_record.arl"
        write_steps(path, TIMES[:2])
        path.write_bytes(path.read_bytes()[:-10])

        result = probe(path)

        assert result.whole is False

    def test_a_repeated_first_step_is_whole_but_reports_a_zero_step(self, tmp_path):
        # Some NOAA HRRR archive files repeat the first time step byte for
        # byte; File() skips the copy, but a probe's cheap math does not know
        # to, so it reads the inserted copy as if it were the second step.
        path = tmp_path / "repeated.arl"
        write_steps(path, TIMES[:2])
        repeat_first_step(path)

        result = probe(path)

        assert result.n_steps == 3
        assert result.whole is True
        assert result.first == TIMES[0]
        assert result.step == pd.Timedelta(0)

    def test_an_empty_file_raises_format_error(self, tmp_path):
        path = tmp_path / "empty.arl"
        path.write_bytes(b"")

        with pytest.raises(ARLFormatError, match="holds no index record"):
            probe(path)

    def test_a_non_arl_file_raises_format_error_naming_the_path(self, tmp_path):
        path = tmp_path / "junk.bin"
        path.write_bytes(b"not an ARL file, just junk bytes" * 10)

        with pytest.raises(ARLFormatError, match="junk.bin"):
            probe(path)

    def test_a_damaged_second_step_raises_format_error(self, tmp_path):
        path = tmp_path / "damaged_second.arl"
        write_steps(path, TIMES[:2])
        raw = bytearray(path.read_bytes())
        # Smash the second index record's header so it no longer reads INDX.
        raw[STEP_LENGTH : STEP_LENGTH + Header.N_BYTES] = b"x" * Header.N_BYTES
        path.write_bytes(bytes(raw))

        with pytest.raises(ARLFormatError, match="damaged_second.arl"):
            probe(path)

    def test_is_a_frozen_dataclass(self, tmp_path):
        path = tmp_path / "three_steps.arl"
        write_steps(path)
        result = probe(path)

        with pytest.raises(FrozenInstanceError):
            result.n_steps = 10  # type: ignore[misc]
