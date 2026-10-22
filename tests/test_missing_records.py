"""Records flagged missing (negative forecast hour or the name NULL) read as NaN (#84)."""

import numpy as np
import pandas as pd
import pytest

import arlmet
from arlmet import File, extract_subset, open_dataset, sample_points
from arlmet.grid import Grid, Projection
from arlmet.header import Header
from arlmet.vertical import PressureAxis

NX = NY = 20
RECORD_LENGTH = Header.N_BYTES + NX * NY
TIMES = [pd.Timestamp("2024-07-18 00:00"), pd.Timestamp("2024-07-18 01:00")]
#: Index record, PRSS, TEMP per time step.
STEP_LENGTH = 3 * RECORD_LENGTH
#: Byte position of the second time step's TEMP record.
SECOND_TEMP = STEP_LENGTH + 2 * RECORD_LENGTH


def make_grid() -> Grid:
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
    return Grid(projection=projection, nx=NX, ny=NY)


def field(value: float) -> np.ndarray:
    base = np.arange(NX * NY, dtype=np.float32).reshape(NY, NX) / 100.0
    return base + value


def write_file(path):
    """Two time steps of PRSS (level 0) and TEMP (level 1), all data."""
    with File(
        path,
        mode="w",
        source="TEST",
        grid=make_grid(),
        vertical_axis=PressureAxis(levels=[0.0, 1000.0]),
    ) as arl:
        for time in TIMES:
            rs = arl.create_recordset(time, forecast=0)
            rs.create_datarecord("PRSS", level=0, forecast=0, data=field(1000.0))
            rs.create_datarecord("TEMP", level=1, forecast=0, data=field(290.0))
    return path


def make_missing(path, position, *, name=None):
    """
    Rewrite the record at ``position`` as HYSPLIT's pakini.f leaves a record.

    Forecast hour -1, NEXP 0, PREC and VAR1 0, and the packed values spaces
    with a carriage return last. With ``name``, the header names it so
    (``NULL``) instead of the forecast hour marking it.
    """
    raw = bytearray(path.read_bytes())
    label = raw[position : position + Header.N_BYTES].decode()
    forecast = label[8:10] if name else "-1"
    variable = name or label[14:18]
    label = (
        label[:8] + forecast + label[10:14] + variable + f"{0:4d}{0.0:14.7E}{0.0:14.7E}"
    )
    assert len(label) == Header.N_BYTES
    payload = b" " * (NX * NY - 1) + b"\r"
    raw[position : position + RECORD_LENGTH] = label.encode() + payload
    path.write_bytes(bytes(raw))
    return path


@pytest.fixture
def missing_file(tmp_path):
    """The 01:00 TEMP record is flagged missing (forecast hour -1)."""
    return make_missing(write_file(tmp_path / "missing.arl"), SECOND_TEMP)


class TestRead:
    def test_missing_record_reads_as_nan(self, missing_file):
        with File(missing_file) as arl:
            record = arl[TIMES[1]][(1, "TEMP")]
            assert record.forecast == -1
            values = record.read()
            assert values.shape == (NY, NX) and values.dtype == np.float32
            assert np.isnan(values).all()
            assert np.isnan(record.data).all()
            window = arl.grid.window_from_bbox((22.0, -8.0, 25.0, -5.0))
            assert np.isnan(record.read(window=window)).all()
            assert record.read(window=window).shape == window.shape
            # The other records are data.
            np.testing.assert_allclose(
                arl[TIMES[0]][(1, "TEMP")].read(), field(290.0), atol=0.01
            )
            np.testing.assert_allclose(
                arl[TIMES[1]][(0, "PRSS")].read(), field(1000.0), atol=0.01
            )

    def test_record_named_null_reads_as_nan(self, tmp_path):
        # The index lists TEMP; the record's own header says NULL.
        path = make_missing(write_file(tmp_path / "null.arl"), SECOND_TEMP, name="NULL")
        with File(path) as arl:
            record = arl[TIMES[1]][(1, "TEMP")]
            assert record.header.variable == "NULL"
            assert np.isnan(record.read()).all()

    def test_open_dataset_reads_missing_as_nan(self, missing_file):
        with open_dataset(missing_file) as ds:
            temp = ds["TEMP"].load()
        assert not np.isnan(temp.isel(time=0)).any()
        assert np.isnan(temp.isel(time=1)).all()

    def test_sample_points_reads_missing_as_nan(self, missing_file):
        points = pd.DataFrame(
            {"lon": [25.0, 25.0], "lat": [-5.0, -5.0], "time": TIMES, "z": [0, 0]}
        )
        got = sample_points(missing_file, points, ["TEMP", "PRSS"], z_kind="native")
        assert not np.isnan(got["TEMP"].iloc[0])
        assert np.isnan(got["TEMP"].iloc[1])
        assert not np.isnan(got["PRSS"]).any()

    def test_missing_dif_record_adds_nothing(self, tmp_path):
        # HYSPLIT's metinp skips a missing DIFW, so W is WWND alone.
        path = tmp_path / "dif.arl"
        target = field(0.123456)
        with File(
            path,
            mode="w",
            source="TEST",
            grid=make_grid(),
            vertical_axis=PressureAxis(levels=[0.0, 1000.0]),
        ) as arl:
            rs = arl.create_recordset(TIMES[0], forecast=0)
            rs.create_datarecord("PRSS", level=0, forecast=0, data=field(1000.0))
            rs.create_datarecord("WWND", level=1, forecast=0, data=target, diff="DIFW")
        with File(path) as arl:
            record = arl[TIMES[0]][(1, "WWND")]
            (dif,) = record.diffs
            alone = record._read_own(None)  # WWND without its DIFW
            assert not np.array_equal(record.read(), alone)
        make_missing(path, dif.position)
        with File(path) as arl:
            np.testing.assert_array_equal(arl[TIMES[0]][(1, "WWND")].read(), alone)


class TestCheck:
    def test_check_reports_a_missing_record(self, missing_file):
        with File(missing_file) as arl:
            assert arl.check() == [
                f"2024-07-18 01:00: the TEMP record at level 1 (byte {SECOND_TEMP}) "
                "is flagged missing (forecast hour -1). HYSPLIT skips it, and "
                "arlmet reads it as NaN."
            ]

    def test_check_reports_a_record_named_null(self, tmp_path):
        path = make_missing(write_file(tmp_path / "null.arl"), SECOND_TEMP, name="NULL")
        with File(path) as arl:
            (problem,) = arl.check()
        assert "is flagged missing (named NULL)" in problem

    def test_check_reports_more_missing_than_hysplit_allows(self, tmp_path):
        # metinp.f stops when a time step has more than 20 missing records.
        path = tmp_path / "many.arl"
        names = [f"V{k:03d}" for k in range(21)]
        nan = np.full((NY, NX), np.nan, dtype=np.float32)
        with File(
            path,
            mode="w",
            source="TEST",
            grid=make_grid(),
            vertical_axis=PressureAxis(levels=[0.0]),
        ) as arl:
            rs = arl.create_recordset(TIMES[0], forecast=0)
            for name in names:
                rs.create_datarecord(name, level=0, forecast=0, data=nan)
        with File(path) as arl:
            problems = arl.check()
        assert len(problems) == 22
        assert problems[-1] == (
            "2024-07-18 00:00 has 21 records flagged missing; HYSPLIT stops at "
            "more than 20 in one time step."
        )


class TestWrite:
    def test_a_field_of_nan_is_written_as_pakini_writes_a_missing_record(
        self, tmp_path
    ):
        written = tmp_path / "written.arl"
        with File(
            written,
            mode="w",
            source="TEST",
            grid=make_grid(),
            vertical_axis=PressureAxis(levels=[0.0, 1000.0]),
        ) as arl:
            for time, temp in zip(TIMES, [field(290.0), None], strict=True):
                rs = arl.create_recordset(time, forecast=0)
                rs.create_datarecord("PRSS", level=0, forecast=0, data=field(1000.0))
                data = temp if temp is not None else np.full((NY, NX), np.nan)
                rs.create_datarecord("TEMP", level=1, forecast=0, data=data)
        expected = make_missing(write_file(tmp_path / "expected.arl"), SECOND_TEMP)

        raw = written.read_bytes()
        assert (
            raw[SECOND_TEMP : SECOND_TEMP + RECORD_LENGTH]
            == expected.read_bytes()[SECOND_TEMP : SECOND_TEMP + RECORD_LENGTH]
        )
        with File(written) as arl:
            record = arl[TIMES[1]][(1, "TEMP")]
            assert record.forecast == -1
            assert record.checksum == 0  # none recorded, as HYSPLIT's index has
            assert np.isnan(record.read()).all()

    def test_crop_keeps_a_missing_record_missing(self, missing_file, tmp_path):
        dest = tmp_path / "crop.arl"
        extract_subset(missing_file, dest, bbox=(20.0, -10.0, 33.0, 3.0))
        with File(dest) as arl:
            record = arl[TIMES[1]][(1, "TEMP")]
            assert record.forecast == -1
            assert np.isnan(record.read()).all()
            assert not np.isnan(arl[TIMES[0]][(1, "TEMP")].read()).any()

    def test_crop_across_the_seam_keeps_a_missing_record_missing(self, tmp_path):
        # A crop across 0 E reads its columns in two pieces (#83); a missing
        # record still reads as NaN of the window's shape.
        grid = Grid(
            projection=Projection(
                pole_lat=90.0,
                pole_lon=359.0,
                tangent_lat=1.0,
                tangent_lon=1.0,
                grid_size=0.0,
                orientation=0.0,
                cone_angle=0.0,
                sync_x=1.0,
                sync_y=1.0,
                sync_lat=-90.0,
                sync_lon=0.0,
            ),
            nx=360,
            ny=181,
        )
        wave = (200.0 + 100.0 * np.sin(np.radians(np.arange(360.0)))).astype(np.float32)
        source = tmp_path / "global.arl"
        with File(
            source,
            mode="w",
            source="GDAS",
            grid=grid,
            vertical_axis=PressureAxis(levels=[0.0, 1000.0]),
        ) as arl:
            rs = arl.create_recordset(TIMES[0], forecast=0)
            rs.create_datarecord(
                "SHGT", level=0, forecast=0, data=np.tile(wave, (181, 1))
            )
            rs.create_datarecord(
                "TEMP", level=1, forecast=0, data=np.full((181, 360), np.nan)
            )

        bbox = (-10.0, 40.0, 10.0, 60.0)
        with File(source) as arl:
            window = arl.grid.window_from_bbox(bbox)
            assert window.x_stop > 360
            temp = arl[TIMES[0]][(1, "TEMP")].read(window=window)
            assert temp.shape == window.shape and np.isnan(temp).all()

        dest = extract_subset(source, tmp_path / "crop.arl", bbox=bbox)
        with File(dest) as arl:
            assert arl.grid.nx == 21
            assert np.isnan(arl[TIMES[0]][(1, "TEMP")].read()).all()
            shgt = arl[TIMES[0]][(0, "SHGT")].read()
        want = 200.0 + 100.0 * np.sin(np.radians(np.arange(-10.0, 11.0)))
        np.testing.assert_allclose(shgt[0], want, atol=0.05)

    def test_dataset_round_trip_leaves_a_missing_record_out(
        self, missing_file, tmp_path
    ):
        # A missing record reads as an all-NaN slice, which write_dataset takes
        # to be no record, as for any all-NaN slice; check() then says so.
        dest = tmp_path / "round.arl"
        with open_dataset(missing_file) as ds:
            arlmet.write_dataset(ds, dest)
        with File(dest) as arl:
            assert (1, "TEMP") not in arl[TIMES[1]]
            assert arl.check() == [
                "2024-07-18 01:00 has 1 of the 2 data records the first time step has."
            ]
