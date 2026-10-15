"""
Tests for streams that carry many DIF records.

Several parents have a DIF each, in one time step, on several levels, in
several time steps. NOAA's HRRR files are like this (``TPP1`` + ``DIFR`` at the
surface, ``WWND`` + ``DIFW`` on every level). A parent can also have several DIF
records in a row, which HYSPLIT adds one after another, and a DIF record need not
follow its parent: HYSPLIT pairs it by name (NOAA's ``era52arl`` lists it last on
its level). The tests read, rewrite
and subset such streams and check the results against values worked out here,
not against arlmet's output:

- ``reference_unpack`` follows HYSPLIT's ``pakinp`` one cell at a time;
- ``walk`` cuts a file into its fixed-size records without ``File``;
- ``raw_record`` and ``raw_time_step`` build a stream byte by byte, as a
  producer other than arlmet would, including streams arlmet's writer cannot
  make (a DIF out of place, a variable listed twice).
"""

from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from arlmet import (
    ARLFormatError,
    ARLFormatWarning,
    File,
    extract_subset,
    open_dataset,
    write_dataset,
)
from arlmet.grid import Grid, GridWindow, Projection
from arlmet.header import Header
from arlmet.indexrecord import IndexRecord, LvlInfo, VarInfo
from arlmet.packing import calculate_checksum
from arlmet.vertical import PressureAxis

NX = NY = 20
RECORD_BYTES = Header.N_BYTES + NX * NY
AXIS = PressureAxis(levels=[0.0, 1000.0, 900.0, 800.0])
TIME0 = pd.Timestamp("2024-07-18 00:00")
# The DIF name that goes with each parent in HYSPLIT's producers and NOAA's files.
DIF_OF = {"TPP1": "DIFR", "WWND": "DIFW", "TKEN": "DIFT"}
PARENT_OF = {dif: parent for parent, dif in DIF_OF.items()}
# Grid cells from lon 21 to 35 and lat -9 to 5: 15 x 15 of the 20 x 20 grid,
# enough for the index record of a one-level file.
CROP = (21.0, -9.0, 35.0, 5.0)
# 18 x 18 cells: enough for the index record of the multi-level file below.
WIDE_CROP = (21.0, -9.0, 38.0, 8.0)

Step = list[tuple[Header, bytes]]  # a time step's records, index record left out


def make_test_grid() -> Grid:
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


GRID = make_test_grid()


# --- Values worked out here ---------------------------------------------------


def reference_unpack(record: bytes, *, nx: int = NX, ny: int = NY) -> np.ndarray:
    """
    Unpack a whole record (header and payload) the way HYSPLIT's ``pakinp`` does.

    Each cell is its left neighbor plus ``(byte - 127) * 2**(exponent - 7)``;
    column 0 continues from the cell above, row 0 from the header's initial
    value. A value smaller than the precision becomes 0, in column 0 too, as
    ``arlmet.unpack`` has it; ``pakinp`` leaves column 0 alone, which changes
    a cell by less than the precision.
    """
    header = Header.from_bytes(record[: Header.N_BYTES])
    payload = record[Header.N_BYTES :]
    assert len(payload) == nx * ny
    scale = 1.0 / 2.0 ** (7 - header.exponent)
    field = np.zeros((ny, nx))
    above = header.initial_value
    for j in range(ny):
        above += (payload[j * nx] - 127) * scale
        value = above
        field[j, 0] = value
        for i in range(1, nx):
            value += (payload[j * nx + i] - 127) * scale
            field[j, i] = value
    field[np.abs(field) < header.precision] = 0.0
    return field


def combined(records: list[bytes]) -> np.ndarray:
    """The field a parent record and the DIF records after it add up to."""
    return sum(reference_unpack(record) for record in records)


def walk(path: Path, *, nx: int = NX, ny: int = NY) -> dict[pd.Timestamp, Step]:
    """
    Cut an ARL file into its time steps without ``File``.

    A time step, keyed by the time of its index record, is the ``(header,
    whole record)`` of each data record after it. ``nx`` and ``ny`` are the
    file's grid size, which sets the record length.
    """
    raw = Path(path).read_bytes()
    length = Header.N_BYTES + nx * ny
    assert len(raw) % length == 0
    steps: dict[pd.Timestamp, Step] = {}
    for start in range(0, len(raw), length):
        record = raw[start : start + length]
        header = Header.from_bytes(record[: Header.N_BYTES])
        if header.variable == "INDX":
            records = steps.setdefault(header.time, [])
        else:
            records.append((header, record))
    return steps


def pairs_in(
    path: Path, *, nx: int = NX, ny: int = NY
) -> list[tuple[pd.Timestamp, Header, bytes, Header, bytes]]:
    """Each parent and its DIF in a file: ``(time, header, record, DIF header, DIF record)``."""
    found = []
    for time, step in walk(path, nx=nx, ny=ny).items():
        for (header, record), (next_header, next_record) in pairwise(step):
            if next_header.variable.startswith("DIF"):
                found.append((time, header, record, next_header, next_record))
    return found


# --- Streams built by hand ----------------------------------------------------


def raw_record(
    time: pd.Timestamp,
    level: int,
    name: str,
    payload: np.ndarray,
    *,
    exponent: int = 7,
    initial_value: float = 0.0,
) -> bytes:
    """One data record: a header and ``payload``, one unsigned byte per cell."""
    header = Header(
        year=time.year,
        month=time.month,
        day=time.day,
        hour=time.hour,
        forecast=0,
        level=level,
        grid=(0, 0),
        variable=name,
        exponent=exponent,
        precision=0.0,  # nothing is rounded to 0, so every value stays exact
        initial_value=initial_value,
    )
    return header.tobytes() + np.asarray(payload, dtype=np.uint8).tobytes()


def raw_time_step(
    time: pd.Timestamp,
    levels: list[list[bytes]],
    *,
    stored: dict[tuple[int, str], tuple[int, str]] | None = None,
) -> bytes:
    """
    An index record and the data records after it, ``levels`` holding each level's in order.

    The index lists each record's checksum, unless ``stored`` gives the
    ``(checksum, flag)`` to write for a ``(level, name)``.
    """
    stored = stored or {}
    infos = []
    for level, (height, records) in enumerate(zip(AXIS.levels, levels, strict=True)):
        variables: list[tuple[str, VarInfo]] = []
        for record in records:
            name = Header.from_bytes(record[: Header.N_BYTES]).variable
            checksum = calculate_checksum(record[Header.N_BYTES :])
            info = VarInfo(*stored.get((level, name), (checksum, "")))
            variables.append((name, info))
        infos.append(LvlInfo(level=level, height=float(height), variables=variables))
    p = GRID.projection
    index = IndexRecord(
        header=Header(
            year=time.year,
            month=time.month,
            day=time.day,
            hour=time.hour,
            forecast=0,
            level=0,
            grid=(0, 0),
            variable="INDX",
            exponent=0,
            precision=0.0,
            initial_value=0.0,
        ),
        source="TEST",
        forecast=0,
        minutes=time.minute,
        pole_lat=p.pole_lat,
        pole_lon=p.pole_lon,
        tangent_lat=p.tangent_lat,
        tangent_lon=p.tangent_lon,
        grid_size=p.grid_size,
        orientation=p.orientation,
        cone_angle=p.cone_angle,
        sync_x=p.sync_x,
        sync_y=p.sync_y,
        sync_lat=p.sync_lat,
        sync_lon=p.sync_lon,
        reserved=AXIS.offset,
        nx=NX,
        ny=NY,
        nz=len(infos),
        vertical_flag=AXIS.flag,
        levels=infos,
    )
    return index.to_record_bytes(RECORD_BYTES) + b"".join(
        record for records in levels for record in records
    )


def noise(rng: np.random.Generator) -> np.ndarray:
    """Payload bytes of small steps, so a field wanders a little, never far."""
    return rng.integers(120, 135, size=(NY, NX))


def parent_and_dif(
    rng: np.random.Generator,
    time: pd.Timestamp,
    level: int,
    parent: str,
    *,
    start: float,
) -> list[bytes]:
    """
    A parent record and its DIF record, with exact values.

    The parent starts at ``start`` and steps by 0.25, the DIF starts at 0.5 and
    steps by 1/128. Both are sums of powers of two that float32 holds exactly,
    so the tests compare for equality.
    """
    return [
        raw_record(time, level, parent, noise(rng), exponent=5, initial_value=start),
        raw_record(
            time, level, DIF_OF[parent], noise(rng), exponent=0, initial_value=0.5
        ),
    ]


def hrrr_like_step(rng: np.random.Generator, time: pd.Timestamp) -> list[list[bytes]]:
    """
    The records of one time step, laid out like NOAA's HRRR files.

    Level 0 has ``TPP1`` with ``DIFR``; level 1 has ``UWND``, ``WWND`` with
    ``DIFW``, then ``TEMP`` (a record after a DIF record); levels 2 and 3 each
    end with a ``WWND`` and its ``DIFW``.
    """
    level1 = [
        raw_record(time, 1, "UWND", noise(rng), exponent=5, initial_value=3.0),
        *parent_and_dif(rng, time, 1, "WWND", start=10.0),
        raw_record(time, 1, "TEMP", noise(rng), exponent=5, initial_value=280.0),
    ]
    return [
        parent_and_dif(rng, time, 0, "TPP1", start=2.0),
        level1,
        parent_and_dif(rng, time, 2, "WWND", start=20.0),
        parent_and_dif(rng, time, 3, "WWND", start=30.0),
    ]


def write_hrrr_like_file(path: Path, n_steps: int = 2) -> list[pd.Timestamp]:
    """Write ``n_steps`` hourly time steps from ``hrrr_like_step``."""
    rng = np.random.default_rng(5)
    times = list(pd.date_range(TIME0, periods=n_steps, freq="h"))
    path.write_bytes(
        b"".join(raw_time_step(time, hrrr_like_step(rng, time)) for time in times)
    )
    return times


# --- A file written by arlmet -------------------------------------------------

N_PAIRS = 3 * 7  # in each of 3 time steps: TPP1, and WWND and TKEN on 3 levels


def write_multi_dif_file(path: Path) -> dict[tuple[pd.Timestamp, int, str], np.ndarray]:
    """
    Write 3 time steps with seven parent and DIF pairs each.

    ``TPP1`` + ``DIFR`` at the surface (with a forecast hour 3 ahead of the index
    record's, as accumulated fields have in NOAA's files), then on levels 1 to 3
    ``UWND``, ``WWND`` + ``DIFW`` and ``TKEN`` + ``DIFT``. Returns the field
    given for each parent that has a DIF.
    """
    rng = np.random.default_rng(11)

    def field(mean: float, std: float, *, positive: bool = False) -> np.ndarray:
        values = mean + std * rng.standard_normal((NY, NX))
        return (np.abs(values) if positive else values).astype(np.float32)

    targets: dict[tuple[pd.Timestamp, int, str], np.ndarray] = {}
    with File(path, mode="w", source="TEST", grid=GRID, vertical_axis=AXIS) as arl:
        for step in range(3):
            time = TIME0 + pd.Timedelta(hours=3 * step)
            hour = 3 * step
            rs = arl.create_recordset(time, forecast=hour)
            parents = [(0, "TPP1", field(0.001, 0.0004, positive=True), hour + 3)]
            rs.create_datarecord("T02M", 0, forecast=hour, data=field(290, 3))
            for level in (1, 2, 3):
                rs.create_datarecord("UWND", level, forecast=hour, data=field(5, 10))
                parents.append((level, "WWND", field(0, 0.3), hour))
                parents.append((level, "TKEN", field(0.5, 0.2, positive=True), hour))
            for level, name, data, forecast in parents:
                rs.create_datarecord(
                    name, level, forecast=forecast, data=data, diff=DIF_OF[name]
                )
                targets[time, level, name] = data
    return targets


# --- Reading ------------------------------------------------------------------


class TestRead:
    def test_each_dif_adds_to_the_parent_before_it(self, tmp_path):
        path = tmp_path / "hrrr_like.arl"
        times = write_hrrr_like_file(path)

        with File(path) as arl:
            assert arl.times == times
            assert arl.check() == []
            for time, step in walk(path).items():
                rs = arl[time]
                # A DIF is part of its parent, not a record of the time step.
                assert [(r.level, r.variable) for r in rs] == [
                    (0, "TPP1"),
                    (1, "UWND"),
                    (1, "WWND"),
                    (1, "TEMP"),
                    (2, "WWND"),
                    (3, "WWND"),
                ]
                assert "DIFW" not in rs

                records = {(h.level, h.variable): raw for h, raw in step}
                for record in rs:
                    level, name = record.level, record.variable
                    if name not in DIF_OF:
                        assert not record.diffs
                        np.testing.assert_array_equal(
                            record.read(), reference_unpack(records[level, name])
                        )
                        continue
                    dif = DIF_OF[name]
                    assert record.diffs
                    assert (record.diffs[0].variable, record.diffs[0].level) == (
                        dif,
                        level,
                    )
                    np.testing.assert_array_equal(
                        record.read(),
                        combined([records[level, name], records[level, dif]]),
                    )
                    np.testing.assert_array_equal(
                        record.diffs[0].read(), reference_unpack(records[level, dif])
                    )

    def test_windowed_read_adds_the_dif_to_the_window(self, tmp_path):
        path = tmp_path / "hrrr_like.arl"
        write_hrrr_like_file(path, n_steps=1)
        records = {(h.level, h.variable): raw for h, raw in walk(path)[TIME0]}
        window = GridWindow(x_start=3, x_stop=12, y_start=2, y_stop=9)

        with File(path) as arl:
            for level in (1, 2, 3):
                expected = combined([records[level, "WWND"], records[level, "DIFW"]])
                np.testing.assert_array_equal(
                    arl[TIME0][(level, "WWND")].read(window=window),
                    expected[window.y_slice, window.x_slice],
                )

    def test_dif_goes_with_the_record_its_name_pairs_with(self, tmp_path):
        # HYSPLIT's reader adds DIFW to the vertical velocity by name, so a
        # record between WWND and its DIFW does not take the correction.
        path = tmp_path / "out_of_place.arl"
        rng = np.random.default_rng(1)
        wwnd = raw_record(TIME0, 1, "WWND", noise(rng), exponent=5, initial_value=10.0)
        temp = raw_record(TIME0, 1, "TEMP", noise(rng), exponent=5, initial_value=280.0)
        difw = raw_record(TIME0, 1, "DIFW", noise(rng), exponent=0, initial_value=0.5)
        path.write_bytes(raw_time_step(TIME0, [[], [wwnd, temp, difw], [], []]))

        with File(path) as arl:
            assert arl[TIME0][(1, "WWND")].diffs[0].variable == "DIFW"
            assert not arl[TIME0][(1, "TEMP")].diffs
            np.testing.assert_array_equal(
                arl[TIME0][(1, "WWND")].read(), combined([wwnd, difw])
            )
            np.testing.assert_array_equal(
                arl[TIME0][(1, "TEMP")].read(), reference_unpack(temp)
            )

    def test_a_variable_listed_twice_on_a_level_is_malformed(self, tmp_path):
        # Only a DIF name may repeat on a level: a second TEMP would hide the first.
        path = tmp_path / "twice.arl"
        rng = np.random.default_rng(2)
        temp = raw_record(TIME0, 1, "TEMP", noise(rng), exponent=5)
        again = raw_record(TIME0, 1, "TEMP", noise(rng), exponent=5)
        path.write_bytes(raw_time_step(TIME0, [[], [temp, again], [], []]))

        with pytest.raises(ARLFormatError) as caught:
            File(path)
        message = str(caught.value)
        assert path.name in message
        assert "lists 'TEMP' twice at level 1 (again at byte 900)" in message

    @pytest.mark.parametrize(
        "where", ["first on a level", "first in a time step"], ids=str
    )
    def test_dif_without_a_parent_on_its_level_is_applied_to_nothing(
        self, tmp_path, where
    ):
        path = tmp_path / "orphan.arl"
        rng = np.random.default_rng(3)
        tpp1, difr = parent_and_dif(rng, TIME0, 0, "TPP1", start=2.0)
        uwnd = raw_record(TIME0, 1, "UWND", noise(rng), exponent=5)
        if where == "first on a level":
            # Level 0 ends with a parent, TPP1; the DIFR that opens level 1 follows
            # a record, but not one on its own level (and above the surface the
            # name table pairs DIFR with nothing).
            orphan = raw_record(TIME0, 1, "DIFR", noise(rng), exponent=0)
            stream = raw_time_step(TIME0, [[tpp1], [orphan, uwnd], [], []])
            position = 2 * RECORD_BYTES
        else:
            # The first time step ends with a parent, UWND; the next one opens with
            # a DIF.
            later = TIME0 + pd.Timedelta(hours=1)
            orphan = raw_record(later, 0, "DIFR", noise(rng), exponent=0)
            stream = raw_time_step(TIME0, [[tpp1, difr], [uwnd], [], []])
            stream += raw_time_step(later, [[orphan], [], [], []])
            position = 5 * RECORD_BYTES
        path.write_bytes(stream)

        # The file opens; the DIF record corrects nothing and one warning
        # says where it is.
        with (
            pytest.warns(ARLFormatWarning, match="1 DIF record") as caught,
            File(path) as arl,
        ):
            with_difs = [
                (time, r.variable) for time in arl.times for r in arl[time] if r.diffs
            ]
            np.testing.assert_array_equal(
                arl[TIME0][(1, "UWND")].read(), reference_unpack(uwnd)
            )
        if where == "first on a level":
            assert with_difs == []
        else:
            assert with_difs == [(TIME0, "TPP1")]
        assert len(caught) == 1
        assert "not applied" in str(caught[0].message)
        assert f"(byte {position})" in str(caught[0].message)

    def test_time_step_without_a_dif_reads_and_check_counts_it_missing(self, tmp_path):
        path = tmp_path / "missing_dif.arl"
        rng = np.random.default_rng(4)
        later = TIME0 + pd.Timedelta(hours=1)
        short = hrrr_like_step(rng, later)
        short[3] = short[3][:1]  # level 3 keeps its WWND and loses the DIFW
        path.write_bytes(
            raw_time_step(TIME0, hrrr_like_step(rng, TIME0))
            + raw_time_step(later, short)
        )

        with File(path) as arl:
            assert arl[TIME0][(3, "WWND")].diffs[0].variable == "DIFW"
            assert not arl[later][(3, "WWND")].diffs
            np.testing.assert_array_equal(
                arl[later][(3, "WWND")].read(), reference_unpack(short[3][0])
            )
            # The first time step has 6 parents and 4 DIF records.
            (problem,) = arl.check()
            assert "has 9 of the 10 data records" in problem


# --- Writing ------------------------------------------------------------------


class TestWrite:
    def test_generated_difs_are_the_residual_of_the_packed_parent(self, tmp_path):
        path = tmp_path / "multi_dif.arl"
        targets = write_multi_dif_file(path)

        pairs = pairs_in(path)
        assert len(pairs) == N_PAIRS == len(targets)
        with File(path) as arl:
            assert arl.check() == []
            for time, parent_h, parent, dif_h, dif in pairs:
                # The DIF follows its parent and shares its time, level and forecast.
                assert dif_h.variable == DIF_OF[parent_h.variable]
                assert (dif_h.time, dif_h.level, dif_h.forecast) == (
                    parent_h.time,
                    parent_h.level,
                    parent_h.forecast,
                )
                target = targets[time, parent_h.level, parent_h.variable]
                packed_parent = reference_unpack(parent)
                packed_dif = reference_unpack(dif)
                tolerance = 2 * dif_h.precision
                # DIF = target - packed parent, to within the DIF's own precision.
                residual = target - packed_parent
                np.testing.assert_allclose(packed_dif, residual, rtol=0, atol=tolerance)
                # The parent alone is off by far more than that, so the DIF is
                # what brings parent + DIF back to the target.
                assert np.abs(residual).max() > 10 * tolerance
                record = arl[time][(parent_h.level, parent_h.variable)]
                np.testing.assert_allclose(
                    record.read(), packed_parent + packed_dif, rtol=0, atol=1e-5
                )
                np.testing.assert_allclose(
                    record.read(), target, rtol=0, atol=tolerance
                )

    @pytest.mark.parametrize(
        ("level", "parent", "dif", "message"),
        [
            (
                1,
                "TEMP",
                "DIFW",
                "A DIFW record at level 1 corrects WWND or DZDT, not TEMP",
            ),
            (
                2,
                "WWND",
                "DIFT",
                "A DIFT record at level 2 corrects TKEN or TKEU, not WWND",
            ),
            (3, "UWND", "DIFX", r"corrects no record \(above the surface only DIFW"),
            (1, "TPP1", "DIFR", r"corrects no record \(above the surface only DIFW"),
            (0, "WWND", "DIFW", r"corrects TPP\* or PRT\* \(at the surface, any DIF\*"),
        ],
    )
    def test_a_dif_the_name_table_does_not_pair_is_refused(
        self, tmp_path, level, parent, dif, message
    ):
        # A reader would not pair this DIF with its parent, so it is not written.
        data = np.ones((NY, NX), dtype=np.float32)
        with File(
            tmp_path / "refused.arl",
            mode="w",
            source="TEST",
            grid=GRID,
            vertical_axis=AXIS,
        ) as arl:
            rs = arl.create_recordset(TIME0, forecast=0)
            with pytest.raises(ValueError, match=message):
                rs.create_datarecord(parent, level, forecast=0, data=data, diff=dif)
            with pytest.raises(ValueError, match=message):
                arl.add_record(TIME0, parent, level, forecast=0, data=data, diff=[dif])
            # Neither call left a record behind.
            assert not list(rs)
            rs.create_datarecord("T02M", 0, forecast=0, data=data)

    def test_every_pair_the_name_table_makes_is_written_and_read_back(self, tmp_path):
        # Above the surface DIFW goes with WWND or DZDT and DIFT with TKEN or
        # TKEU; at the surface any DIF* name goes with TPP* or PRT*.
        path = tmp_path / "table.arl"
        pairs = {
            (0, "TPP6"): "DIFR",
            (0, "PRT3"): "DIFW",
            (1, "WWND"): "DIFW",
            (1, "TKEU"): "DIFT",
            (2, "DZDT"): "DIFW",
            (2, "TKEN"): "DIFT",
        }
        rng = np.random.default_rng(17)
        targets = {
            key: rng.standard_normal((NY, NX)).astype(np.float32) for key in pairs
        }
        with File(path, mode="w", source="TEST", grid=GRID, vertical_axis=AXIS) as arl:
            for (level, name), dif in pairs.items():
                arl.add_record(
                    TIME0, name, level, forecast=0, data=targets[level, name], diff=dif
                )

        with File(path) as arl:
            assert arl.check() == []
            for (level, name), dif in pairs.items():
                record = arl[TIME0][(level, name)]
                assert [d.variable for d in record.diffs] == [dif]
                np.testing.assert_allclose(
                    record.read(),
                    targets[level, name],
                    rtol=0,
                    atol=2 * record.diffs[0].header.precision + 1e-6,
                )

    def test_a_dif_record_is_not_created_on_its_own(self, tmp_path):
        with File(
            tmp_path / "direct.arl",
            mode="w",
            source="TEST",
            grid=GRID,
            vertical_axis=AXIS,
        ) as arl:
            rs = arl.create_recordset(TIME0, forecast=0)
            with pytest.raises(
                ValueError, match="Create DIF records through the parent"
            ):
                rs.create_datarecord("DIFW", 1, forecast=0)


# --- Subsetting without a crop: the records are copied ------------------------


def kept_records(
    steps: dict[pd.Timestamp, Step],
    *,
    levels: list[int] | None = None,
    variables: list[str] | None = None,
) -> dict[pd.Timestamp, list[bytes]]:
    """
    The records an uncropped subset keeps, each with its level renumbered.

    A parent is kept when its level and its name are selected, and the DIF
    after it goes with it. Kept levels are renumbered from 0 in order.
    """
    kept_levels = sorted(set(levels)) if levels is not None else None
    out = {}
    for time, step in steps.items():
        out[time] = []
        parent_kept = False
        for header, record in step:
            if header.variable.startswith("DIF"):
                keep = parent_kept
            else:
                keep = (kept_levels is None or header.level in kept_levels) and (
                    variables is None or header.variable in variables
                )
                parent_kept = keep
            if keep:
                level = header.level
                if kept_levels is not None:
                    level = kept_levels.index(level)
                # The level is bytes 10 and 11 of the header.
                out[time].append(record[:10] + f"{level:2d}".encode() + record[12:])
    return out


def records_of(steps: dict[pd.Timestamp, Step]) -> dict[pd.Timestamp, list[bytes]]:
    """A walked file's whole records, by time step."""
    return {time: [record for _, record in step] for time, step in steps.items()}


def expected_listing(records: list[bytes]) -> list[tuple[int, str, int, str]]:
    """What an index record lists for ``records``: ``(level, name, checksum, flag)``."""
    listing = []
    for record in records:
        header = Header.from_bytes(record[: Header.N_BYTES])
        checksum = calculate_checksum(record[Header.N_BYTES :])
        listing.append((header.level, header.variable, checksum, ""))
    return listing


def index_listing(path: Path, time: pd.Timestamp) -> list[tuple[int, str, int, str]]:
    """What a time step's index record lists, in stream order: each parent, then its DIFs."""
    listing = []
    with File(path) as arl:
        for record in arl[time]:
            for rec in (record, *record.diffs):
                listing.append(
                    (rec.level, rec.variable, rec.checksum, rec._reserved or "")
                )
    return listing


SUBSETS = [
    pytest.param({}, id="all"),
    pytest.param({"levels": [0, 2, 3]}, id="levels"),
    pytest.param({"levels": [3, 0, 1]}, id="unsorted-levels"),
    pytest.param({"variables": ["TPP1", "WWND"]}, id="variables"),
    pytest.param(
        {"levels": [2, 0], "variables": ["UWND", "TKEN"]}, id="levels-and-variables"
    ),
]


class TestSubsetCopy:
    def test_uncropped_subset_of_a_multi_dif_file_is_the_same_file(self, tmp_path):
        source = tmp_path / "multi_dif.arl"
        subset = tmp_path / "subset.arl"
        write_multi_dif_file(source)

        extract_subset(source, subset)

        assert subset.read_bytes() == source.read_bytes()

    @pytest.mark.parametrize("kwargs", SUBSETS)
    def test_subset_keeps_every_parent_with_its_dif_byte_for_byte(
        self, tmp_path, kwargs
    ):
        source = tmp_path / "multi_dif.arl"
        subset = tmp_path / "subset.arl"
        write_multi_dif_file(source)

        extract_subset(source, subset, **kwargs)

        expected = kept_records(walk(source), **kwargs)
        assert records_of(walk(subset)) == expected
        assert any(
            Header.from_bytes(record[: Header.N_BYTES]).variable.startswith("DIF")
            for records in expected.values()
            for record in records
        )
        with File(subset) as out:
            assert out.check() == []
            assert out.times == list(expected)
        for time, records in expected.items():
            assert index_listing(subset, time) == expected_listing(records)

    def test_subset_of_another_producers_stream_rebuilds_only_the_index(self, tmp_path):
        source = tmp_path / "hrrr_like.arl"
        subset = tmp_path / "subset.arl"
        levels = hrrr_like_step(np.random.default_rng(5), TIME0)
        # NOAA stores a checksum of 0 for records it did not pack, and sets flags.
        stored = {
            (0, "TPP1"): (0, "R"),
            (0, "DIFR"): (0, "R"),
            (1, "WWND"): (7, "R"),
            (1, "DIFW"): (0, ""),
        }
        source.write_bytes(raw_time_step(TIME0, levels, stored=stored))
        assert index_listing(source, TIME0)[:2] == [
            (0, "TPP1", 0, "R"),
            (0, "DIFR", 0, "R"),
        ]

        extract_subset(source, subset, levels=[0, 1, 3])

        # The records are the source's bytes, level renumbered; the index record
        # lists their checksums and no flags, as arl-met writes it.
        expected = kept_records(walk(source), levels=[0, 1, 3])
        assert records_of(walk(subset)) == expected
        assert index_listing(subset, TIME0) == expected_listing(expected[TIME0])
        with File(subset) as out:
            assert out.check() == []
            for record in out[TIME0]:
                assert record.verify_checksum()
                assert all(dif.verify_checksum() for dif in record.diffs)
            np.testing.assert_array_equal(
                out[TIME0][(2, "WWND")].read(), combined(levels[3])
            )


# --- Rewriting: the records are unpacked and packed again ---------------------


class TestRewrite:
    def test_low_level_rewrite_keeps_the_stream_and_the_values(self, tmp_path):
        source = tmp_path / "multi_dif.arl"
        rewritten = tmp_path / "rewritten.arl"
        write_multi_dif_file(source)

        with (
            File(source) as src,
            File(
                rewritten,
                mode="w",
                source=src.source,
                grid=src.grid,
                vertical_axis=src.vertical_axis,
            ) as dst,
        ):
            for time in src.times:
                rs = dst.create_recordset(time, forecast=src[time].forecast)
                for record in src[time]:
                    rs.create_datarecord(
                        record.variable,
                        record.level,
                        forecast=record.header.forecast,
                        data=record.read(),
                        diff=[dif.variable for dif in record.diffs],
                    )

        # The same records in the same order with the same headers, and fields that
        # agree to within the DIF's precision (exactly, where there is no DIF).
        with File(source) as src, File(rewritten) as out:
            assert out.check() == []
            assert out.times == src.times
            for time in src.times:
                assert [(r.level, r.variable) for r in out[time]] == [
                    (r.level, r.variable) for r in src[time]
                ]
                for old, new in zip(src[time], out[time], strict=True):
                    assert new.header.forecast == old.header.forecast
                    if not old.diffs:
                        np.testing.assert_array_equal(new.read(), old.read())
                        continue
                    assert new.diffs
                    assert new.diffs[0].variable == old.diffs[0].variable
                    assert new.diffs[0].header.forecast == old.diffs[0].header.forecast
                    np.testing.assert_allclose(
                        new.read(),
                        old.read(),
                        rtol=0,
                        atol=2 * new.diffs[0].header.precision,
                    )
        # A record without a DIF is the same bytes; a parent that has one is packed
        # again from parent + DIF, so its bytes are not guaranteed to be the same.
        for old_step, new_step in zip(
            walk(source).values(), walk(rewritten).values(), strict=True
        ):
            for (header, old), (_, new) in zip(old_step, new_step, strict=True):
                if header.variable in ("UWND", "T02M"):
                    assert new == old

    def test_cropped_subset_writes_each_dif_against_its_new_parent(self, tmp_path):
        source = tmp_path / "multi_dif.arl"
        subset = tmp_path / "subset.arl"
        write_multi_dif_file(source)

        extract_subset(source, subset, bbox=WIDE_CROP)

        old_records = {
            (time, header.level, header.variable): record
            for time, step in walk(source).items()
            for header, record in step
        }
        with File(source) as src, File(subset) as out:
            assert out.check() == []
            window = src.grid.window_from_bbox(WIDE_CROP)
            assert (out.grid.nx, out.grid.ny) == window.shape[::-1]
            for time in src.times:
                assert [
                    (r.level, r.variable, r.header.forecast) for r in out[time]
                ] == [(r.level, r.variable, r.header.forecast) for r in src[time]]
        pairs = pairs_in(subset, nx=window.nx, ny=window.ny)
        assert len(pairs) == N_PAIRS
        for time, parent_h, parent, dif_h, dif in pairs:
            level, name = parent_h.level, parent_h.variable
            assert dif_h.variable == DIF_OF[name]
            assert (dif_h.level, dif_h.forecast) == (level, parent_h.forecast)
            # What the source holds in the window is the target of the new pair:
            # the new parent is packed from it, the DIF is what the packing left.
            source_field = reference_unpack(
                old_records[time, level, name]
            ) + reference_unpack(old_records[time, level, DIF_OF[name]])
            target = source_field[window.y_slice, window.x_slice]
            new_parent = reference_unpack(parent, nx=window.nx, ny=window.ny)
            new_dif = reference_unpack(dif, nx=window.nx, ny=window.ny)
            tolerance = 2 * dif_h.precision + 1e-6
            np.testing.assert_allclose(
                new_dif, target - new_parent, rtol=0, atol=tolerance
            )
            np.testing.assert_allclose(
                new_parent + new_dif, target, rtol=0, atol=tolerance
            )

    def test_cropped_subset_moves_the_correction_into_the_parent(self, tmp_path):
        # A parent of 10 and a DIF of 0.25 everywhere. The crop packs 10.25 as the
        # parent, which holds it exactly, so the new DIF has nothing left to correct.
        source = tmp_path / "flat.arl"
        subset = tmp_path / "subset.arl"
        flat = np.full((NY, NX), 127)
        parent = raw_record(TIME0, 0, "TPP1", flat, initial_value=10.0)
        dif = raw_record(TIME0, 0, "DIFR", flat, initial_value=0.25)
        source.write_bytes(raw_time_step(TIME0, [[parent, dif], [], [], []]))

        extract_subset(source, subset, bbox=CROP, levels=[0])

        ((_, step),) = walk(subset, nx=15, ny=15).items()
        ((parent_h, new_parent), (dif_h, new_dif)) = step
        assert (parent_h.variable, dif_h.variable) == ("TPP1", "DIFR")
        with File(subset) as out:
            assert (out.grid.nx, out.grid.ny) == (15, 15)
            assert out.check() == []
            np.testing.assert_array_equal(
                out[TIME0][(0, "TPP1")].read(), np.full((15, 15), 10.25)
            )
        np.testing.assert_array_equal(
            reference_unpack(new_parent, nx=15, ny=15), np.full((15, 15), 10.25)
        )
        np.testing.assert_array_equal(
            reference_unpack(new_dif, nx=15, ny=15), np.zeros((15, 15))
        )


# --- Datasets -----------------------------------------------------------------


class TestDataset:
    def test_open_dataset_adds_each_dif_to_its_parent(self, tmp_path):
        path = tmp_path / "multi_dif.arl"
        write_multi_dif_file(path)
        ds = open_dataset(path)
        cropped = open_dataset(path, bbox=WIDE_CROP)

        assert not [name for name in ds.data_vars if str(name).startswith("DIF")]
        with File(path) as arl:
            window = arl.grid.window_from_bbox(WIDE_CROP)
            for time in arl.times:
                for record in arl[time]:
                    if not record.diffs:
                        continue
                    values = ds[record.variable].sel(time=time)
                    part = cropped[record.variable].sel(time=time)
                    if record.level:
                        values = values.sel(level=record.level)
                        part = part.sel(level=record.level)
                    np.testing.assert_array_equal(values.values, record.read())
                    np.testing.assert_array_equal(
                        part.values, record.read(window=window)
                    )

    def test_write_dataset_writes_a_dif_for_each_declared_parent(self, tmp_path):
        source = tmp_path / "multi_dif.arl"
        written = tmp_path / "written.arl"
        write_multi_dif_file(source)
        ds = open_dataset(source)
        for parent, dif in DIF_OF.items():
            ds[parent].attrs["diff"] = dif

        write_dataset(ds, written)

        with File(source) as src, File(written) as out:
            assert out.check() == []
            assert out.times == src.times
            for time in src.times:
                assert sorted((r.level, r.variable) for r in out[time]) == sorted(
                    (r.level, r.variable) for r in src[time]
                )
                for record in out[time]:
                    if record.variable not in DIF_OF:
                        assert not record.diffs
                        continue
                    assert record.diffs
                    assert record.diffs[0].variable == DIF_OF[record.variable]
                    # The pair adds up to the Dataset's values.
                    old = src[time][(record.level, record.variable)]
                    np.testing.assert_allclose(
                        record.read(),
                        old.read(),
                        rtol=0,
                        atol=2 * record.diffs[0].header.precision + 1e-6,
                    )
        # Each DIF is in the stream right after its parent, on its level.
        for step in walk(written).values():
            for (header, _), (next_header, _) in pairwise(step):
                if next_header.variable.startswith("DIF"):
                    assert header.variable == PARENT_OF[next_header.variable]
                    assert header.level == next_header.level
        assert len(pairs_in(written)) == N_PAIRS

    def test_write_dataset_writes_no_dif_for_a_parent_that_declares_none(
        self, tmp_path
    ):
        source = tmp_path / "multi_dif.arl"
        written = tmp_path / "written.arl"
        write_multi_dif_file(source)
        ds = open_dataset(source)
        ds["WWND"].attrs["diff"] = "DIFW"

        write_dataset(ds, written)

        with File(written) as out:
            with_dif = {(r.level, r.variable) for r in out[TIME0] if r.diffs}
        assert with_dif == {(1, "WWND"), (2, "WWND"), (3, "WWND")}


# --- Several DIF records after one parent -------------------------------------
#
# HYSPLIT's reader (metinp) adds each DIFW it reads into the vertical velocity of
# its level, each DIFT into the TKE and, at the surface, each DIF* into the
# precipitation, one after another, so two DIF records after one parent both
# count. arl-met reads them the same way: every DIF record that the name table
# pairs with a parent is added to it in stream order.

# The DIF records after each parent in ``chain_step``, by level.
CHAINS = {
    (0, "TPP1"): ("DIFR", "DIFR"),
    (1, "WWND"): ("DIFW", "DIFW"),
    (2, "WWND"): ("DIFW", "DIFW", "DIFW"),
    (2, "TKEN"): ("DIFT", "DIFT"),
    (3, "WWND"): ("DIFW",),
}


def parent_and_difs(
    rng: np.random.Generator,
    time: pd.Timestamp,
    level: int,
    parent: str,
    names: tuple[str, ...],
    *,
    start: float,
) -> list[bytes]:
    """
    A parent record and the DIF records ``names`` after it, with exact values.

    The parent starts at ``start`` and steps by 0.25; the k-th DIF starts at
    ``0.5 / 2**k`` and steps by 1/128, so every value and every sum is exact.
    """
    return [
        raw_record(time, level, parent, noise(rng), exponent=5, initial_value=start),
        *(
            raw_record(time, level, name, noise(rng), initial_value=0.5 / 2**k)
            for k, name in enumerate(names)
        ),
    ]


def chain_step(rng: np.random.Generator, time: pd.Timestamp) -> list[list[bytes]]:
    """
    One time step whose parents have one, two or three DIF records each.

    Level 0 has ``TPP1`` with two ``DIFR``; level 1 has ``UWND``, ``WWND`` with
    two ``DIFW``, then ``TEMP``; level 2 ``WWND`` with three ``DIFW`` and
    ``TKEN`` with two ``DIFT``; level 3 ``WWND`` with one ``DIFW``.
    """

    def pair(level: int, name: str, start: float) -> list[bytes]:
        return parent_and_difs(rng, time, level, name, CHAINS[level, name], start=start)

    return [
        pair(0, "TPP1", 2.0),
        [
            raw_record(time, 1, "UWND", noise(rng), exponent=5, initial_value=3.0),
            *pair(1, "WWND", 10.0),
            raw_record(time, 1, "TEMP", noise(rng), exponent=5, initial_value=280.0),
        ],
        [*pair(2, "WWND", 20.0), *pair(2, "TKEN", 1.0)],
        pair(3, "WWND", 30.0),
    ]


def write_chain_file(path: Path, n_steps: int = 2) -> list[pd.Timestamp]:
    """Write ``n_steps`` hourly time steps from ``chain_step``."""
    rng = np.random.default_rng(7)
    times = list(pd.date_range(TIME0, periods=n_steps, freq="h"))
    path.write_bytes(
        b"".join(raw_time_step(time, chain_step(rng, time)) for time in times)
    )
    return times


def chains_in(
    path: Path, *, nx: int = NX, ny: int = NY
) -> dict[tuple[pd.Timestamp, int, str], list[tuple[Header, bytes]]]:
    """Each parent in a file with the DIF records after it, by ``(time, level, name)``."""
    found: dict[tuple[pd.Timestamp, int, str], list[tuple[Header, bytes]]] = {}
    for time, step in walk(path, nx=nx, ny=ny).items():
        chain: list[tuple[Header, bytes]] = []
        for header, record in step:
            if header.variable.startswith("DIF"):
                chain.append((header, record))
                continue
            chain = [(header, record)]
            found[time, header.level, header.variable] = chain
    return found


class TestSeveralDifs:
    def test_each_dif_in_a_row_adds_to_the_parent(self, tmp_path):
        path = tmp_path / "chains.arl"
        times = write_chain_file(path)
        window = GridWindow(x_start=3, x_stop=12, y_start=2, y_stop=9)

        with File(path) as arl:
            assert arl.times == times
            assert arl.check() == []
            chains = chains_in(path)
            for time in times:
                rs = arl[time]
                # The DIF records are part of their parents, not records of the
                # time step.
                assert [(r.level, r.variable) for r in rs] == [
                    (0, "TPP1"),
                    (1, "UWND"),
                    (1, "WWND"),
                    (1, "TEMP"),
                    (2, "WWND"),
                    (2, "TKEN"),
                    (3, "WWND"),
                ]
                assert "DIFW" not in rs
                for record in rs:
                    key = (record.level, record.variable)
                    chain = [raw for _, raw in chains[(time, *key)]]
                    assert tuple(d.variable for d in record.diffs) == CHAINS.get(
                        key, ()
                    )
                    assert all(d.level == record.level for d in record.diffs)
                    np.testing.assert_array_equal(record.read(), combined(chain))
                    np.testing.assert_array_equal(
                        record.read(window=window),
                        combined(chain)[window.y_slice, window.x_slice],
                    )
                    for dif, raw in zip(record.diffs, chain[1:], strict=True):
                        np.testing.assert_array_equal(dif.read(), reference_unpack(raw))

    def test_difs_are_added_one_after_another_in_float32_as_hysplit_does(
        self, tmp_path
    ):
        # HYSPLIT adds each DIF into the field as it reads it, in REAL (float32):
        # (parent + DIF) + DIF. Near 2**24 float32 steps by 2, so adding 1 twice
        # leaves 16777220 where adding the two DIFs first would give 16777222.
        path = tmp_path / "order.arl"
        flat = np.full((NY, NX), 127)  # every step 0: each field is constant
        parent = raw_record(TIME0, 1, "WWND", flat, initial_value=16777220.0)
        difs = [raw_record(TIME0, 1, "DIFW", flat, initial_value=1.0)] * 2
        path.write_bytes(raw_time_step(TIME0, [[], [parent, *difs], [], []]))

        expected = np.float32(16777220.0)
        for _ in difs:
            expected = np.float32(expected + np.float32(1.0))
        assert expected == np.float32(16777220.0)
        with File(path) as arl:
            values = arl[TIME0][(1, "WWND")].read()
        assert values.dtype == np.float32
        np.testing.assert_array_equal(values, np.full((NY, NX), expected))

    def test_the_index_record_keeps_a_repeated_dif_name(self, tmp_path):
        path = tmp_path / "chains.arl"
        write_chain_file(path, n_steps=1)

        with open(path, "rb") as handle:
            index = IndexRecord.from_position(handle, 0)
        assert [name for name, _ in index.levels[1].variables] == [
            "UWND",
            "WWND",
            "DIFW",
            "DIFW",
            "TEMP",
        ]
        assert index.to_record_bytes(RECORD_BYTES) == path.read_bytes()[:RECORD_BYTES]

    def test_generated_difs_each_hold_what_packing_left(self, tmp_path):
        path = tmp_path / "written.arl"
        rng = np.random.default_rng(13)
        wwnd = (0.3 * rng.standard_normal((NY, NX))).astype(np.float32)
        tpp1 = np.abs(0.001 + 0.0004 * rng.standard_normal((NY, NX))).astype(np.float32)
        with File(path, mode="w", source="TEST", grid=GRID, vertical_axis=AXIS) as arl:
            rs = arl.create_recordset(TIME0, forecast=0)
            rs.create_datarecord(
                "TPP1", 0, forecast=3, data=tpp1, diff=("DIFR", "DIFR")
            )
            rs.create_datarecord("TEMP", 1, forecast=0, data=wwnd + 280, diff=[])
            arl.add_record(TIME0, "WWND", 1, forecast=0, data=wwnd, diff=["DIFW"] * 3)

        chains = chains_in(path)
        assert [h.variable for h, _ in chains[TIME0, 1, "WWND"]] == ["WWND"] + [
            "DIFW"
        ] * 3
        assert [h.variable for h, _ in chains[TIME0, 0, "TPP1"]] == [
            "TPP1",
            "DIFR",
            "DIFR",
        ]
        assert len(chains[TIME0, 1, "TEMP"]) == 1  # an empty list writes no DIF
        with File(path) as out:
            assert out.check() == []
            for (_, level, name), target in {
                (TIME0, 0, "TPP1"): tpp1,
                (TIME0, 1, "WWND"): wwnd,
            }.items():
                chain = chains[TIME0, level, name]
                parent_h = chain[0][0]
                # Each DIF follows its parent with its time, level and forecast...
                for header, _ in chain[1:]:
                    assert (header.time, header.level, header.forecast) == (
                        parent_h.time,
                        parent_h.level,
                        parent_h.forecast,
                    )
                # ...and is the target minus what the records before it hold.
                packed = reference_unpack(chain[0][1])
                for header, raw in chain[1:]:
                    tolerance = 2 * header.precision + 1e-7
                    np.testing.assert_allclose(
                        reference_unpack(raw), target - packed, rtol=0, atol=tolerance
                    )
                    packed = packed + reference_unpack(raw)
                np.testing.assert_allclose(
                    out[TIME0][(level, name)].read(), packed, rtol=0, atol=1e-6
                )
                np.testing.assert_allclose(packed, target, rtol=0, atol=tolerance)
            # The second DIF corrects what the first left, not the parent alone.
            chain = chains[TIME0, 1, "WWND"]
            second_h, second = chain[2]
            left_by_parent = wwnd - reference_unpack(chain[0][1])
            assert (
                np.abs(reference_unpack(second) - left_by_parent).max()
                > 10 * second_h.precision
            )

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({}, id="all"),
            pytest.param({"levels": [0, 2]}, id="levels"),
            pytest.param({"variables": ["WWND", "TEMP"]}, id="variables"),
        ],
    )
    def test_uncropped_subset_copies_every_dif_byte_for_byte(self, tmp_path, kwargs):
        source = tmp_path / "chains.arl"
        subset = tmp_path / "subset.arl"
        write_chain_file(source)

        extract_subset(source, subset, **kwargs)

        expected = kept_records(walk(source), **kwargs)
        assert records_of(walk(subset)) == expected
        with File(subset) as out:
            assert out.check() == []
            assert out.times == list(expected)
            assert any(len(r.diffs) > 1 for r in out[TIME0])
        for time, records in expected.items():
            assert index_listing(subset, time) == expected_listing(records)
        if not kwargs:
            assert subset.read_bytes() == source.read_bytes()

    def test_cropped_subset_keeps_each_parents_difs(self, tmp_path):
        source = tmp_path / "chains.arl"
        subset = tmp_path / "subset.arl"
        write_chain_file(source)

        extract_subset(source, subset, bbox=WIDE_CROP)

        with File(source) as src, File(subset) as out:
            assert out.check() == []
            window = src.grid.window_from_bbox(WIDE_CROP)
            assert (out.grid.nx, out.grid.ny) == window.shape[::-1]
            for time in src.times:
                old_records, new_records = list(src[time]), list(out[time])
                assert [
                    (r.level, r.variable, tuple(d.variable for d in r.diffs))
                    for r in new_records
                ] == [
                    (r.level, r.variable, tuple(d.variable for d in r.diffs))
                    for r in old_records
                ]
                for old, new in zip(old_records, new_records, strict=True):
                    target = old.read(window=window)
                    if not old.diffs:
                        np.testing.assert_array_equal(new.read(), target)
                        continue
                    # The parent is packed from the source's sum and each DIF
                    # holds what packing left, so the new chain adds up to it.
                    precision = max(d.header.precision for d in new.diffs)
                    np.testing.assert_allclose(
                        new.read(), target, rtol=0, atol=2 * precision + 1e-6
                    )

    def test_a_crop_too_small_for_every_dif_in_the_index_is_refused(self, tmp_path):
        # 16 x 16 cells hold the index record with one DIF per parent (286
        # bytes) but not with all of them (326 bytes), so the check before
        # writing must count every DIF.
        source = tmp_path / "chains.arl"
        subset = tmp_path / "subset.arl"
        write_chain_file(source, n_steps=1)
        bbox = (21.0, -9.0, 36.0, 6.0)
        with File(source) as src:
            assert src.grid.window_from_bbox(bbox).shape == (16, 16)

        with pytest.raises(ValueError, match="needs 326 bytes"):
            extract_subset(source, subset, bbox=bbox)
        assert not subset.exists()

    def test_low_level_rewrite_keeps_the_stream(self, tmp_path):
        source = tmp_path / "chains.arl"
        rewritten = tmp_path / "rewritten.arl"
        write_chain_file(source)

        with (
            File(source) as src,
            File(
                rewritten,
                mode="w",
                source=src.source,
                grid=src.grid,
                vertical_axis=src.vertical_axis,
            ) as dst,
        ):
            for time in src.times:
                rs = dst.create_recordset(time, forecast=src[time].forecast)
                for record in src[time]:
                    rs.create_datarecord(
                        record.variable,
                        record.level,
                        forecast=record.header.forecast,
                        data=record.read(),
                        diff=[d.variable for d in record.diffs],
                    )

        # The same records in the same order, and fields that agree to within
        # the DIFs' precision.
        assert [[h.variable for h, _ in step] for step in walk(rewritten).values()] == [
            [h.variable for h, _ in step] for step in walk(source).values()
        ]
        with File(source) as src, File(rewritten) as out:
            assert out.check() == []
            for time in src.times:
                for old, new in zip(src[time], out[time], strict=True):
                    atol = 2 * max((d.header.precision for d in new.diffs), default=0)
                    np.testing.assert_allclose(
                        new.read(), old.read(), rtol=0, atol=atol + 1e-6
                    )

    def test_open_dataset_adds_every_dif(self, tmp_path):
        path = tmp_path / "chains.arl"
        write_chain_file(path, n_steps=1)
        chains = chains_in(path)

        ds = open_dataset(path)

        assert not [name for name in ds.data_vars if str(name).startswith("DIF")]
        np.testing.assert_array_equal(
            ds["TPP1"].sel(time=TIME0).values,
            combined([raw for _, raw in chains[TIME0, 0, "TPP1"]]),
        )
        for level in (1, 2, 3):
            expected = combined([raw for _, raw in chains[TIME0, level, "WWND"]])
            np.testing.assert_array_equal(
                ds["WWND"].sel(time=TIME0, level=level).values, expected
            )


# --- Pairing by name ----------------------------------------------------------
#
# HYSPLIT's reader (metinp) pairs a DIF record by its name, not its place: DIFW
# goes into the vertical velocity of its level (filled by WWND or DZDT), DIFT
# into the TKE (TKEN or TKEU), and at the surface any DIF* into the
# precipitation (TPP* or PRT*). NOAA's era52arl lists DIFW and DIFR last on each
# level, after RELH and USTR, so the record before a DIF is not its parent.

# Each DIF in ``era5_like_step``'s file and the record it corrects.
ERA5_PARENT_OF = {"DIFR": "TPP1", "DIFW": "WWND"}


def era5_like_step(rng: np.random.Generator, time: pd.Timestamp) -> list[list[bytes]]:
    """
    One time step in era52arl's default order, each level's DIF record last.

    Level 0 has ``TPP1 USTR DIFR``; levels 1 to 3 have ``UWND WWND RELH DIFW``.
    """

    def rec(level: int, name: str, start: float) -> bytes:
        exponent = 0 if name.startswith("DIF") else 5
        return raw_record(
            time, level, name, noise(rng), exponent=exponent, initial_value=start
        )

    return [
        [rec(0, "TPP1", 1.0), rec(0, "USTR", 0.25), rec(0, "DIFR", 0.5)],
        *(
            [
                rec(level, "UWND", 2.0),
                rec(level, "WWND", 3.0),
                rec(level, "RELH", 50.0),
                rec(level, "DIFW", 0.5),
            ]
            for level in (1, 2, 3)
        ),
    ]


def write_era5_like_file(path: Path, n_steps: int = 2) -> list[pd.Timestamp]:
    """Write ``n_steps`` hourly time steps from ``era5_like_step``."""
    rng = np.random.default_rng(0)
    times = list(pd.date_range(TIME0, periods=n_steps, freq="h"))
    path.write_bytes(
        b"".join(raw_time_step(time, era5_like_step(rng, time)) for time in times)
    )
    return times


def expected_values(path: Path) -> dict[tuple[pd.Timestamp, int, str], np.ndarray]:
    """
    Each data record's values in a file of ``era5_like_step``, by name pairing.

    A parent is its record plus its DIF; any other record is its own values.
    """
    expected = {}
    for time, step in walk(path).items():
        records = {(h.level, h.variable): raw for h, raw in step}
        for (level, name), raw in records.items():
            if name.startswith("DIF"):
                continue
            difs = [d for d, parent in ERA5_PARENT_OF.items() if parent == name]
            expected[time, level, name] = combined(
                [raw, *(records[level, dif] for dif in difs)]
            )
    return expected


def raw_index_listing(path: Path, *, nx: int = NX, ny: int = NY) -> list[list[str]]:
    """Each time step's ``level:name`` entries as its index record lists them."""
    length = Header.N_BYTES + nx * ny
    raw = path.read_bytes()
    listings = []
    with open(path, "rb") as handle:
        for start in range(0, len(raw), length):
            if (
                Header.from_bytes(raw[start : start + Header.N_BYTES]).variable
                != "INDX"
            ):
                continue
            index = IndexRecord.from_position(handle, start)
            listings.append(
                [
                    f"{lvl.level}:{name}"
                    for lvl in index.levels
                    for name, _ in lvl.variables
                ]
            )
    return listings


def disk_listing(path: Path, *, nx: int = NX, ny: int = NY) -> list[list[str]]:
    """Each time step's ``level:name`` records in the order they are on disk."""
    return [
        [f"{h.level}:{h.variable}" for h, _ in step]
        for step in walk(path, nx=nx, ny=ny).values()
    ]


class TestPairByName:
    def test_era52arl_layout_pairs_each_dif_with_its_parent(self, tmp_path):
        path = tmp_path / "era5_like.arl"
        times = write_era5_like_file(path)
        expected = expected_values(path)

        with File(path) as arl:
            assert arl.check() == []
            for time in times:
                for record in arl[time]:
                    level, name = record.level, record.variable
                    assert [d.variable for d in record.diffs] == [
                        d for d, parent in ERA5_PARENT_OF.items() if parent == name
                    ]
                    np.testing.assert_array_equal(
                        record.read(), expected[time, level, name]
                    )

    def test_open_dataset_adds_the_dif_to_its_parent(self, tmp_path):
        path = tmp_path / "era5_like.arl"
        write_era5_like_file(path, n_steps=1)
        expected = expected_values(path)

        ds = open_dataset(path)

        np.testing.assert_array_equal(
            ds["TPP1"].sel(time=TIME0).values, expected[TIME0, 0, "TPP1"]
        )
        np.testing.assert_array_equal(
            ds["USTR"].sel(time=TIME0).values, expected[TIME0, 0, "USTR"]
        )
        for level in (1, 2, 3):
            for name in ("WWND", "RELH"):
                np.testing.assert_array_equal(
                    ds[name].sel(time=TIME0, level=level).values,
                    expected[TIME0, level, name],
                )

    @pytest.mark.parametrize(
        ("names", "parent"),
        [
            # The nearest record before the DIF that its name allows.
            (["WWND", "DZDT", "DIFW"], "DZDT"),
            (["DZDT", "WWND", "DIFW"], "WWND"),
            (["WWND", "TEMP", "TKEN", "DIFW"], "WWND"),
            (["TKEU", "WWND", "DIFT"], "TKEU"),
            (["TKEN", "DIFT", "TKEU", "DIFT"], None),  # checked in the test
        ],
        ids=["dzdt-last", "wwnd-last", "others-between", "tkeu", "two-parents"],
    )
    def test_dif_goes_with_the_nearest_record_its_name_allows(
        self, tmp_path, names, parent
    ):
        path = tmp_path / "upper.arl"
        rng = np.random.default_rng(23)
        records = [
            raw_record(
                TIME0, 1, name, noise(rng), exponent=0 if name.startswith("DIF") else 5
            )
            for name in names
        ]
        path.write_bytes(raw_time_step(TIME0, [[], records, [], []]))

        with File(path) as arl:
            rs = arl[TIME0]
            got = {r.variable: [d.variable for d in r.diffs] for r in rs}
            if parent is None:
                # Each DIFT corrects the TKE record nearest before it.
                assert got == {"TKEN": ["DIFT"], "TKEU": ["DIFT"]}
                np.testing.assert_array_equal(
                    rs[(1, "TKEN")].read(), combined(records[:2])
                )
                np.testing.assert_array_equal(
                    rs[(1, "TKEU")].read(), combined(records[2:])
                )
                return
            dif = records[-1]
            for record, raw in zip(rs, records[:-1], strict=True):
                if record.variable == parent:
                    assert got[parent] == [names[-1]]
                    np.testing.assert_array_equal(record.read(), combined([raw, dif]))
                else:
                    assert not record.diffs
                    np.testing.assert_array_equal(record.read(), reference_unpack(raw))

    def test_at_the_surface_any_dif_corrects_the_precipitation(self, tmp_path):
        # HYSPLIT matches the first three letters at the surface: DIFW there
        # adds into the precipitation, and PRT* fills it like TPP*.
        path = tmp_path / "surface.arl"
        rng = np.random.default_rng(29)
        names = ["PRT6", "T02M", "DIFW", "DIFR"]
        records = [raw_record(TIME0, 0, name, noise(rng)) for name in names]
        path.write_bytes(raw_time_step(TIME0, [records, [], [], []]))

        with File(path) as arl:
            prt6 = arl[TIME0][(0, "PRT6")]
            assert [d.variable for d in prt6.diffs] == ["DIFW", "DIFR"]
            assert not arl[TIME0][(0, "T02M")].diffs
            np.testing.assert_array_equal(
                prt6.read(), combined([records[0], records[2], records[3]])
            )

    @pytest.mark.parametrize(
        ("level", "names", "unpaired"),
        [
            (1, ["WWND", "DIFX"], "DIFX"),  # a name the table does not know
            (1, ["WWND", "DIFR"], "DIFR"),  # DIFR pairs only at the surface
            (1, ["TEMP", "DIFW"], "DIFW"),  # no WWND or DZDT on the level
            (1, ["DIFW", "WWND"], "DIFW"),  # the parent comes after the DIF
            (0, ["T02M", "DIFR"], "DIFR"),  # no TPP* or PRT* at the surface
        ],
        ids=["unknown-name", "upper-difr", "no-parent", "parent-after", "surface"],
    )
    def test_a_dif_the_name_table_does_not_pair_is_applied_to_nothing(
        self, tmp_path, level, names, unpaired
    ):
        path = tmp_path / "unpaired.arl"
        rng = np.random.default_rng(31)
        records = [raw_record(TIME0, level, name, noise(rng)) for name in names]
        levels: list[list[bytes]] = [[], [], [], []]
        levels[level] = records
        path.write_bytes(raw_time_step(TIME0, levels))

        with (
            pytest.warns(ARLFormatWarning) as caught,
            File(path) as arl,
        ):
            data = {r.variable: r for r in arl[TIME0]}
            assert not any(r.diffs for r in data.values())
            for name, raw in zip(names, records, strict=True):
                if name in data:
                    np.testing.assert_array_equal(
                        data[name].read(), reference_unpack(raw)
                    )
        (warning,) = caught
        message = str(warning.message)
        assert f"{unpaired} at level {level} (byte" in message
        assert "not applied" in message
        assert "DIFW corrects WWND or DZDT" in message

    def test_one_warning_counts_every_unpaired_dif(self, tmp_path):
        path = tmp_path / "unpaired.arl"
        rng = np.random.default_rng(37)
        levels = [
            [raw_record(TIME0, level, name, noise(rng)) for name in ("TEMP", "DIFW")]
            for level in range(4)
        ]
        path.write_bytes(raw_time_step(TIME0, levels))

        with pytest.warns(ARLFormatWarning) as caught, File(path):
            pass
        (warning,) = caught
        assert "4 DIF record(s)" in str(warning.message)

    def test_low_level_rewrite_puts_each_dif_after_its_parent(self, tmp_path):
        source = tmp_path / "era5_like.arl"
        rewritten = tmp_path / "rewritten.arl"
        write_era5_like_file(source)

        with (
            File(source) as src,
            File(
                rewritten,
                mode="w",
                source=src.source,
                grid=src.grid,
                vertical_axis=src.vertical_axis,
            ) as dst,
        ):
            for time in src.times:
                rs = dst.create_recordset(time, forecast=src[time].forecast)
                for record in src[time]:
                    rs.create_datarecord(
                        record.variable,
                        record.level,
                        forecast=record.header.forecast,
                        data=record.read(),
                        diff=[d.variable for d in record.diffs],
                    )

        moved = [
            ["0:TPP1", "0:DIFR", "0:USTR"],
            *([f"{n}:UWND", f"{n}:WWND", f"{n}:DIFW", f"{n}:RELH"] for n in (1, 2, 3)),
        ]
        moved_step = [entry for level in moved for entry in level]
        assert disk_listing(rewritten) == [moved_step] * 2
        assert raw_index_listing(rewritten) == disk_listing(rewritten)
        expected = expected_values(source)
        with File(rewritten) as out:
            assert out.check() == []
            for time in out.times:
                for record in out[time]:
                    atol = 2 * max(
                        (d.header.precision for d in record.diffs), default=0
                    )
                    np.testing.assert_allclose(
                        record.read(),
                        expected[time, record.level, record.variable],
                        rtol=0,
                        atol=atol + 1e-6,
                    )

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({}, id="all"),
            pytest.param({"levels": [0, 2]}, id="levels"),
            pytest.param({"variables": ["WWND", "USTR", "TPP1"]}, id="variables"),
        ],
    )
    def test_uncropped_subset_copies_each_dif_after_its_parent(self, tmp_path, kwargs):
        source = tmp_path / "era5_like.arl"
        subset = tmp_path / "subset.arl"
        write_era5_like_file(source)

        extract_subset(source, subset, **kwargs)

        # The index lists the records in the order they are on disk, each DIF
        # right after its parent.
        listing = disk_listing(subset)
        assert raw_index_listing(subset) == listing
        for step in listing:
            for before, entry in pairwise(step):
                level, name = entry.split(":")
                if name.startswith("DIF"):
                    assert before == f"{level}:{ERA5_PARENT_OF[name]}"
        # Each record is the source's bytes with its level renumbered, and the
        # values are exactly the source's, DIF included.
        kept = sorted(kwargs.get("levels", range(4)))
        source_records = {
            (time, h.level, h.variable): raw
            for time, step in walk(source).items()
            for h, raw in step
        }
        for time, step in walk(subset).items():
            for header, raw in step:
                old = source_records[time, kept[header.level], header.variable]
                assert raw[:10] == old[:10] and raw[12:] == old[12:]
        expected = expected_values(source)
        with File(subset) as out:
            assert out.check() == []
            n_difs = 0
            for time in out.times:
                for record in out[time]:
                    n_difs += len(record.diffs)
                    np.testing.assert_array_equal(
                        record.read(),
                        expected[time, kept[record.level], record.variable],
                    )
            assert n_difs > 0

    def test_cropped_subset_writes_each_dif_after_its_parent(self, tmp_path):
        source = tmp_path / "era5_like.arl"
        subset = tmp_path / "subset.arl"
        write_era5_like_file(source)

        extract_subset(source, subset, bbox=WIDE_CROP)

        with File(source) as src:
            window = src.grid.window_from_bbox(WIDE_CROP)
        nx, ny = window.nx, window.ny
        listing = disk_listing(subset, nx=nx, ny=ny)
        assert raw_index_listing(subset, nx=nx, ny=ny) == listing
        for step in listing:
            for before, entry in pairwise(step):
                level, name = entry.split(":")
                if name.startswith("DIF"):
                    assert before == f"{level}:{ERA5_PARENT_OF[name]}"
        expected = expected_values(source)
        with File(subset) as out:
            assert out.check() == []
            for time in out.times:
                for record in out[time]:
                    target = expected[time, record.level, record.variable]
                    target = target[window.y_slice, window.x_slice]
                    if record.variable in ERA5_PARENT_OF.values():
                        assert [d.variable for d in record.diffs] == [
                            d for d, p in ERA5_PARENT_OF.items() if p == record.variable
                        ]
                        atol = 2 * record.diffs[0].header.precision + 1e-6
                    else:
                        assert not record.diffs
                        atol = 0
                    np.testing.assert_allclose(record.read(), target, rtol=0, atol=atol)

    @pytest.mark.parametrize("bbox", [None, WIDE_CROP], ids=["copy", "crop"])
    def test_a_subset_that_moves_a_dif_to_the_surface_is_refused(self, tmp_path, bbox):
        # Leaving out level 0 makes level 1 the surface, where a DIFW record
        # would correct nothing (HYSPLIT's xtrct_grid always keeps level 0).
        source = tmp_path / "era5_like.arl"
        subset = tmp_path / "subset.arl"
        write_era5_like_file(source, n_steps=1)

        with pytest.raises(ValueError, match="puts level 1 at level 0") as caught:
            extract_subset(source, subset, bbox=bbox, levels=[1, 2])
        assert "DIFW record of WWND would correct nothing" in str(caught.value)
        assert not subset.exists()

        # Without the records that carry a DIF, the same levels are fine.
        extract_subset(source, subset, bbox=bbox, levels=[1, 2], variables=["RELH"])
        with File(subset) as out:
            assert [(r.level, r.variable) for r in out[TIME0]] == [
                (0, "RELH"),
                (1, "RELH"),
            ]
