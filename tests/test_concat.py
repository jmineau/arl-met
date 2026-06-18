"""Tests for concatenating ARL files into a single file."""

import numpy as np
import pandas as pd
import pytest

from arlmet import File, concat, concat_by_time
from arlmet.grid import Grid, Projection
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


def write_arl(path, times, *, grid=None, vertical_axis=None, source="TEST"):
    """Write a small ARL file with PRSS+TEMP records at each given time."""
    grid = grid if grid is not None else make_test_grid()
    vertical_axis = (
        vertical_axis
        if vertical_axis is not None
        else PressureAxis(levels=[0.0, 1000.0])
    )
    base = np.arange(grid.nx * grid.ny, dtype=np.float32).reshape(grid.ny, grid.nx)

    with File(
        path, mode="w", source=source, grid=grid, vertical_axis=vertical_axis
    ) as arl:
        for i, time in enumerate(times):
            time = pd.Timestamp(time)
            rs = arl.create_recordset(time, forecast=0)
            rs.create_datarecord("PRSS", level=0, forecast=0, data=1000.0 + base + i)
            rs.create_datarecord("TEMP", level=1, forecast=0, data=280.0 + base + i)
    return path


def test_concat_joins_files_in_time_order(tmp_path):
    early = write_arl(tmp_path / "early.arl", ["2024-01-01 00:00", "2024-01-01 06:00"])
    late = write_arl(tmp_path / "late.arl", ["2024-01-01 12:00", "2024-01-01 18:00"])
    out = tmp_path / "day.arl"

    # Pass out of order; sort=True should chronologically order the output.
    concat([late, early], out)

    with File(out) as merged:
        assert merged.times == [
            pd.Timestamp("2024-01-01 00:00"),
            pd.Timestamp("2024-01-01 06:00"),
            pd.Timestamp("2024-01-01 12:00"),
            pd.Timestamp("2024-01-01 18:00"),
        ]
        assert merged.grid == make_test_grid()
        assert merged.vertical_axis.levels.tolist() == [0.0, 1000.0]


def test_concat_is_byte_level_append(tmp_path):
    a = write_arl(tmp_path / "a.arl", ["2024-01-01 00:00"])
    b = write_arl(tmp_path / "b.arl", ["2024-01-01 06:00"])
    out = tmp_path / "ab.arl"

    concat([a, b], out, sort=False)

    # With sort=False the output is exactly the inputs appended in order.
    assert out.read_bytes() == a.read_bytes() + b.read_bytes()


def test_concat_preserves_record_values(tmp_path):
    a = write_arl(tmp_path / "a.arl", ["2024-01-01 00:00"])
    b = write_arl(tmp_path / "b.arl", ["2024-01-01 06:00"])
    out = tmp_path / "ab.arl"

    concat([a, b], out)

    with File(a) as fa, File(b) as fb, File(out) as merged:
        for src in (fa, fb):
            for time in src.times:
                for level, var in ((0, "PRSS"), (1, "TEMP")):
                    np.testing.assert_array_equal(
                        merged[time][(level, var)].read(),
                        src[time][(level, var)].read(),
                    )


def test_concat_returns_open_file(tmp_path):
    a = write_arl(tmp_path / "a.arl", ["2024-01-01 00:00"])
    b = write_arl(tmp_path / "b.arl", ["2024-01-01 06:00"])
    out = tmp_path / "ab.arl"

    result = concat([a, b], out)
    try:
        assert isinstance(result, File)
        assert result.path == out
        assert len(result.times) == 2
    finally:
        result.close()


def test_concat_single_source_acts_as_copy(tmp_path):
    a = write_arl(tmp_path / "a.arl", ["2024-01-01 00:00", "2024-01-01 06:00"])
    out = tmp_path / "copy.arl"

    concat([a], out)

    assert out.read_bytes() == a.read_bytes()


def test_concat_rejects_grid_mismatch(tmp_path):
    a = write_arl(tmp_path / "a.arl", ["2024-01-01 00:00"], grid=make_test_grid(20, 20))
    b = write_arl(tmp_path / "b.arl", ["2024-01-01 06:00"], grid=make_test_grid(25, 25))
    out = tmp_path / "out.arl"

    with pytest.raises(ValueError, match="Grid mismatch"):
        concat([a, b], out)


def test_concat_rejects_vertical_axis_mismatch(tmp_path):
    a = write_arl(
        tmp_path / "a.arl",
        ["2024-01-01 00:00"],
        vertical_axis=PressureAxis(levels=[0.0, 1000.0]),
    )
    b = write_arl(
        tmp_path / "b.arl",
        ["2024-01-01 06:00"],
        vertical_axis=PressureAxis(levels=[0.0, 925.0]),
    )
    out = tmp_path / "out.arl"

    with pytest.raises(ValueError, match="Vertical axis mismatch"):
        concat([a, b], out)


def test_concat_rejects_duplicate_times(tmp_path):
    a = write_arl(tmp_path / "a.arl", ["2024-01-01 00:00", "2024-01-01 06:00"])
    b = write_arl(tmp_path / "b.arl", ["2024-01-01 06:00", "2024-01-01 12:00"])
    out = tmp_path / "out.arl"

    with pytest.raises(ValueError, match="appears in both"):
        concat([a, b], out)


def test_concat_rejects_bare_string_source(tmp_path):
    a = write_arl(tmp_path / "a.arl", ["2024-01-01 00:00"])
    out = tmp_path / "out.arl"

    with pytest.raises(TypeError, match="iterable of paths"):
        concat(str(a), out)


def test_concat_rejects_empty_sources(tmp_path):
    with pytest.raises(ValueError, match="at least one source"):
        concat([], tmp_path / "out.arl")


def test_concat_rejects_destination_in_sources(tmp_path):
    a = write_arl(tmp_path / "a.arl", ["2024-01-01 00:00"])

    with pytest.raises(ValueError, match="also one of the sources"):
        concat([a], a)


def test_concat_rejects_empty_source_file(tmp_path):
    a = write_arl(tmp_path / "a.arl", ["2024-01-01 00:00"])
    empty = tmp_path / "empty.arl"
    empty.write_bytes(b"")
    out = tmp_path / "out.arl"

    with pytest.raises(ValueError, match="contains no records"):
        concat([a, empty], out)


# --- concat_by_time -------------------------------------------------------


def populate_hrrr_dir(directory):
    """Write 6-hourly single-time ARL files spanning two days into `directory`."""
    directory.mkdir(parents=True, exist_ok=True)
    times = [
        "2024-01-01 00:00",
        "2024-01-01 06:00",
        "2024-01-01 12:00",
        "2024-01-01 18:00",
        "2024-01-02 00:00",
        "2024-01-02 06:00",
    ]
    for time in times:
        stamp = pd.Timestamp(time)
        write_arl(directory / f"{stamp:%Y%m%d_%H}_hrrr", [time])
    return times


def test_concat_by_time_groups_by_day(tmp_path):
    src = tmp_path / "hrrr"
    out = tmp_path / "daily"
    populate_hrrr_dir(src)

    written = concat_by_time(
        src, out, freq="1D", pattern="*_hrrr", template="{time:%Y%m%d}_hrrr"
    )

    assert [p.name for p in written] == ["20240101_hrrr", "20240102_hrrr"]
    with File(written[0]) as day1, File(written[1]) as day2:
        assert day1.times == [
            pd.Timestamp("2024-01-01 00:00"),
            pd.Timestamp("2024-01-01 06:00"),
            pd.Timestamp("2024-01-01 12:00"),
            pd.Timestamp("2024-01-01 18:00"),
        ]
        assert day2.times == [
            pd.Timestamp("2024-01-02 00:00"),
            pd.Timestamp("2024-01-02 06:00"),
        ]


def test_concat_by_time_respects_freq(tmp_path):
    src = tmp_path / "hrrr"
    out = tmp_path / "halfday"
    populate_hrrr_dir(src)

    written = concat_by_time(
        src, out, freq="12h", pattern="*_hrrr", template="{time:%Y%m%d_%H}_hrrr"
    )

    # Day 1 splits into 00Z (00,06) and 12Z (12,18); day 2 has just 00Z (00,06).
    assert [p.name for p in written] == [
        "20240101_00_hrrr",
        "20240101_12_hrrr",
        "20240102_00_hrrr",
    ]


def test_concat_by_time_time_range_filters(tmp_path):
    src = tmp_path / "hrrr"
    out = tmp_path / "daily"
    populate_hrrr_dir(src)

    written = concat_by_time(
        src,
        out,
        freq="1D",
        pattern="*_hrrr",
        template="{time:%Y%m%d}_hrrr",
        time_range=("2024-01-02 00:00", "2024-01-02 23:00"),
    )

    assert [p.name for p in written] == ["20240102_hrrr"]
    with File(written[0]) as day2:
        assert len(day2.times) == 2


def test_concat_by_time_creates_output_dir(tmp_path):
    src = tmp_path / "hrrr"
    out = tmp_path / "nested" / "daily"
    populate_hrrr_dir(src)

    written = concat_by_time(
        src, out, freq="1D", pattern="*_hrrr", template="{time:%Y%m%d}_hrrr"
    )

    assert out.is_dir()
    assert all(p.exists() for p in written)


def test_concat_by_time_raises_on_no_matches(tmp_path):
    src = tmp_path / "hrrr"
    populate_hrrr_dir(src)

    with pytest.raises(ValueError, match="No files matched"):
        concat_by_time(src, tmp_path / "out", pattern="*.nope")


def test_concat_by_time_rejects_non_arl_file(tmp_path):
    src = tmp_path / "hrrr"
    populate_hrrr_dir(src)
    (src / "junk_hrrr").write_bytes(b"not an arl file")

    with pytest.raises(ValueError, match="Could not read an ARL index record"):
        concat_by_time(
            src, tmp_path / "out", pattern="*_hrrr", template="{time:%Y%m%d}_hrrr"
        )
