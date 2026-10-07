"""Tests for direct ARL subset extraction."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from arlmet import File, extract_subset, open_dataset
from arlmet.exceptions import ARLFormatError
from arlmet.grid import Grid, Projection
from arlmet.index import IndexRecord
from arlmet.ops import subset as subset_module
from arlmet.vertical import PressureAxis, SigmaAxis


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


def write_subset_source(path):
    grid = make_test_grid()
    vertical_axis = PressureAxis(levels=[0.0, 1000.0, 2000.0])
    base = np.arange(grid.nx * grid.ny, dtype=np.float32).reshape(grid.ny, grid.nx)

    time0 = pd.Timestamp("2024-07-18 00:00")
    time1 = pd.Timestamp("2024-07-18 03:00")

    with File(
        path, mode="w", source="TEST", grid=grid, vertical_axis=vertical_axis
    ) as arl:
        rs0 = arl.create_recordset(time0)
        rs0.create_datarecord("PRSS", level=0, forecast=0, data=1000.0 + base)
        rs0.create_datarecord("TEMP", level=1, forecast=0, data=280.0 + base)
        rs0.create_datarecord("UWND", level=2, forecast=0, data=-5.0 + base)

        rs1 = arl.create_recordset(time1)
        rs1.create_datarecord("PRSS", level=0, forecast=3, data=1001.0 + base)
        rs1.create_datarecord("TEMP", level=1, forecast=3, data=281.0 + base)
        rs1.create_datarecord("UWND", level=2, forecast=3, data=-4.0 + base)


def test_extract_subset_crops_bbox_and_compacts_levels(tmp_path):
    source = tmp_path / "source.arl"
    destination = tmp_path / "subset.arl"
    write_subset_source(source)

    extract_subset(
        source,
        destination,
        bbox=(22.0, -8.0, 33.0, 3.0),
        levels=[0, 2],
        variables=["PRSS", "UWND"],
    )

    with File(source) as original, File(destination) as subset:
        assert subset.grid.nx == 12
        assert subset.grid.ny == 12
        assert subset.vertical_axis.levels.tolist() == [0.0, 2000.0]
        assert subset.times == original.times

        source_window = original.grid.window_from_bbox((22.0, -8.0, 33.0, 3.0))
        source_prss = original[0][(0, "PRSS")].read(window=source_window)
        source_uwnd = original[0][(2, "UWND")].read(window=source_window)

        subset_prss = subset[0][(0, "PRSS")].read()
        subset_uwnd = subset[0][(1, "UWND")].read()

        np.testing.assert_allclose(
            subset_prss, source_prss, atol=subset[0][(0, "PRSS")].header.precision
        )
        np.testing.assert_allclose(
            subset_uwnd, source_uwnd, atol=subset[0][(1, "UWND")].header.precision
        )


def test_extract_subset_returns_output_path(tmp_path):
    source = tmp_path / "source.arl"
    destination = tmp_path / "subset.arl"
    write_subset_source(source)

    result = extract_subset(
        path=source,
        dest=str(destination),
        bbox=(22.0, -8.0, 33.0, 3.0),
        levels=[0, 2],
        variables=["PRSS", "UWND"],
    )

    assert isinstance(result, Path)
    assert result == destination
    with File(result) as subset:
        assert subset.grid.nx == 12
        assert subset.grid.ny == 12


def test_file_extract_subset_method_matches_module_function(tmp_path):
    source = tmp_path / "source.arl"
    destination = tmp_path / "subset.arl"
    write_subset_source(source)

    with File(source) as met:
        result = met.extract_subset(
            dest=destination,
            bbox=(22.0, -8.0, 33.0, 3.0),
            levels=[0, 2],
            variables=["PRSS", "UWND"],
        )
        assert result == destination
    with File(source) as met, File(result) as subset:
        assert subset.grid.nx == 12
        assert subset.vertical_axis.levels.tolist() == [0.0, 2000.0]
        assert subset.times == met.times
        ds = subset.to_dataset()
        assert set(ds.data_vars) >= {"PRSS", "UWND"}


def test_extract_subset_rejects_out_of_bounds_levels(tmp_path):
    source = tmp_path / "source.arl"
    destination = tmp_path / "subset.arl"
    write_subset_source(source)

    with pytest.raises(ValueError, match="levels"):
        extract_subset(source, destination, levels=[99])


def test_extract_subset_rejects_non_intersecting_bbox(tmp_path):
    source = tmp_path / "source.arl"
    destination = tmp_path / "subset.arl"
    write_subset_source(source)

    with pytest.raises(ValueError, match="bbox"):
        extract_subset(source, destination, bbox=(200.0, 50.0, 201.0, 51.0))


def test_extract_subset_rejects_subset_too_small_for_index_record(tmp_path):
    source = tmp_path / "source.arl"
    destination = tmp_path / "subset.arl"
    write_subset_source(source)

    with pytest.raises(ValueError, match="too small to encode an ARL index record"):
        extract_subset(
            source,
            destination,
            bbox=(22.0, -8.0, 24.0, -6.0),
            levels=[0, 2],
            variables=["PRSS", "UWND"],
        )


def test_extract_subset_allows_mixed_record_forecasts(tmp_path):
    source = tmp_path / "mixed_forecast_source.arl"
    destination = tmp_path / "mixed_forecast_subset.arl"
    grid = make_test_grid()
    vertical_axis = PressureAxis(levels=[0.0, 1000.0])
    data = np.ones((grid.ny, grid.nx), dtype=np.float32)
    time0 = pd.Timestamp("2025-09-01 00:00")

    with File(
        source, mode="w", source="TEST", grid=grid, vertical_axis=vertical_axis
    ) as arl:
        rs = arl.create_recordset(time0)
        rs.create_datarecord("PRSS", level=0, forecast=0, data=data)
        rs.create_datarecord("TEMP", level=1, forecast=3, data=data)

    extract_subset(source, destination, variables=["PRSS", "TEMP"])

    with File(destination) as subset:
        assert subset[time0].forecast == 0
        assert subset[time0][(0, "PRSS")].forecast == 0
        assert subset[time0][(1, "TEMP")].forecast == 3


def test_extract_subset_preserves_diff_records(tmp_path):
    source = tmp_path / "diff_source.arl"
    destination = tmp_path / "diff_subset.arl"
    grid = make_test_grid()
    vertical_axis = PressureAxis(levels=[1000.0])
    time0 = pd.Timestamp("2024-07-18 00:00")
    data = (
        0.123
        + np.arange(grid.nx * grid.ny, dtype=np.float32).reshape(grid.ny, grid.nx)
        * 0.0073
    )

    with File(
        source, mode="w", source="TEST", grid=grid, vertical_axis=vertical_axis
    ) as arl:
        rs = arl.create_recordset(time0)
        rs.create_datarecord("WWND", level=0, forecast=0, data=data, diff="DIFW")

    bbox = (22.0, -8.0, 33.0, 3.0)
    extract_subset(source, destination, bbox=bbox)

    with File(source) as original, File(destination) as subset:
        source_window = original.grid.window_from_bbox(bbox)
        source_record = original[time0][(0, "WWND")]
        subset_record = subset[time0][(0, "WWND")]

        assert subset_record.diff is not None
        assert subset_record.diff.variable == "DIFW"

        np.testing.assert_allclose(
            subset_record.read(),
            source_record.read(window=source_window),
            atol=subset_record.header.precision,
        )


def test_extract_subset_recomputes_diff_records_no_systematic_bias(tmp_path):
    """
    Cropping a diff-encoded record must not introduce a systematic value bias.

    Regression test for the bug where ``extract_subset`` copied diff records
    verbatim while repacking the parent with a new exponent + initial_value,
    leaving the diff aligned with the old quantization grid. The result was a
    small but non-zero-mean offset across the entire cropped grid that
    compounded in downstream STILT trajectory integrations.

    The fixture is engineered so the full grid's data range (driven by large
    outside-window values) yields a parent exponent that differs from the
    cropped subset's exponent — which is exactly the regime where the old
    bug manifested. The diff record carries the high-frequency content that
    only round-trips correctly if recomputed against the newly packed parent.
    """
    source = tmp_path / "diff_source.arl"
    destination = tmp_path / "diff_subset.arl"
    grid = make_test_grid(nx=60, ny=60)
    vertical_axis = PressureAxis(levels=[1000.0])
    time0 = pd.Timestamp("2024-07-18 00:00")

    rng = np.random.default_rng(42)
    data = (100.0 + 50.0 * rng.standard_normal((grid.ny, grid.nx))).astype(np.float32)
    # Cropped window is approximately cells [5..50, 5..50].  Put small,
    # smoothly-varying values there so the cropped exponent is much smaller
    # than the full-grid exponent set by the surrounding noise.
    yy, xx = np.mgrid[5:50, 5:50].astype(np.float32)
    data[5:50, 5:50] = 0.001 * (np.sin(xx) + np.cos(yy)) + 0.0001 * rng.standard_normal(
        (45, 45)
    ).astype(np.float32)

    with File(
        source, mode="w", source="TEST", grid=grid, vertical_axis=vertical_axis
    ) as arl:
        rs = arl.create_recordset(time0)
        rs.create_datarecord("WWND", level=0, forecast=0, data=data, diff="DIFW")

    bbox = (25.0, -5.0, 70.0, 40.0)
    extract_subset(source, destination, bbox=bbox)

    with File(source) as original, File(destination) as subset:
        source_window = original.grid.window_from_bbox(bbox)
        source_record = original[time0][(0, "WWND")]
        subset_record = subset[time0][(0, "WWND")]

        diff_grid = subset_record.read() - source_record.read(window=source_window)
        mean_signed_bias = float(diff_grid.mean())
        precision = subset_record.header.precision

        # Random per-cell quantization noise averages to ~0; a systematic
        # bias on the order of the precision quantum indicates the diff
        # record was not recomputed against the newly packed parent.
        assert abs(mean_signed_bias) < precision * 0.01, (
            f"Systematic bias {mean_signed_bias:+.3e} exceeds 1% of the "
            f"packing precision {precision:.3e}; diff record likely not "
            f"recomputed after repacking the parent."
        )


def test_open_dataset_bbox_and_levels_reads_only_selected_subset(tmp_path):
    source = tmp_path / "source.arl"
    write_subset_source(source)

    bbox = (22.0, -8.0, 24.0, -6.0)
    ds = open_dataset(
        source,
        bbox=bbox,
        levels=[0, 2],
        drop_variables=["TEMP"],
    )

    with File(source) as original:
        source_window = original.grid.window_from_bbox(bbox)
        source_prss = original[0][(0, "PRSS")].read(window=source_window)
        source_uwnd = original[0][(2, "UWND")].read(window=source_window)

    assert set(ds.data_vars) == {"PRSS", "UWND", "forecast_hour"}
    np.testing.assert_array_equal(ds["forecast_hour"].values, [0, 3])
    assert ds.sizes["time"] == 2
    # PRSS is sfc (no level dim); UWND is the only upper var → level size 1
    assert ds.sizes["level"] == 1
    assert ds.sizes["lat"] == 3
    assert ds.sizes["lon"] == 3
    # Level coord is integer ARL index; physical coord (pressure) carries hPa values
    np.testing.assert_array_equal(ds.coords["level"].values, [2])
    np.testing.assert_array_equal(ds.coords["pressure"].values, [2000.0])
    assert ds.arl.grid.nx == 3
    assert ds.arl.grid.ny == 3
    # vertical_axis is the surface plus the loaded levels, compacted: level 1
    # (1000 hPa) was not loaded, so it is absent rather than filled with 0.0
    assert ds.arl.vertical_axis.levels.tolist() == [0.0, 2000.0]
    # PRSS has no level dim; UWND level dim has one element (index 0 → 2000 hPa)
    np.testing.assert_allclose(np.asarray(ds["PRSS"].isel(time=0)), source_prss)
    np.testing.assert_allclose(
        np.asarray(ds["UWND"].isel(time=0, level=0)), source_uwnd
    )


def test_extract_subset_rejects_dest_equal_to_path(tmp_path):
    source = tmp_path / "source.arl"
    write_subset_source(source)
    size = source.stat().st_size

    with pytest.raises(ValueError, match="same file as the input"):
        extract_subset(source, source, levels=[0, 1])
    link = tmp_path / "link.arl"
    try:
        link.symlink_to(source)
    except OSError:  # Windows without symlink privilege
        pass
    else:
        with pytest.raises(ValueError, match="same file as the input"):
            extract_subset(source, link, levels=[0, 1])

    assert source.stat().st_size == size


@pytest.mark.parametrize(
    "bbox",
    [None, (22.0, -8.0, 33.0, 3.0)],
    ids=["byte-copy", "repack"],
)
def test_extract_subset_failure_leaves_no_partial_output(tmp_path, monkeypatch, bbox):
    source = tmp_path / "source.arl"
    write_subset_source(source)
    destination = tmp_path / "subset.arl"
    to_record_bytes = IndexRecord.to_record_bytes
    calls = []

    def boom(self, record_length):
        # Fail on the second time step, after the first has been written.
        calls.append(record_length)
        if len(calls) > 1:
            raise RuntimeError("interrupted")
        return to_record_bytes(self, record_length)

    monkeypatch.setattr(IndexRecord, "to_record_bytes", boom)
    with pytest.raises(RuntimeError, match="interrupted"):
        extract_subset(source, destination, bbox=bbox, levels=[0, 1])

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == [source]


def test_extract_subset_output_honors_umask(tmp_path):
    source = tmp_path / "source.arl"
    write_subset_source(source)
    destination = tmp_path / "subset.arl"
    plain = tmp_path / "plain"
    plain.touch()

    extract_subset(source, destination, levels=[0, 1])

    # Same permissions as any normally created file (not mkstemp's 0600).
    assert destination.stat().st_mode & 0o777 == plain.stat().st_mode & 0o777


# --- Byte-copy fast path (no horizontal crop) ---------------------------------


def write_byte_copy_source(path, *, diff=True):
    """
    Write a source with several times and levels, and mixed forecasts.

    With ``diff`` (the default) it also has DIF records, on odd levels.
    """
    grid = make_test_grid()
    vertical_axis = SigmaAxis(levels=[1.0, 0.98, 0.9, 0.7, 0.5])
    rng = np.random.default_rng(7)

    def field(mean, std):
        return (mean + std * rng.standard_normal((grid.ny, grid.nx))).astype(np.float32)

    with File(
        path, mode="w", source="TEST", grid=grid, vertical_axis=vertical_axis
    ) as arl:
        for step in range(3):
            time = pd.Timestamp("2024-07-18") + pd.Timedelta(hours=3 * step)
            rs = arl.create_recordset(time, forecast=step)
            rs.create_datarecord("PRSS", level=0, forecast=step, data=field(850, 20))
            rs.create_datarecord("T02M", level=0, forecast=-1, data=field(290, 3))
            for level in range(1, 5):
                rs.create_datarecord(
                    "TEMP", level=level, forecast=step, data=field(280, 5)
                )
                rs.create_datarecord(
                    "WWND",
                    level=level,
                    forecast=step,
                    data=field(0, 0.3),
                    diff="DIFW" if diff and level % 2 else None,
                )
                rs.create_datarecord(
                    "UWND", level=level, forecast=step, data=field(5, 10)
                )


def extract_both_paths(monkeypatch, source, tmp_path, **kwargs):
    """Run extract_subset via the byte-copy path and the forced repack path."""
    copied = []
    copy_records = subset_module._copy_subset_records

    def spy(*args, **kw):
        copied.append(True)
        return copy_records(*args, **kw)

    fast = tmp_path / "fast.arl"
    slow = tmp_path / "slow.arl"
    with monkeypatch.context() as m:
        m.setattr(subset_module, "_copy_subset_records", spy)
        extract_subset(source, fast, **kwargs)
    assert copied, "uncropped subset did not take the byte-copy path"
    with monkeypatch.context() as m:
        m.setattr(subset_module, "_is_full_window", lambda grid, window: False)
        extract_subset(source, slow, **kwargs)
    return fast, slow


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"levels": [0, 2, 4]},
        {"levels": [3, 1]},
        {"levels": [1, 3], "variables": ["WWND"]},
        {"variables": ["PRSS", "TEMP", "UWND"]},
        {"levels": [0, 2, 3], "variables": ["T02M", "WWND", "UWND"]},
        {"bbox": (19.5, -10.5, 39.5, 9.5)},
        {"bbox": (0.0, -50.0, 60.0, 50.0), "levels": [0, 1, 4]},
    ],
    ids=[
        "all",
        "levels",
        "unsorted-levels",
        "diff-only",
        "no-diff-vars",
        "levels-and-vars",
        "bbox-exact-grid",
        "bbox-covers-grid",
    ],
)
def test_extract_subset_byte_copy_matches_repack(tmp_path, monkeypatch, kwargs):
    source = tmp_path / "source.arl"
    write_byte_copy_source(source, diff=False)

    fast, slow = extract_both_paths(monkeypatch, source, tmp_path, **kwargs)

    # Re-packing an arlmet-written record reproduces its bytes, so copying
    # them must give exactly the file the repack path writes.
    assert fast.read_bytes() == slow.read_bytes()
    with File(source) as src, File(fast) as out:
        assert out.times == src.times
        assert (out.grid.nx, out.grid.ny) == (src.grid.nx, src.grid.ny)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"levels": [3, 1]},
        {"levels": [1, 3], "variables": ["WWND"]},
        {"levels": [0, 2, 3], "variables": ["T02M", "WWND", "UWND"]},
        {"bbox": (19.5, -10.5, 39.5, 9.5), "levels": [0, 1, 4]},
    ],
    ids=["all", "unsorted-levels", "diff-only", "levels-and-vars", "bbox"],
)
def test_extract_subset_byte_copy_with_diff_records(tmp_path, monkeypatch, kwargs):
    source = tmp_path / "source.arl"
    write_byte_copy_source(source)

    fast, slow = extract_both_paths(monkeypatch, source, tmp_path, **kwargs)

    levels = sorted(kwargs.get("levels", range(5)))
    with File(source) as src, File(fast) as out, File(slow) as repacked:
        assert out.times == src.times == repacked.times
        assert out.vertical_axis == repacked.vertical_axis
        for time in out.times:
            assert [(r.level, r.variable) for r in out[time]] == [
                (r.level, r.variable) for r in repacked[time]
            ]
            for record in out[time]:
                original = src[time][(levels[record.level], record.variable)]
                assert (record.diff is None) == (original.diff is None)
                # Copied verbatim except the header's level field, so the
                # values are exactly the source's, DIF correction included.
                pairs = [(record, original)]
                if record.diff is not None:
                    assert record.diff.variable == original.diff.variable
                    pairs.append((record.diff, original.diff))
                for rec, orig in pairs:
                    assert rec.header.level == record.level
                    assert rec.bytes[:10] == orig.bytes[:10]
                    assert rec.bytes[12:] == orig.bytes[12:]
                    assert rec.verify_checksum()
                np.testing.assert_array_equal(record.read(), original.read())
                # The repack path re-quantizes parent + DIF, so it agrees only
                # to within the DIF precision.
                other = repacked[time][(record.level, record.variable)]
                atol = 0.0
                if record.diff is not None:
                    atol = 2 * max(
                        record.diff.header.precision, other.diff.header.precision
                    )
                np.testing.assert_allclose(record.read(), other.read(), atol=atol)


def test_extract_subset_byte_copy_empty_selection(tmp_path):
    source = tmp_path / "source.arl"
    destination = tmp_path / "subset.arl"
    write_byte_copy_source(source)

    extract_subset(source, destination, variables=["NONE"])

    assert destination.exists()
    assert destination.stat().st_size == 0


def test_extract_subset_byte_copy_rejects_mismatched_record_header(tmp_path):
    source = tmp_path / "source.arl"
    destination = tmp_path / "subset.arl"
    write_byte_copy_source(source)
    with File(source) as src:
        position = src[0][(1, "TEMP")].position
    raw = bytearray(source.read_bytes())
    raw[position + 10 : position + 12] = b" 4"  # claim the wrong level
    source.write_bytes(bytes(raw))

    with pytest.raises(ARLFormatError, match="header mismatch"):
        extract_subset(source, destination, levels=[1])
    assert not destination.exists()


def test_extract_subset_validation_counts_diff_records(tmp_path):
    # A 16x8 crop has 178-byte records: room for an index listing WWND
    # (174 bytes) but not WWND plus its DIFW record (182 bytes).
    source = tmp_path / "source.arl"
    destination = tmp_path / "subset.arl"
    grid = make_test_grid()
    rng = np.random.default_rng(3)
    data = rng.standard_normal((grid.ny, grid.nx)).astype(np.float32)
    with File(
        source,
        mode="w",
        source="TEST",
        grid=grid,
        vertical_axis=PressureAxis(levels=[1000.0]),
    ) as arl:
        rs = arl.create_recordset(pd.Timestamp("2024-07-18"), forecast=0)
        rs.create_datarecord("WWND", level=0, forecast=0, data=data, diff="DIFW")

    with pytest.raises(ValueError, match="too small to encode an ARL index record"):
        extract_subset(source, destination, bbox=(22.0, -8.0, 37.0, -1.0))
    assert not destination.exists()
