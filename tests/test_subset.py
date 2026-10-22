"""Tests for direct ARL subset extraction."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from arlmet import File, extract_subset, open_dataset
from arlmet.exceptions import ARLFormatError
from arlmet.grid import Grid, Projection
from arlmet.indexrecord import IndexRecord
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
        rs.create_datarecord("TPP1", level=0, forecast=0, data=data, diff="DIFR")

    bbox = (22.0, -8.0, 33.0, 3.0)
    extract_subset(source, destination, bbox=bbox)

    with File(source) as original, File(destination) as subset:
        source_window = original.grid.window_from_bbox(bbox)
        source_record = original[time0][(0, "TPP1")]
        subset_record = subset[time0][(0, "TPP1")]

        assert subset_record.diffs
        assert subset_record.diffs[0].variable == "DIFR"

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
        rs.create_datarecord("TPP1", level=0, forecast=0, data=data, diff="DIFR")

    bbox = (25.0, -5.0, 70.0, 40.0)
    extract_subset(source, destination, bbox=bbox)

    with File(source) as original, File(destination) as subset:
        source_window = original.grid.window_from_bbox(bbox)
        source_record = original[time0][(0, "TPP1")]
        subset_record = subset[time0][(0, "TPP1")]

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


def test_extract_subset_projected_crop_keeps_source_cell_positions(tmp_path):
    """
    Every cell of a projected crop sits where the source puts it (#78).

    The source's sync point is the corner of a Lambert grid (as in a NOAA HRRR
    file); a crop's own corner longitude would be rounded to 0.01 degree in
    the index record and move the crop by up to 0.15 cells.
    """
    projection = Projection(
        pole_lat=90.0,
        pole_lon=0.0,
        tangent_lat=38.5,
        tangent_lon=-97.5,
        grid_size=3.0,
        orientation=0.0,
        cone_angle=38.5,
        sync_x=1.0,
        sync_y=1.0,
        sync_lat=40.0,
        sync_lon=-112.6,
    )
    grid = Grid(projection=projection, nx=60, ny=50)
    lat = grid.calculate_coords()["lat"][1]
    source = tmp_path / "source.arl"
    with File(
        source,
        mode="w",
        source="TEST",
        grid=grid,
        vertical_axis=SigmaAxis(levels=[1.0, 0.99]),
    ) as arl:
        rs = arl.create_recordset(pd.Timestamp("2024-07-18 00:00"))
        rs.create_datarecord("PRSS", level=0, forecast=0, data=800.0 + lat)
        rs.create_datarecord("TEMP", level=1, forecast=0, data=250.0 + lat)

    outer, inner = (-112.4, 40.1, -111.6, 40.8), (-112.3, 40.2, -111.8, 40.65)
    crop = extract_subset(source, tmp_path / "crop.arl", bbox=outer)
    crop2 = extract_subset(crop, tmp_path / "crop2.arl", bbox=inner)

    with File(source) as src:
        expected = src.grid.calculate_coords()
        for path, bbox in ((crop, outer), (crop2, inner)):
            win = src.grid.window_from_bbox(bbox)
            assert win.x_start > 0 and win.y_start > 0
            with File(path) as out:
                p = out.grid.projection
                assert (p.sync_lat, p.sync_lon) == (40.0, -112.6)
                assert (p.sync_x, p.sync_y) == (1.0 - win.x_start, 1.0 - win.y_start)
                coords = out.grid.calculate_coords()
                for name in ("lon", "lat"):
                    np.testing.assert_allclose(
                        coords[name][1],
                        expected[name][1][win.y_slice, win.x_slice],
                        rtol=0,
                        atol=1e-9,
                    )
                # the same cells' values (each crop packs them again)
                np.testing.assert_allclose(
                    out[0][(0, "PRSS")].read(),
                    src[0][(0, "PRSS")].read()[win.y_slice, win.x_slice],
                    atol=2 * out[0][(0, "PRSS")].header.precision,
                )

    ds = open_dataset(crop)
    ds_bbox = open_dataset(source, bbox=outer)
    np.testing.assert_allclose(ds["lon"].values, ds_bbox["lon"].values, atol=1e-9)
    np.testing.assert_allclose(ds["lat"].values, ds_bbox["lat"].values, atol=1e-9)


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
        {"levels": [3, 0, 1]},
        {"levels": [0, 1, 3], "variables": ["WWND"]},
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
                assert len(record.diffs) == len(original.diffs)
                # Copied verbatim except the header's level field, so the
                # values are exactly the source's, DIF correction included.
                pairs = [(record, original)]
                if record.diffs:
                    assert record.diffs[0].variable == original.diffs[0].variable
                    pairs.append((record.diffs[0], original.diffs[0]))
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
                if record.diffs:
                    atol = 2 * max(
                        record.diffs[0].header.precision,
                        other.diffs[0].header.precision,
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
        extract_subset(source, destination, levels=[0, 1])
    assert not destination.exists()


def test_extract_subset_validation_counts_diff_records(tmp_path):
    # A 16x8 crop has 178-byte records: room for an index listing TPP1
    # (174 bytes) but not TPP1 plus its DIFR record (182 bytes).
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
        rs.create_datarecord("TPP1", level=0, forecast=0, data=data, diff="DIFR")

    with pytest.raises(ValueError, match="too small to encode an ARL index record"):
        extract_subset(source, destination, bbox=(22.0, -8.0, 37.0, -1.0))
    assert not destination.exists()


def _stored_longitudes(path) -> tuple[float, float, float]:
    """The first index record's raw ``sync_lon``, ``pole_lon`` and spacing."""
    fixed = Path(path).read_bytes()[50:158].decode("ascii")
    values = [float(fixed[9 + 7 * i : 16 + 7 * i]) for i in range(12)]
    return values[10], values[1], values[3]


def _hysplit_x(path, lon: float) -> float:
    """One-based x of ``lon`` as HYSPLIT's ``gblset.f`` and ``gbl2xy.f`` place it."""
    sync_lon, pole_lon, dlon = _stored_longitudes(path)
    prime = pole_lon >= 0.0 and sync_lon < 0.0
    if not prime:
        if lon < 0.0:
            lon += 360.0
        if lon > 360.0:
            lon -= 360.0
    lon -= sync_lon
    if lon < 0.0:
        lon += 360.0
    return 1.0 + lon / dlon


def test_rewritten_latlon_files_keep_hysplit_longitudes(tmp_path):
    """
    Crops, copies and rewrites of a 0-360 lat/lon file stay 0-360 (#82).

    A rewritten sync longitude of -125 for 235 made HYSPLIT place the grid
    360 degrees away, since it measures from a negative sync longitude only
    when the pole longitude is not negative.
    """
    from arlmet import write_dataset

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
    source = tmp_path / "global.arl"
    lon360 = np.tile(np.arange(360.0, dtype=np.float32), (181, 1))
    with File(
        source,
        mode="w",
        source="GDAS",
        grid=grid,
        vertical_axis=PressureAxis(levels=[0.0, 1000.0]),
    ) as arl:
        rs = arl.create_recordset(pd.Timestamp("2024-07-18"))
        rs.create_datarecord("SHGT", level=0, forecast=0, data=lon360)
        rs.create_datarecord("TEMP", level=1, forecast=0, data=lon360)

    crop = extract_subset(source, tmp_path / "crop.arl", bbox=(-125, 30, -100, 50))
    outputs = {
        "crop": crop,
        "crop of crop": extract_subset(
            crop, tmp_path / "crop2.arl", bbox=(-120, 35, -105, 45)
        ),
        "levels-only copy": extract_subset(crop, tmp_path / "copy.arl", levels=[0, 1]),
        "write_dataset": tmp_path / "rewrite.arl",
    }
    write_dataset(open_dataset(crop), outputs["write_dataset"])

    salt_lake_city = -111.89
    for name, path in outputs.items():
        sync_lon, pole_lon, _ = _stored_longitudes(path)
        assert 180.0 < sync_lon < pole_lon < 360.0, name
        with File(path) as arl:
            x = float(arl.grid.fractional_indices(salt_lake_city, 40.0)[0])
            temp = arl[0][(1, "TEMP")].read()
        assert _hysplit_x(path, salt_lake_city) == pytest.approx(x + 1.0), name
        # each cell holds its 0-360 longitude
        assert temp[0, round(x)] == pytest.approx(248.0), name


@pytest.mark.parametrize(
    ("sync_lon", "bbox"),
    [(0.0, (-10, 40, 10, 60)), (-180.0, (170, -10, -170, 10))],
)
def test_extract_subset_across_the_seam_keeps_the_box(tmp_path, sync_lon, bbox):
    """
    A crop across a global grid's seam holds only the box's columns (#83).

    It used to keep all 360 degrees. The crop is written as HYSPLIT's
    xtrct_grid writes its header (sync_lon 350 or 170, pole_lon 20 cells on),
    with each cell's value read from the source's column for it.
    """
    grid = Grid(
        projection=Projection(
            pole_lat=90.0,
            pole_lon=sync_lon + 359.0,
            tangent_lat=1.0,
            tangent_lon=1.0,
            grid_size=0.0,
            orientation=0.0,
            cone_angle=0.0,
            sync_x=1.0,
            sync_y=1.0,
            sync_lat=-90.0,
            sync_lon=sync_lon,
        ),
        nx=360,
        ny=181,
    )
    lons = sync_lon + np.arange(360.0)
    wave = (200.0 + 100.0 * np.sin(np.radians(lons))).astype(np.float32)
    source = tmp_path / "global.arl"
    with File(
        source,
        mode="w",
        source="GDAS",
        grid=grid,
        vertical_axis=PressureAxis(levels=[0.0, 1000.0]),
    ) as arl:
        rs = arl.create_recordset(pd.Timestamp("2024-07-18"))
        rs.create_datarecord("SHGT", level=0, forecast=0, data=np.tile(wave, (181, 1)))
        rs.create_datarecord("TEMP", level=1, forecast=0, data=np.tile(wave, (181, 1)))

    crop = extract_subset(source, tmp_path / "crop.arl", bbox=bbox)
    with File(crop) as arl:
        assert (arl.grid.nx, arl.grid.ny) == (21, 21)
        crop_lons = arl.grid.calculate_coords()["lon"][1]
        temp = arl[0][(1, "TEMP")].read()
    np.testing.assert_allclose(crop_lons, bbox[0] + np.arange(21.0))
    want = 200.0 + 100.0 * np.sin(np.radians(crop_lons))
    np.testing.assert_allclose(temp[0], want, atol=0.05)
    sync, pole, _ = _stored_longitudes(crop)
    assert (sync, pole) == ((350.0, 370.0) if sync_lon == 0.0 else (170.0, 190.0))
    for lon in (bbox[0], bbox[2]):
        x = float(open_dataset(crop).arl.grid.fractional_indices(lon, 50.0)[0])
        assert _hysplit_x(crop, lon) == pytest.approx(x + 1.0)

    ds = open_dataset(source, bbox=bbox)
    assert ds.sizes["lon"] == 21
    np.testing.assert_allclose(ds["TEMP"].isel(time=0, level=0, lat=0), want, atol=0.05)
