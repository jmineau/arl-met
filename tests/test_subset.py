"""Tests for direct ARL subset extraction."""

import numpy as np
import pandas as pd
import pytest

from arlmet import File, extract_subset, open_dataset
from arlmet.grid import Grid, Projection
from arlmet.vertical import VerticalAxis


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
    vertical_axis = VerticalAxis(flag=2, levels=[0.0, 1000.0, 2000.0])
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
    vertical_axis = VerticalAxis(flag=2, levels=[0.0, 1000.0])
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
    vertical_axis = VerticalAxis(flag=2, levels=[1000.0])
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
    vertical_axis = VerticalAxis(flag=2, levels=[1000.0])
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
    # vertical_axis is reconstructed from the subset — level 1 (1000 hPa) was not
    # loaded so it shows as 0.0 in the reconstructed levels array
    assert ds.arl.vertical_axis.levels.tolist() == [0.0, 0.0, 2000.0]
    # PRSS has no level dim; UWND level dim has one element (index 0 → 2000 hPa)
    np.testing.assert_allclose(np.asarray(ds["PRSS"].isel(time=0)), source_prss)
    np.testing.assert_allclose(
        np.asarray(ds["UWND"].isel(time=0, level=0)), source_uwnd
    )
