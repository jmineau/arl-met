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

    with File(path, mode="w", source="TEST", grid=grid, vertical_axis=vertical_axis) as arl:
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


def test_open_dataset_bbox_and_levels_reads_only_selected_subset(tmp_path):
    source = tmp_path / "source.arl"
    write_subset_source(source)

    bbox = (22.0, -8.0, 24.0, -6.0)
    ds = open_dataset(
        source,
        squeeze=False,
        bbox=bbox,
        levels=[0, 2],
        drop_variables=["TEMP"],
    )

    with File(source) as original:
        source_window = original.grid.window_from_bbox(bbox)
        source_prss = original[0][(0, "PRSS")].read(window=source_window)
        source_uwnd = original[0][(2, "UWND")].read(window=source_window)

    assert set(ds.data_vars) == {"PRSS", "UWND"}
    assert ds.sizes["time"] == 2
    assert ds.sizes["level"] == 2
    assert ds.sizes["lat"] == 3
    assert ds.sizes["lon"] == 3
    # Level coord holds native heights (hPa for flag=2): indices 0,2 → [0.0, 2000.0]
    np.testing.assert_array_equal(ds.coords["level"].values, [0.0, 2000.0])
    assert ds.attrs["arl_nx"] == 3
    assert ds.attrs["arl_ny"] == 3
    assert ds.attrs["vertical_axis"].levels.tolist() == [0.0, 1000.0, 2000.0]
    np.testing.assert_allclose(np.asarray(ds["PRSS"].isel(time=0, level=0)), source_prss)
    np.testing.assert_allclose(np.asarray(ds["UWND"].isel(time=0, level=1)), source_uwnd)
