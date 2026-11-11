"""Tests for the xarray backend entrypoint (``engine="arl"``)."""

import copy
import pickle

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import arlmet
from arlmet import File
from arlmet._grid import Grid, Projection
from arlmet._xarray._entrypoint import ARLBackendEntrypoint
from arlmet.vertical import PressureAxis


def write_arl(path):
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
    grid = Grid(projection=projection, nx=20, ny=20)
    base = np.arange(400, dtype=np.float32).reshape(20, 20)
    with File(
        path,
        mode="w",
        source="TEST",
        grid=grid,
        vertical_axis=PressureAxis(levels=[0.0, 1000.0, 900.0]),
    ) as arl:
        for i, time in enumerate(["2024-01-01 00:00", "2024-01-01 01:00"]):
            rs = arl.create_recordset(pd.Timestamp(time), forecast=0)
            rs.create_datarecord("PRSS", level=0, forecast=0, data=1000.0 + base + i)
            for level in (1, 2):
                rs.create_datarecord(
                    "TEMP", level=level, forecast=0, data=280.0 + base + level
                )
    return path


def test_engine_matches_open_dataset(tmp_path):
    path = write_arl(tmp_path / "met.arl")

    via_xarray = xr.open_dataset(path, engine="arl")
    direct = arlmet.open_dataset(path)

    xr.testing.assert_identical(via_xarray, direct)
    assert via_xarray.arl.grid == direct.arl.grid


@pytest.mark.parametrize(
    "indexers",
    [
        {},
        {"time": 1},
        {"time": 0, "level": 1, "lat": 3, "lon": 4},
        {"lat": slice(2, 9, 3), "lon": slice(None, None, -2)},
        {"level": [1, 0], "lat": [5, 1, 5], "lon": [0, 19]},
        {"time": [1, 0], "lon": slice(4, 8)},
    ],
)
def test_lazy_indexing_matches_the_loaded_array(tmp_path, indexers):
    path = write_arl(tmp_path / "met.arl")
    loaded = arlmet.open_dataset(path)["TEMP"].load()

    lazy = arlmet.open_dataset(path)["TEMP"].isel(indexers)

    xr.testing.assert_identical(lazy.load(), loaded.isel(indexers))


def test_engine_passes_bbox_levels_and_drop_variables(tmp_path):
    path = write_arl(tmp_path / "met.arl")
    kwargs = {"bbox": (22.0, -8.0, 24.0, -6.0), "levels": [0, 2]}

    via_xarray = xr.open_dataset(path, engine="arl", drop_variables="PRSS", **kwargs)
    direct = arlmet.open_dataset(path, drop_variables=["PRSS"], **kwargs)

    xr.testing.assert_identical(via_xarray, direct)
    assert "PRSS" not in via_xarray


def test_engine_passes_bbox_crs(tmp_path):
    path = write_arl(tmp_path / "met.arl")
    kwargs = {"bbox": (300_000, 9_220_000, 700_000, 9_900_000), "bbox_crs": 32735}

    via_xarray = xr.open_dataset(path, engine="arl", **kwargs)
    direct = arlmet.open_dataset(path, **kwargs)

    xr.testing.assert_identical(via_xarray, direct)
    assert (direct.sizes["lon"], direct.sizes["lat"]) == (5, 7)


def test_engine_supports_dask_chunks(tmp_path):
    pytest.importorskip("dask")
    path = write_arl(tmp_path / "met.arl")

    ds = xr.open_dataset(path, engine="arl", chunks={"time": 1})

    assert ds["TEMP"].chunks is not None
    np.testing.assert_allclose(
        ds["TEMP"].values, arlmet.open_dataset(path)["TEMP"].values
    )


@pytest.mark.parametrize("opener", [arlmet.open_dataset, "engine"])
def test_dataset_copies_and_pickles_after_a_read(tmp_path, opener):
    path = write_arl(tmp_path / "met.arl")
    ds = xr.open_dataset(path, engine="arl") if opener == "engine" else opener(path)
    expected = ds["TEMP"].isel(time=0).values  # the read leaves a handle open

    for copied in (
        ds.copy(deep=True),
        copy.deepcopy(ds),
        pickle.loads(pickle.dumps(ds)),
    ):
        np.testing.assert_array_equal(copied["TEMP"].isel(time=0).values, expected)


def test_engine_is_detected_without_engine_argument(tmp_path):
    path = write_arl(tmp_path / "met.arl")

    assert ARLBackendEntrypoint().guess_can_open(path)
    assert "arl" in xr.backends.list_engines()
    assert "TEMP" in xr.open_dataset(path)


def test_guess_can_open_rejects_other_files(tmp_path):
    other = tmp_path / "notes.txt"
    other.write_text("not an ARL file at all")
    backend = ARLBackendEntrypoint()

    assert not backend.guess_can_open(other)
    assert not backend.guess_can_open(tmp_path / "missing.arl")
    assert not backend.guess_can_open(tmp_path)  # a directory
    assert not backend.guess_can_open(b"INDX")  # not a path


def test_engine_rejects_file_objects(tmp_path):
    path = write_arl(tmp_path / "met.arl")

    with open(path, "rb") as handle, pytest.raises(TypeError, match="by path"):
        xr.open_dataset(handle, engine="arl")
