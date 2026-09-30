"""Tests for the xarray backend entrypoint (``engine="arl"``)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import arlmet
from arlmet import File
from arlmet.grid import Grid, Projection
from arlmet.vertical import PressureAxis
from arlmet.xarray._entrypoint import ARLBackendEntrypoint


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


def test_engine_passes_bbox_levels_and_drop_variables(tmp_path):
    path = write_arl(tmp_path / "met.arl")
    kwargs = {"bbox": (22.0, -8.0, 24.0, -6.0), "levels": [0, 2]}

    via_xarray = xr.open_dataset(path, engine="arl", drop_variables="PRSS", **kwargs)
    direct = arlmet.open_dataset(path, drop_variables=["PRSS"], **kwargs)

    xr.testing.assert_identical(via_xarray, direct)
    assert "PRSS" not in via_xarray


def test_engine_supports_dask_chunks(tmp_path):
    pytest.importorskip("dask")
    path = write_arl(tmp_path / "met.arl")

    ds = xr.open_dataset(path, engine="arl", chunks={"time": 1})

    assert ds["TEMP"].chunks is not None
    np.testing.assert_allclose(
        ds["TEMP"].values, arlmet.open_dataset(path)["TEMP"].values
    )


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
