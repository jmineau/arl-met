"""Memory use of the ARL writers: peak during a write and what is held after."""

import gc
import tracemalloc

import numpy as np
import pandas as pd

from arlmet import File, extract_subset, open_dataset, write_dataset
from arlmet.grid import Grid, Projection
from arlmet.vertical import PressureAxis

N_TIMES = 12


def write_multistep_source(path, n: int = 200) -> None:
    """
    Write a source with many time steps of fields large enough (n x n) that
    array buffers dominate Python object overhead in the measurements.
    """
    projection = Projection(
        pole_lat=90.0,
        pole_lon=0.0,
        tangent_lat=0.1,
        tangent_lon=0.1,
        grid_size=0.0,
        orientation=0.0,
        cone_angle=0.0,
        sync_x=1.0,
        sync_y=1.0,
        sync_lat=20.0,
        sync_lon=-120.0,
    )
    grid = Grid(projection=projection, nx=n, ny=n)
    vertical_axis = PressureAxis(levels=[1000.0, 900.0, 800.0])
    rng = np.random.default_rng(0)

    def field(mean: float, std: float) -> np.ndarray:
        return rng.normal(mean, std, (n, n)).astype(np.float32)

    with File(
        path, mode="w", source="TEST", grid=grid, vertical_axis=vertical_axis
    ) as arl:
        for hour in range(N_TIMES):
            rs = arl.create_recordset(
                pd.Timestamp("2024-07-18") + pd.Timedelta(hours=hour), forecast=0
            )
            rs.create_datarecord("PRSS", level=0, forecast=0, data=field(900, 5))
            for level in (1, 2):
                rs.create_datarecord(
                    "TEMP", level=level, forecast=0, data=field(280, 5)
                )
                rs.create_datarecord(
                    "WWND", level=level, forecast=0, data=field(0, 1), diff="DIFW"
                )


def traced_peak(func) -> int:
    """Peak traced memory (bytes) allocated while running ``func()``."""
    gc.collect()
    tracemalloc.start()
    try:
        baseline = tracemalloc.get_traced_memory()[0]
        func()
        return tracemalloc.get_traced_memory()[1] - baseline
    finally:
        tracemalloc.stop()


# Writers used to hold every record's float32, packed, and byte copies until
# close, peaking at ~6x the output size. Writing one time step at a time
# bounds the peak by one time step (~6/N_TIMES of the output).


def test_extract_subset_peak_memory_is_one_time_step(tmp_path):
    source = tmp_path / "source.arl"
    destination = tmp_path / "subset.arl"
    write_multistep_source(source)

    peak = traced_peak(lambda: extract_subset(source, destination).close())

    assert peak < destination.stat().st_size


def test_write_dataset_peak_memory_is_one_time_step(tmp_path):
    source = tmp_path / "source.arl"
    destination = tmp_path / "written.arl"
    write_multistep_source(source)
    # Load up front so the measurement covers only the writer's allocations.
    ds = open_dataset(source).load()
    ds["WWND"].attrs["diff"] = "DIFW"

    peak = traced_peak(lambda: write_dataset(ds, destination))

    assert peak < destination.stat().st_size


def test_write_dataset_from_lazy_dataset_does_not_cache_source(tmp_path):
    # A lazy open_dataset() Dataset keeps its source records alive. Reads
    # used to cache each full field on its record (and each DIF record its
    # own copy), so writing it out held the whole source file in memory.
    source = tmp_path / "source.arl"
    destination = tmp_path / "written.arl"
    write_multistep_source(source)
    ds = open_dataset(source)  # lazy: each slice is read from disk while writing
    ds["WWND"].attrs["diff"] = "DIFW"

    peak = traced_peak(lambda: write_dataset(ds, destination))

    assert peak < destination.stat().st_size


def test_extract_subset_releases_buffers_without_gc(tmp_path):
    # File, RecordSet, and DataRecord reference each other, so after
    # extract_subset returns they are freed only by the cycle collector,
    # which can run long after the call. Their record buffers must not
    # wait with them.
    source = tmp_path / "source.arl"
    destination = tmp_path / "subset.arl"
    write_multistep_source(source)

    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    tracemalloc.start()
    try:
        baseline = tracemalloc.get_traced_memory()[0]
        extract_subset(source, destination).close()
        retained = tracemalloc.get_traced_memory()[0] - baseline
    finally:
        tracemalloc.stop()
        if gc_was_enabled:
            gc.enable()

    assert retained < 0.25 * destination.stat().st_size
