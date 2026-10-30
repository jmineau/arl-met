"""Slow parity checks for extract_subset against xtrct_grid on real files."""

from __future__ import annotations

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from urllib.request import urlopen

import numpy as np
import pytest

from arlmet import File, extract_subset, open_dataset
from arlmet.archives import GDASArchive, HRRRArchive

_XTRCT_GRID_URL = (
    "https://raw.githubusercontent.com/uataq/stilt/main/bin/linux_x64/xtrct_grid"
)

_HRRR_TEST_TIME = "2024-07-18 03:00"
_GDAS_TEST_TIME = "2024-07-18 00:00"
_SLV_BBOX = (-114.0, 39.5, -110.5, 42.0)
_WEST_NA_BBOX = (-140.0, 20.0, -85.0, 60.0)
_XTRCT_LEVELS = list(range(18))

_SUBSET_CASES = [
    pytest.param(HRRRArchive(), _HRRR_TEST_TIME, _SLV_BBOX, id="hrrr"),
    pytest.param(GDASArchive(), _GDAS_TEST_TIME, _WEST_NA_BBOX, id="gdas1"),
]


@pytest.fixture(scope="session")
def xtrct_grid_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    binary = tmp_path_factory.mktemp("xtrct-grid") / "xtrct_grid"
    if not binary.exists():
        with urlopen(_XTRCT_GRID_URL) as response:
            binary.write_bytes(response.read())
        binary.chmod(binary.stat().st_mode | 0o111)
    return binary


def _run_xtrct_grid(
    binary: Path, source: Path, bbox: tuple[float, float, float, float], output: Path
) -> None:
    stdin = "\n".join(
        (
            f"{source.parent}/",
            source.name,
            f"{bbox[1]} {bbox[0]}",
            f"{bbox[3]} {bbox[2]}",
            "18",
            "",
        )
    )
    with TemporaryDirectory() as workdir:
        proc = subprocess.run(
            [str(binary)],
            input=stdin,
            text=True,
            cwd=workdir,
            capture_output=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"xtrct_grid failed with exit code {proc.returncode}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
        Path(workdir, "extract.bin").replace(output)


def _max_finite_abs_diff(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if not mask.any():
        return 0.0
    return float(np.max(np.abs(left[mask] - right[mask])))


def _xtrct_cells(source_path: Path, bbox) -> dict[str, slice]:
    """
    Where xtrct_grid's crop lies in extract_subset's, as ``isel`` slices.

    On a lat/lon grid the two crops are the same. On a projected grid
    xtrct_grid keeps the window of the box's south-west and north-east corners
    (rounded like NINT), and extract_subset the window of the whole box, which
    holds it (#131).
    """
    with File(source_path) as arl_file:
        grid = arl_file.grid
    if grid.is_latlon:
        return {}
    window = grid.window_from_bbox(bbox)
    x, y = grid.fractional_indices([bbox[0], bbox[2]], [bbox[1], bbox[3]])
    xs = sorted(int(np.floor(v + 0.5)) for v in x)
    ys = sorted(int(np.floor(v + 0.5)) for v in y)
    assert window.x_start <= xs[0] and xs[1] < window.x_stop
    assert window.y_start <= ys[0] and ys[1] < window.y_stop
    return {
        "x": slice(xs[0] - window.x_start, xs[1] + 1 - window.x_start),
        "y": slice(ys[0] - window.y_start, ys[1] + 1 - window.y_start),
    }


def _assert_subset_parity(
    source_ds, xtrct_ds, subset_ds, xtrct_precisions, subset_precisions, xtrct_cells
) -> None:
    assert set(source_ds.data_vars) == set(xtrct_ds.data_vars)
    assert set(source_ds.data_vars) == set(subset_ds.data_vars)

    # time/level: both outputs must match source exactly.
    for coord in ("time", "level"):
        if coord in source_ds.coords:
            expected = source_ds[coord].values
            np.testing.assert_array_equal(
                xtrct_ds[coord].values, expected, err_msg=coord
            )
            np.testing.assert_array_equal(
                subset_ds[coord].values, expected, err_msg=coord
            )

    # lat/lon: the crop's cells sit where the source puts them. source_ds is
    # the source read with the same bbox, so this compares the crop's stored
    # grid with the source's. xtrct_grid writes its crop's corner latitude and
    # longitude to 0.01 degree, so on a projected grid its cells move by up to
    # that much; it must still have chosen the same cells.
    projected = "x" in source_ds.dims
    xtrct_source = source_ds.isel(xtrct_cells)
    for coord in ("lat", "lon"):
        if coord in source_ds.coords:
            np.testing.assert_allclose(
                subset_ds[coord].values,
                source_ds[coord].values,
                rtol=0,
                # #86: the crop writes HRRR's sync_lon 237.281 as -122.72
                atol=2e-3 if projected else 1e-9,
                err_msg=f"{coord}: extract_subset moved the cells",
            )
            np.testing.assert_allclose(
                xtrct_ds[coord].values,
                xtrct_source[coord].values,
                rtol=0,
                atol=0.006 if projected else 1e-9,
                err_msg=f"{coord}: extract_subset and xtrct_grid chose different cells",
            )

    for name in sorted(source_ds.data_vars):
        xtrct_error = _max_finite_abs_diff(
            xtrct_source[name].values, xtrct_ds[name].values
        )
        subset_error = _max_finite_abs_diff(
            xtrct_source[name].values, subset_ds.isel(xtrct_cells)[name].values
        )
        precision_budget = max(
            xtrct_precisions.get(name, 0.0),
            subset_precisions.get(name, 0.0),
        )
        assert subset_error <= xtrct_error + precision_budget + 1e-6, (
            f"{name}: extract_subset max error {subset_error} exceeds "
            f"xtrct_grid max error {xtrct_error} plus precision budget {precision_budget}"
        )


def _max_variable_precision(path: Path) -> dict[str, float]:
    precisions: dict[str, float] = {}
    with File(path) as arl_file:
        for time in arl_file.times:
            for record in arl_file[time].records:
                precisions[record.variable] = max(
                    precisions.get(record.variable, 0.0),
                    record.header.precision,
                )
    return precisions


@pytest.mark.network
@pytest.mark.slow
@pytest.mark.parametrize("source,time,bbox", _SUBSET_CASES)
def test_extract_subset_matches_xtrct_grid_error_envelope(
    tmp_path: Path,
    record_property: pytest.RecordProperty,
    xtrct_grid_binary: Path,
    source,
    time,
    bbox,
):
    [source_path] = source.fetch(time, time, dest_dir=tmp_path)
    xtrct_path = tmp_path / "xtrct_grid.arl"
    subset_path = tmp_path / "extract_subset.arl"

    start = perf_counter()
    _run_xtrct_grid(xtrct_grid_binary, source_path, bbox, xtrct_path)
    xtrct_seconds = perf_counter() - start

    start = perf_counter()
    extract_subset(source_path, subset_path, bbox=bbox, levels=_XTRCT_LEVELS)
    subset_seconds = perf_counter() - start

    record_property("xtrct_grid_seconds", xtrct_seconds)
    record_property("extract_subset_seconds", subset_seconds)
    record_property("extract_subset_speedup", xtrct_seconds / subset_seconds)

    source_ds = open_dataset(source_path, bbox=bbox, levels=_XTRCT_LEVELS).load()
    xtrct_ds = open_dataset(xtrct_path).load()
    subset_ds = open_dataset(subset_path).load()
    xtrct_precisions = _max_variable_precision(xtrct_path)
    subset_precisions = _max_variable_precision(subset_path)
    _assert_subset_parity(
        source_ds,
        xtrct_ds,
        subset_ds,
        xtrct_precisions,
        subset_precisions,
        _xtrct_cells(source_path, bbox),
    )
