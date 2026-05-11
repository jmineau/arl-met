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
from arlmet.sources import GDASSource, HRRRSource

_XTRCT_GRID_URL = (
    "https://raw.githubusercontent.com/uataq/stilt/main/bin/linux_x64/xtrct_grid"
)

_HRRR_TEST_TIME = "2024-07-18 03:00"
_GDAS_TEST_TIME = "2024-07-18 00:00"
_SLV_BBOX = (-114.0, 39.5, -110.5, 42.0)
_WEST_NA_BBOX = (-140.0, 20.0, -85.0, 60.0)
_XTRCT_LEVELS = list(range(18))

_SUBSET_CASES = [
    pytest.param(HRRRSource(), _HRRR_TEST_TIME, _SLV_BBOX, id="hrrr"),
    pytest.param(GDASSource(), _GDAS_TEST_TIME, _WEST_NA_BBOX, id="gdas1"),
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


def _assert_subset_parity(
    source_ds, xtrct_ds, subset_ds, xtrct_precisions, subset_precisions
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

    # lat/lon: cropping updates the sync/reference point in the header, so
    # coordinates recomputed from the cropped file differ from those computed
    # from the source header (this is expected for projected grids).
    # What matters is that xtrct_grid and extract_subset chose the same cells,
    # so compare the two cropped outputs against each other.
    for coord in ("lat", "lon"):
        if coord in xtrct_ds.coords:
            np.testing.assert_allclose(
                subset_ds[coord].values,
                xtrct_ds[coord].values,
                rtol=1e-7,
                atol=0,
                err_msg=f"{coord}: extract_subset and xtrct_grid disagree on grid coordinates",
            )

    for name in sorted(source_ds.data_vars):
        xtrct_error = _max_finite_abs_diff(
            source_ds[name].values, xtrct_ds[name].values
        )
        subset_error = _max_finite_abs_diff(
            source_ds[name].values, subset_ds[name].values
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
    [source_path] = source.fetch(time, time, local_dir=tmp_path)
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
        source_ds, xtrct_ds, subset_ds, xtrct_precisions, subset_precisions
    )
