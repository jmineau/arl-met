"""Compare extract_subset with HYSPLIT's xtrct_grid on one met file, and write the numbers.

usage: compare.py CASE SOURCE OUT_JSON

CASE is one of tests/test_subset_slow.py's cases (``hrrr``, ``gdas1``): it gives the time and
the box. SOURCE is that case's file, already downloaded (the test fetches it from NOAA). The
crop, the checks and the error budget are the test's own functions, so this records what the
test asserts: for each variable, the largest difference from the source of xtrct_grid's crop
and of extract_subset's, on xtrct_grid's cells, the packing precision that bounds them, and the
largest difference between the two crops.
Also the time each tool took. xtrct_grid is validation/build/xtrct_grid (build.sh) or
$ARLMET_XTRCT_GRID.
"""

import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path

import arlmet
from arlmet import extract_subset, open_dataset

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "test_subset_slow", HERE.parents[1] / "tests" / "test_subset_slow.py"
)
assert _spec is not None and _spec.loader is not None
slow = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(slow)

CASES = {param.id: param.values for param in slow._SUBSET_CASES}


def main(case: str, source: Path, out: Path) -> None:
    archive, valid_time, bbox = CASES[case]
    # ignore_cleanup_errors: on NFS a file xarray still holds open leaves a .nfs file behind.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        xtrct_path = Path(tmp) / "xtrct_grid.arl"
        subset_path = Path(tmp) / "extract_subset.arl"

        start = time.perf_counter()
        slow._run_xtrct_grid(slow._XTRCT_GRID, source, bbox, xtrct_path)
        xtrct_seconds = time.perf_counter() - start

        start = time.perf_counter()
        extract_subset(source, subset_path, bbox=bbox, levels=slow._XTRCT_LEVELS)
        subset_seconds = time.perf_counter() - start

        source_ds = open_dataset(source, bbox=bbox, levels=slow._XTRCT_LEVELS).load()
        xtrct_ds = open_dataset(xtrct_path).load()
        subset_ds = open_dataset(subset_path).load()
        xtrct_precisions = slow._max_variable_precision(xtrct_path)
        subset_precisions = slow._max_variable_precision(subset_path)
        cells = slow._xtrct_cells(source, bbox)
        # The test's assertions; this raises where the test would fail.
        slow._assert_subset_parity(
            source_ds, xtrct_ds, subset_ds, xtrct_precisions, subset_precisions, cells
        )

        xtrct_source = source_ds.isel(cells)
        variables = {}
        for name in sorted(source_ds.data_vars):
            variables[name] = {
                "xtrct_grid_error": slow._max_finite_abs_diff(
                    xtrct_source[name].values, xtrct_ds[name].values
                ),
                "extract_subset_error": slow._max_finite_abs_diff(
                    xtrct_source[name].values, subset_ds.isel(cells)[name].values
                ),
                "crop_difference": slow._max_finite_abs_diff(
                    xtrct_ds[name].values, subset_ds.isel(cells)[name].values
                ),
                "precision": max(
                    xtrct_precisions.get(name, 0.0), subset_precisions.get(name, 0.0)
                ),
            }
        result = {
            "case": case,
            "file": source.name,
            "archive": type(archive).__name__,
            "time": valid_time,
            "bbox": list(bbox),
            "levels": len(slow._XTRCT_LEVELS),
            "arlmet": arlmet.__version__,
            "xtrct_grid": str(slow._XTRCT_GRID),
            "xtrct_grid_cells": {k: [v.start, v.stop] for k, v in cells.items()},
            "extract_subset_shape": {k: int(v) for k, v in subset_ds.sizes.items()},
            "xtrct_grid_shape": {k: int(v) for k, v in xtrct_ds.sizes.items()},
            "xtrct_grid_seconds": xtrct_seconds,
            "extract_subset_seconds": subset_seconds,
            "source_bytes": source.stat().st_size,
            "xtrct_grid_bytes": xtrct_path.stat().st_size,
            "extract_subset_bytes": subset_path.stat().st_size,
            "variables": variables,
        }
        for ds in (source_ds, xtrct_ds, subset_ds):
            ds.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]))
