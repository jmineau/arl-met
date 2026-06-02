# Changelog

All notable changes to arl-met are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `typing_extensions>=4.0` as a runtime dependency; `override` is now imported directly instead of via a `TYPE_CHECKING` shim in each module
- Module-level docstrings and missing function docstrings across `collection.py`, `recordset.py`, `xarray/_accessor.py`, `xarray/_backend.py`, `xarray/_coords.py`, `xarray/_vertical.py`, and `xarray/dataset.py`; docstring coverage is now 100%
- `__repr__` on `Projection`, `Grid`, `VerticalAxis`, `DataRecord`, `RecordSet`, `File`, and `VariableView` — compact, informative string representations for all core classes
- `VerticalAxis.__len__`: `len(vaxis)` returns the number of levels
- `RecordSet.__contains__`: `"UWND" in rs` tests variable membership by name
- `File.__contains__`: `pd.Timestamp(...) in f` tests whether a time step is present
- `VariableView._lazy_shape`: infers `(time, level, ny, nx)` shape from record metadata without loading data
- pyrefly pre-commit hook (`facebook/pyrefly-pre-commit`) for static type checking

### Changed

- `__version__` simplified to `importlib.metadata.version("arlmet")`; the pyproject.toml fallback path has been removed
- Type checker switched from pyright to pyrefly (`preset = "strict"`); typing improved across all source modules

## [0.1.0a2] - 2026-05-13

### Added

- C extension `_pack` (`_pack.c`) implementing the ARL feedback-loop differential encoder; replaces the pure-Python inner loop in `pack()`
- `TestPackCore` tests validating the C extension byte-for-byte against a Python reference implementation across gradient, signed-value, and running-reconstructed-value cases
- `test_file_copy_is_byte_identical` end-to-end test: writes a synthetic ARL file, reads every record, rewrites to a copy, and asserts binary equality

### Changed

- `pack()` delegates the inner feedback loop to the C extension; `numba` dependency removed
- `setup.py` added alongside `pyproject.toml` to declare the C extension with a dynamic `numpy.get_include()` path
- `pyproject.toml` build-system now requires `numpy>=1.24` so the C extension can be compiled at install time

### Removed

- `numba` dependency and JIT-compiled `_pack_core` — eliminated ~1.4 s per-process warm-up with no change to correctness

### Fixed

- Codecov uploads not running: replaced `!always()` condition with `!cancelled()`
- Coverage XML not generated: added `--cov-report=xml` to pytest command
- Coverage upload missing explicit `files: coverage.xml`
- Test results upload switched from `codecov/test-results-action@v1` to `codecov/codecov-action@v5` with `report_type: test_results` and explicit `files: junit.xml`

## [0.1.0a1] - 2026-05-11

Initial public alpha of `arl-met`, providing the current core package surface for
reading, writing, subsetting, and sampling NOAA ARL meteorology files.

- low-level ARL file reading and writing through `File`, `RecordSet`, and `DataRecord`
- xarray Dataset read/write support for the common flat ARL Dataset contract
- direct subset extraction through `extract_subset()`
- point sampling through `sample_points()`
- NOAA source helpers for common ARL archives
- vertical helper functions `pressure()`, `z_agl()`, and `z_msl()`
- parent-led `DIF*` generation for low-level writes and `write_dataset()`
- switched package versioning to PEP 440 semver-style alpha releases
- tightened dependency metadata with explicit minimum version pins
- `write_dataset()` intentionally targets the common-case Dataset contract
- complex multi-record DIFF chains are not yet tested
- WRF vertical flag 5 is not implemented
