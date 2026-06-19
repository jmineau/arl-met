# Changelog

All notable changes to arl-met are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `concat(sources, destination, *, sort=True)`: concatenate multiple ARL files into one via a byte-level append. Inputs are first scanned to ensure they share a grid and vertical axis and do not repeat valid times, then joined in valid-time order (`sort=True`, the default) or in the order given. New public export `arlmet.concat`
- `concat_by_time(directory, output_directory, freq="1D", *, pattern, time_range, template, sort)`: batch form of `concat` that groups a directory of ARL files into time-binned chunks — e.g. 6-hourly HRRR files into daily files. Each file is assigned to a bin by its first valid time, read from the index record rather than parsed from the filename. New public export `arlmet.concat_by_time`

### Changed

- Internal layout: the file operations now live under an `arlmet.ops` subpackage. `arlmet.subset` → `arlmet.ops.subset`, `arlmet.sampling` → `arlmet.ops.sample` (module renamed for consistency with `concat`/`subset`), and `arlmet.concat` → `arlmet.ops.concat`. The public top-level API is unchanged — `arlmet.extract_subset`, `arlmet.sample_points`, `arlmet.concat`, and `arlmet.concat_by_time` all still import from `arlmet` directly. Only code importing the submodules by path is affected

## [0.1.0a4] - 2026-06-03

### Added

- `sample_points()` now accepts paths in addition to open `File` objects, for a single source or a sequence (mix of paths and files allowed). Paths are opened in read mode and closed automatically; caller-opened files are left open. This removes the need for the caller to manage file lifecycles (e.g. `contextlib.ExitStack`) when sampling across multiple files
- `File.extract_subset(destination, ...)` method, mirroring the module-level `extract_subset()` for callers that already hold an open file
- `arlmet.vertical.hypsometric_z_agl()`: a pure-NumPy hypsometric height helper, usable without xarray. The xarray `z_agl()` helper and point sampling now share this single implementation
- User guides for point sampling (`docs/sampling.rst`) and vertical coordinates (`docs/vertical.rst`); the vertical helpers `pressure()`, `z_agl()`, and `z_msl()` are now listed in the API reference
- Polymorphic `VerticalAxis` subclasses: `SigmaAxis`, `PressureAxis`, `TerrainAxis`, `HybridAxis`. Each subclass owns its `to_pressure()` and `to_height_agl()` methods, mirroring HYSPLIT's `prfcom` flag dispatcher. Construct via `VerticalAxis.from_flag(flag, levels, offset=)` or use a subclass directly
- New public exports: `arlmet.SigmaAxis`, `arlmet.PressureAxis`, `arlmet.TerrainAxis`, `arlmet.HybridAxis`

### Changed

- `sample_points(source, ...)`: `source` may now be a path, an open `File`, or a sequence of either (previously a single `File` or sequence of `File`)
- `extract_subset()` now returns the newly written subset opened as a read-mode `File` (previously returned `None`), so it can be chained into analysis (`with extract_subset(...) as sub: ...`). Callers that only need the file on disk can ignore the return value
- **`VerticalAxis` is now an abstract base class.** Direct `VerticalAxis(flag=..., levels=...)` construction no longer works; use `VerticalAxis.from_flag(...)` or a subclass constructor
- Vertical coordinate dispatch in `pressure()`, `z_agl()`, `z_msl()`, and `sample_points()` now delegates to subclass methods instead of if/elif flag chains
- `z_agl()` for pressure-level (flag=2) files now requires `HGTS` in the dataset, matching HYSPLIT `PRFPRS`. The previous hypsometric fallback for flag=2 files without HGTS has been removed
- `z_agl()` for sigma/hybrid (flag=1/4) files now always uses hypsometric integration from `PRSS` and `TEMP`, matching HYSPLIT `PRFSIG`/`PRFECM`. These files never contain `HGTS` in practice

### Removed

- `VerticalAxis.sigma_to_pressure()` — replaced by `SigmaAxis.to_pressure()` and `HybridAxis.to_pressure()`
- `VerticalAxis.FLAGS` dict and `VerticalAxis.coord_system` property — `coord_system` is now a class attribute on each subclass
- Public `arlmet.sampling.sample_points_from_file`; single-file sampling is covered by `File.sample_points()` (method) and `sample_points()` (module function). The internal workhorse is now the private `_sample_points_from_file`

## [0.1.0a3] - 2026-06-01

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

### Fixed

- `extract_subset()` copied diff (`DIF*`) records verbatim while repacking the parent with a new exponent and initial value tuned to the cropped window, leaving the diff aligned to the old quantization grid. This produced a small systematic value offset across the cropped subset (~3% of packing precision) that compounded in downstream STILT integrations. Diff records are now recomputed against the newly packed parent via `create_datarecord(diff=...)`, matching reference HYSPLIT behavior (#15, closes #14)

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
