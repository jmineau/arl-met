# Changelog

All notable changes to arl-met are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Support for Python 3.13 and 3.14: tested in CI and built as wheels for Linux, macOS, and Windows

### Changed

- Wheels are built with cibuildwheel 4 and smoke-tested (the C packer must import and round-trip) before publishing. The wheel build also runs on pull requests that touch the build configuration, so a broken build is caught before a release is tagged
- Building from source needs setuptools >= 77 (for the SPDX `license` field)
- `CONTRIBUTING.md` setup steps now work: they used a `dev` extra that does not exist (it is a dependency group)
- `extract_subset()` and `write_dataset()` write to a temporary file next to the output and rename it into place when complete, so an interrupted run never leaves a truncated file under the final name (which a later run could mistake for a finished output, e.g. a cached crop)
- `write_dataset()` renumbers upper-air levels `1..N` in `level` coordinate order, so a Dataset holding a subset of a file's levels (from `open_dataset(levels=...)` or `ds.sel(level=...)`) is written as a compact file, as `extract_subset()` does. An explicit `vertical_axis=` must have one level per `level` coordinate value plus the surface
- `open_dataset()` records the input path in `ds.encoding["source"]`, as xarray's own backends do

### Removed

- **Breaking:** Python 3.10 support. arlmet now requires Python 3.11 or newer (3.10 reaches end-of-life in October 2026, and current NumPy no longer supports it)

### Fixed

- `extract_subset(p, p)` and `write_dataset(open_dataset(p), p)` truncated the input file before reading it, destroying it on multi-time files. Both now raise `ValueError` when the output is the input (including through a symlink or hard link)
- `open_dataset()` followed by `write_dataset()` failed on files with a variable stored on only some levels: `open_dataset()` fills those levels with NaN (its docstring wrongly said it never NaN-pads), and `write_dataset()` rejected any NaN. An all-NaN slice is now written as no record; a partly-NaN slice still raises
- `open_dataset(levels=[0, 2])` (and `ds.sel(level=...)`) rebuilt the vertical axis with 0.0 filled in for every level left out, so `ds.arl.vertical_axis` was wrong (`[0, 0, 2000]`) and `write_dataset()` wrote bogus levels. The axis is now the surface plus the Dataset's levels (`[0, 2000]`)
- README and the writing guide built a vertical axis with `arlmet.VerticalAxis(flag=2, ...)`, which raises `TypeError` since `VerticalAxis` became abstract; they now use `arlmet.PressureAxis(...)`
- `MeteorologySource.fetch(..., bbox=...)` / `levels=` staged the full uncropped download in the system temp directory (~3 GB per HRRR file), which could fill small `/tmp` partitions, and `extract_subset` wrote the crop directly under the cached name, so an interrupted fetch left a truncated file that later calls reused as a cache hit. The download and the crop are now both staged as hidden `.<name>.<random>.partial` files in `local_dir`, the crop is moved into place with `os.replace` only once complete, and both temp files are removed on success or failure
- Plain (uncropped) downloads used a fixed `<name>.tmp` temp path, so concurrent fetches of the same file (e.g. parallel SLURM jobs) wrote to the same temp file. Each download now gets its own unique hidden `.partial` file, moved into place with `os.replace`
- Cached file names rounded the bbox to two decimals, so bboxes differing only in the third decimal shared a cache file and `fetch` returned the wrong crop. Values with more than two decimals are now kept in full (`.crop_-111.925_...`); bboxes with at most two decimals keep exactly the same name as before (`.crop_-112.00_40.25_...`), so existing caches still match
- `start_date` was declared on every source but never checked, so a range before the archive began produced keys for files that don't exist and failed at download time. `keys_for_range()` and `fetch()` now raise `ValueError` if the range begins before the source's `start_date`

## [0.1.0a9] - 2026-09-29

### Added

- `MeteorologySource.fetch(..., levels=...)`: keep only the given vertical levels of each downloaded file, with or without `bbox`. The levels are part of the cached file's name (`.levels_0-19`), so files cropped to different levels are cached separately

## [0.1.0a8] - 2026-09-22

### Added

- `sample_points(..., earth_relative=True)` (also on `File.sample_points`): rotate sampled `UWND`/`VWND` and `U10M`/`V10M` from grid-relative to east/north using the meridian convergence of the file's grid, so sampled winds can be compared with observations. Winds in ARL files on projected grids are stored grid-relative, as HYSPLIT expects. Both components of a pair must be requested; lat/lon grids are unaffected
- `Grid.meridian_convergence(lon, lat)` and `Grid.rotate_winds(u, v, lon, lat)`: the angle from grid north to true north, and the rotation itself, for any projected grid. On the HRRR grid this reproduces the rotation in NOAA's HRRR FAQ (`0.622515 * (lon + 97.5)` degrees)

## [0.1.0a7] - 2026-09-22

### Added

- Zenodo citation metadata (`CITATION.cff`, `.zenodo.json`) so releases are archived on Zenodo and mint a citable DOI

## [0.1.0a6] - 2026-09-21

### Added

- `File.flush()`: write pending record sets to disk and release their in-memory data. When writing many time steps with the low-level `File` API, call it after filling each one so memory stays bounded by a single time step

### Changed

- `DataRecord.read()` no longer caches the full field on the record: each call reads from disk and returns a new array. Use `DataRecord.data` for a cached copy. Lazy `open_dataset()` Datasets therefore re-read from disk on each access, like other xarray backends; call `.load()` to keep the data in memory
- `extract_subset()` and `concat()` docs no longer say the returned `File` may be ignored; close it (`extract_subset(...).close()`) if you only need the file on disk, since an unclosed `File` keeps its file handle open until it is garbage collected
- Development tooling: the pre-commit ruff hook now matches the ruff pinned in `uv.lock` (0.15.10), so `ruff format --check` and the hook agree; docstring coverage is back to 100% (the Code Quality workflow requires 95%)

### Fixed

- Writers held every record in memory until the file was closed, keeping float32, packed, and byte copies of each field (~6x the output size). Worse, `File`, `RecordSet`, and `DataRecord` reference each other, so those buffers outlived the call until Python's cycle collector ran, and memory piled up across repeated calls (e.g. `extract_subset()` in a loop). Records now drop their buffers once written, and `extract_subset()` and `write_dataset()` write one time step at a time. Peak memory for a 12-time-step `extract_subset()` fell from ~6x to ~0.6x the output size, and memory still held after it returns fell from ~6x to ~0.05x
- Closing a write-mode `File` twice truncated the written file to 0 bytes: the second close reopened it with mode `"wb"`. Writers now reopen in append mode
- A `File` whose handle was closed by xarray's global file cache (which keeps at most `file_cache_maxsize`, default 128, files open) raised `ValueError: seek of closed file` on the next read or write. The handle is now reopened when needed
- Reading a lazy `open_dataset()` Dataset (opened without `bbox`) cached every full field it touched on the source records, plus a second copy for each DIF record, so it gradually held the whole file in memory. `write_dataset(open_dataset(path), ...)` peaked at ~5x the output size; it now peaks at ~0.6x

## [0.1.0a5] - 2026-06-19

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
