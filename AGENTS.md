# AGENTS.md

Orientation for contributors and coding agents working on arl-met.
Contribution mechanics are in [CONTRIBUTING.md](CONTRIBUTING.md); reference
material (HYSPLIT routines arl-met mirrors, the import graph, the vertical-axis
conversion table, benchmarks) is in the [developer notes](docs/development.rst).

> **Keep this file current.** If you change the module layout, the build/test
> commands, or learn a new invariant or gotcha, update the matching section in
> the same change. A section that no longer matches the code is worse than no
> section: fix it or delete it.
>
> Personal or machine-specific notes (local paths, cluster setup) belong in an
> untracked file, not here. Anything matching `*.local.md` is gitignored for
> this (e.g. `CLAUDE.local.md`); add your tool's local files to `.gitignore` if
> they are not covered. Some agents stop reading `AGENTS.md` once a local
> instruction file exists, so import or reference it from yours.

## What this is

arl-met (import `arlmet`) reads, writes, crops, samples, and joins NOAA ARL
packed meteorology files, the input format of HYSPLIT and STILT. The public API
is listed in `docs/api.rst` and shown in the README and the user guides. Its main
downstream user is PYSTILT, which downloads met with `arlmet.archives`, crops it
with `extract_subset` before HYSPLIT runs, and reads headers through `File`, so
the speed and memory use of those paths matter.

## Layout

```
src/arlmet/
  file.py        File: file handle, scan of index records, check(), factories
  recordset.py   RecordSet: the records of one valid time
  record.py      DataRecord: one packed 2D field (one variable, level, time)
  collection.py  RecordCollection protocol, VariableView, VariableAccessor
  header.py      Header: the 50-byte record header codec
  index.py       IndexRecord, VarInfo, LvlInfo: the index record codec
  grid.py        Projection, Grid, GridWindow: horizontal geometry only
  vertical.py    VerticalAxis (ABC): SigmaAxis, PressureAxis, TerrainAxis, HybridAxis
  packing.py     pack(), unpack(), calculate_checksum()
  _pack.c        C extension arlmet._pack: the packing encoder (pack_core)
  exceptions.py  ARLFormatError, ARLFormatWarning
  _io.py         atomic_output(), reject_same_file(): safe output paths
  _time.py       ensure_timestamp()
  archives.py    Archive and its subclasses, ARCHIVES, get_archive(): NOAA downloads
  ops/           subset.py (extract_subset), sample.py, concat.py (concat, concat_by_time)
  xarray/        dataset.py (open/write_dataset), the engine="arl" backend, the ds.arl
                 accessor, the arl_grid coordinate codec, pressure/z_agl/z_msl
tests/           one test_<area>.py per area, plus test_memory.py and test_subset_slow.py
docs/            Sphinx site (PyData theme): user guides, API reference, developer notes
```

`ops/` and `xarray/` sit on top of `file.py`. `File.extract_subset`,
`File.sample_points` and `File.to_dataset` import them inside the method, which
keeps the import graph one-way; keep it that way. `archives.py` imports `fsspec`
and `ops.subset` only inside the methods that use them, so `import
arlmet.archives` works without the `archives` extra.

## ARL format rules

The format facts that cause silent bugs when ignored.

- **Records.** A file is a flat sequence of fixed-size records of
  `Header.N_BYTES + nx * ny` bytes (50 + nx*ny), one 2D field each. Each time
  step starts with an index record (`INDX`) that describes the grid, the
  vertical axis and each level's variables, and must fit in one record. A crop
  too small for its index record is refused before anything is written
  (`validate_subset_record_length` in `ops/subset.py`).
- **Projection fields.** Of the 12 projection values in the index record, only
  the longitudes (`pole_lon`, `tangent_lon`, `sync_lon`) are wrapped to
  [-180, 180] on read. HYSPLIT reads all 12 raw (`metset.f`); `sync_x`/`sync_y`
  are grid indices and can exceed 180, so never wrap them.
- **Grids over 999 cells.** The thousands of `nx`/`ny` are a letter in the header
  (`A` = 1000, ...; `header.py`). `Header.grid` holds the thousands and the index
  record's fixed part only the remainder, but `IndexRecord.nx`/`ny` are always the
  full sizes; `IndexRecord.__post_init__` rejects a `header.grid` that does not
  match, and `index_length` is computed, never stored.
- **Packing.** Values are packed as unsigned bytes of differences from the
  previous value (HYSPLIT's `pakout`/`pakinp`). `pack` and `unpack` are in
  `packing.py`; the encoder loop is the C extension `_pack.c`. Do not change
  either without a pack/unpack round-trip test.
- **Checksums.** The index record stores a checksum per data record: the sum of
  its packed bytes, less 255 each time it reaches 256, so 1 to 255 unless every
  byte is 0. A stored 0 means none was recorded, and `verify_checksum()` returns
  True for it: HYSPLIT's writer leaves 0 for records it did not pack, and its
  reader never compares checksums. NOAA's GDAS and NAM12 files store 0 for the
  6-hour-forecast fields at 00, 06, 12 and 18 UTC, and their data are good.
- **`DIF*` records** are in-stream corrections. On read, a `DIF*` record belongs
  to the nearest preceding non-`DIF` record of the same time step, and
  `DataRecord.read()` returns parent + diff. Implement only the documented subset
  of DIFF behavior; do not assume every `DIF*` name corrects the record before it.
- **Malformed content raises `ARLFormatError`** (a `ValueError` subclass) from
  `File._scan`: unparseable headers or index records, a size that is not a whole
  number of records, an index record declaring more records than remain,
  inconsistent index metadata. Bad arguments stay plain `ValueError`/`TypeError`.
  A time step repeated byte for byte (some NOAA HRRR files) is skipped with an
  `ARLFormatWarning`; a repeat that differs raises. `File.check()` lists damage
  that opens cleanly but stops HYSPLIT (short or unevenly spaced time steps, bad
  data record headers); it reads headers only, not values.
- **The vertical flag is the type.** `VerticalAxis.from_flag(flag, levels,
  offset=)` builds the subclass, mirroring HYSPLIT's `prfcom`: 1 `SigmaAxis`,
  2 `PressureAxis`, 3 `TerrainAxis`, 4 `HybridAxis`; any other flag (including
  5, WRF) raises `ValueError`. Sigma and hybrid files never have `HGTS`, so their
  heights are hypsometric (`PRSS` + `TEMP`); pressure files always have `HGTS`,
  and their heights come from it. Never add a fallback across flags ("use `HGTS`
  if present", or a hypsometric fallback for pressure files). Each subclass uses
  only the keyword-only inputs it needs and raises `ValueError` naming a missing one.
- **Rotated grids** (`orientation != 0`) raise `NotImplementedError` in `Projection`.

## Design invariants

1. **`grid.py` is horizontal only.** Vertical geometry is in `vertical.py`. Wind
   rotation (`Grid.meridian_convergence`, `Grid.rotate_winds`) belongs to the
   projection and stays in `grid.py`.
2. **`level` is the stored vertical dimension.** `open_dataset` returns lazy
   NumPy-backed arrays (`ArlVariableArray`; dask only through `xr.open_dataset(...,
   chunks=)`) and adds no derived 3D coordinates (`z_agl`, `z_msl` are helpers).
3. **No live Python objects in xarray attrs.** Attrs are plain values (the grid is
   the scalar `arl_grid` coordinate's attrs), so datasets survive netCDF.
4. **`extract_subset` crops before unpacking.** With a bbox it passes a
   `GridWindow` to `DataRecord.read(window=...)`, which decodes only that tile.
   Without a horizontal crop it does not unpack: `_copy_subset_records` copies
   each record's bytes (DIF records included), renumbers the header's level, and
   rebuilds the index records. Keep both paths writing the same index records.
5. **Windowing matches HYSPLIT's `xtrct_grid`.** On lat/lon grids a cell is kept
   when its center is inside the bbox, edges included, and `west > east` crosses
   the dateline. On projected grids the SW and NE corners are mapped to grid
   indices and rounded like Fortran `NINT`. `test_grid.py` holds the reference.
   A bbox is `(west, south, east, north)` in degrees (EPSG:4326), never projected.
6. **`write_dataset` handles the common case only:** the flat Dataset that
   `open_dataset` returns (surface variables without `level`, upper-air variables
   on one `level` coordinate, `forecast_hour(time)` as the index-record forecast).
   DIF records are written from `attrs["diff"] = "DIF..."` on the parent.
   Per-variable forecast hours and other irregular layouts use `File`.
7. **No backward compatibility.** When a metadata representation changes, change
   every read and write path together; no code that accepts old attrs or formats.
8. **Writers stream one time step at a time.** `DataRecord._flush()` drops its
   buffers after writing (the record becomes mode "r"), and the writers call
   `File.flush()` after each time step. `File`, `RecordSet` and `DataRecord` form
   reference cycles, so any buffer a written `File` keeps lives until the cycle
   collector runs. `tests/test_memory.py` guards peak and retained memory.
9. **`DataRecord.read()` never caches; `.data` is the cache.** Lazy
    `open_dataset` arrays keep their records alive, so a caching `read()` would
    hold the whole file.
10. **Handles come from xarray's `CachingFileManager`**, so files reopen after
    `close()` or xarray's LRU cache closing them (`File.handle` reacquires). It
    gets bare "r"/"w" through `_open_binary`: xarray turns only "w" into "a" on
    reopen, so "wb" would truncate the output each time.
11. **Outputs appear only when complete.** `extract_subset`, `write_dataset` and
    archive downloads write a hidden `.<name>.<random>.partial` file and rename
    it on success (`_io.atomic_output`); `extract_subset` and `write_dataset`
    refuse an output that is their input (`_io.reject_same_file`).
12. **Metadata types are immutable values.** `Projection`, `Grid`, `GridWindow`,
    `Header` and `IndexRecord` are frozen dataclasses; `VerticalAxis` blocks
    assignment and stores `levels` read-only. Hashing uses the defining fields,
    and `cached_property` (`Grid.crs`, `Grid.origin`) is safe only because of
    this. Build a new value (`dataclasses.replace`, `Grid.subset`); never mutate.
13. **The public API is explicit.** `arlmet/__init__.py`, `ops/__init__.py` and
    `xarray/__init__.py` re-export it, and the modules define `__all__` (all but
    `ops/subset.py` and `ops/sample.py`). Options after the obvious leading
    arguments are keyword-only. `unpack` keeps six positional arguments (the
    wheel smoke test in `publish.yml` calls it that way); only `window` is
    keyword-only.
14. **Names.** Outputs are `dest` (a file) or `dest_dir` (a directory).
    `source` means only the 4-character ARL source ID (`File.source`,
    `ds.attrs["source"]`); the download classes are archives and their server is
    the `mirror`. `open_dataset`/`write_dataset` keep xarray's `filename_or_obj`.
15. **Containers follow `File.times`.** `File[int]`, `iter(File)` and
    `File.records` use sorted valid-time order, not on-disk order. `key in rs`
    accepts the `(level, variable)` keys `rs[key]` takes, and a bare variable
    name meaning "at any level".

## Commands

The environment is uv-managed (`uv sync`, which also builds `_pack`); every task
is a `just` recipe, and CI runs the same recipes (`just` lists them).

| Task | Command |
|---|---|
| Set up / refresh `.venv` | `uv sync` |
| Everything CI checks (lint, types, docstrings, tests) | `just quality-check` |
| Tests (no network/slow) | `just test` (parallel; extra args go to pytest) |
| Tests that download from NOAA S3 | `just test-network` |
| Lint / fix and format / type check (pyrefly, strict) | `just lint` / `just format` / `just type-check` |
| Docs (runs every example; warnings are errors) / live preview | `just build-docs` / `just docs-serve` |
| All pre-commit hooks / build and check dists | `just pre-commit` / `just dist` |
| Draft changelog entries | `just changelog` |

After editing dependencies, run `uv lock` online (`--offline` churns unrelated
pins) and commit `uv.lock` with them. Python 3.11+; runtime dependencies are
numpy, pandas, pyproj, typing_extensions and xarray (fsspec, s3fs for `[archives]`).

## Docs

`docs/api.rst` lists the API by topic. Each class gets a page with tables of its
attributes and methods, and each member a page of its own
(`docs/_templates/autosummary/`, `docs/_ext/api_pages.py`); a property's page is
where its See Also and Examples go, so give every public attribute and property a
docstring. A new public function gets a user guide (a gerund page such as
`cropping.rst`, listed in `docs/user-guides.rst`), its API entry, and a README
line in the same change. Examples run when the docs build, so they show real
output and a broken one fails the build:

- **A figure in a docstring:** a `.. plot::` directive in the Examples section
  runs its code (doctest `>>>` lines work) and shows the figure under it.
- **Code in an `.rst` page:** a `.. ipython:: python` block runs and shows each
  line with its output; `@savefig name.png` on the line above a plotting call
  saves its figure. A block that raises or warns fails the build (`:okexcept:` or
  `:okwarning:` allows one meant to). All `.rst` pages share one session, so each
  page imports and defines what it uses.
- `just build-docs` runs everything, as CI does; `just docs-serve` reruns a page's
  blocks when that page is rebuilt. Notebooks and Markdown code cells are off
  (`copier update --data docs_notebooks=true` turns them on).
- **Data:** the docs build on GitHub Actions, so an example uses synthetic data,
  a small file in the repository, or a public download, never a machine-specific
  path. Seed random numbers so figures don't change between builds.
- **ARL files:** the repository has no sample ARL file. An example builds a small
  one with `arlmet.write_dataset`, as the tests do. Archive downloads are
  gigabytes, so an example shows the call as a plain `code-block` and never runs
  it. If an example needs an extra, add `arlmet[extra]` to the `dev` group.

## Tests

- Tests build small synthetic ARL files whose grid must hold the index record
  (`50 + nx*ny >= len(index.tobytes())`); 20x20 fits a few variables and levels.
- `just test` runs `-m "not network and not slow"`, as CI does. A bare `pytest`
  also runs the network tests, which download from NOAA S3. CI runs the fast
  network tests on every run and the slow ones (`test_subset_slow.py`: parity of
  `extract_subset` with HYSPLIT's `xtrct_grid` on real files) only when started by
  hand.
- Markers are strict and warnings are errors (assert expected ones with `pytest.warns`).
- `test_memory.py` bounds writer memory with `tracemalloc`; keep its grids about
  200x200 so array buffers dominate. It can fail when pytest's temp directory is
  on a filesystem with a large block size (Python sizes file buffers from
  `st_blksize`); point `TMPDIR` at a local disk.

## Commits, changelog, releases

- Conventional Commits (`fix(record): ...`, `docs: ...`, `!` if breaking).
  User-visible changes go under `## [Unreleased]` in `CHANGELOG.md`.
- The version comes from git tags (setuptools-scm); there is no version string to
  bump, and a release does not change `uv.lock`. Steps are in
  [RELEASING.md](RELEASING.md). **Do not cut a release, create a tag, or push
  unless the maintainer asks.**
- Docs are versioned on GitHub Pages from the gh-pages branch (`dev/`, one folder
  per release, `stable/`), which `.github/scripts/docs_versions.py` maintains.
  Never edit gh-pages by hand.
- Tooling comes from [jmineau/python-template](https://github.com/jmineau/python-template)
  via copier; `copier update` pulls in its changes (never edit `.copier-answers.yml`).

## Gotchas

- `ruff format` skips `README.md` and `AGENTS.md` (hand-aligned examples).
- `docs/_autosummary/` is generated; `just build-docs` deletes and rebuilds it.
- uv rebuilds the editable install when `_pack.c`, `setup.py`, `pyproject.toml`
  or the git state change (`[tool.uv] cache-keys`), so `__version__` follows new
  commits and tags.
