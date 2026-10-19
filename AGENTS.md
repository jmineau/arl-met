# AGENTS.md

Orientation for contributors and coding agents working on arl-met.
Contribution mechanics are in [CONTRIBUTING.md](CONTRIBUTING.md); reference
material (the full module layout and import graph, the HYSPLIT routines arl-met
mirrors, the vertical-axis conversions, benchmarks) is in the
[developer notes](docs/development.rst).

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
packed meteorology files, HYSPLIT's input format; the API is in `docs/api.rst`,
the README and the user guides. Its main user, PYSTILT, downloads met with
`arlmet.archives`, crops it with `extract_subset` before HYSPLIT runs, and reads
headers through `File`, so the speed and memory use of those paths matter.

## Layout

```
src/arlmet/
  file.py, recordset.py, record.py  File > RecordSet (one valid time) > DataRecord (one 2D field)
  header.py, indexrecord.py         codecs of the 50-byte record header and the index record
  grid.py, vertical.py              horizontal geometry; VerticalAxis and its five subclasses
  packing.py, _pack.c               pack, unpack, checksum; the C extension for the encoder
  collection.py                     RecordCollection protocol, VariableView, VariableAccessor
  exceptions.py, _io.py, _time.py   ARLFormatError/Warning; atomic outputs; timestamps
  archives.py; directoryindex.py    NOAA archive downloads (Archive, ARCHIVES, get_archive); index, DirectoryIndex
  ops/                              extract_subset, sample_points, concat, concat_by_time
  xarray/                           open/write_dataset, engine="arl", ds.arl, pressure/z_agl/z_msl
```

`ops/` and `xarray/` sit on top of `file.py`; `File` imports them inside its
methods, keeping the import graph one-way. `archives.py` imports `fsspec` only
when it downloads, so it imports without the `archives` extra.

## ARL format rules

- **Records.** A file is a flat sequence of fixed-size records of
  `Header.N_BYTES + nx * ny` bytes (50 + nx*ny), one 2D field each. Each time
  step starts with an index record (`INDX`: grid, vertical axis, each level's
  variables), which must fit in one record; `validate_subset_record_length`
  (`ops/subset.py`) refuses a crop too small for it before writing anything.
- **Projection fields.** The 12 projection values are kept as stored (HYSPLIT reads
  them raw; to its `gbl2xy` a lat/lon `sync_lon` of -125 is not 235). `sync_x`/`sync_y`
  can lie off the grid: a projected crop keeps its source's sync point (`Grid.subset`).
- **Grids over 999 cells.** The thousands of `nx`/`ny` are a letter in the header
  (`A` = 1000, ...). `Header.grid` holds the thousands and the index record only
  the remainder, but `IndexRecord.nx`/`ny` are the full sizes (`__post_init__`
  rejects a mismatched `header.grid`); `index_length` is computed, never stored.
- **Packing.** Values are unsigned bytes of differences from the previous value.
  `pack` (`_pack.c`) and `unpack` give HYSPLIT's `pakout`/`pakinp` results bit for bit;
  `test_packing.py` transcribes both. Change them only with a round-trip test.
- **Checksums** are 1 to 255 (the byte sum, less 255 each time it reaches 256)
  unless every byte is 0. A stored 0 means none was recorded, so
  `verify_checksum()` returns True for it (HYSPLIT writes 0 for records it did not
  pack and never compares checksums; `docs/format.rst` has the NOAA cases).
- **`DIF*` records** pair by name, as HYSPLIT's `metinp` does (`record._corrects`):
  each with the nearest earlier record of its level the name allows (`DIFW`: `WWND`,
  `DZDT`; `DIFT`: `TKEN`, `TKEU`; surface `DIF*`: `TPP*`, `PRT*`), else none (warned).
  Writers refuse other pairs and write each DIF after its parent; `read()` adds all.
- **Malformed content raises `ARLFormatError`** (a `ValueError` subclass) from
  `File._scan` (the cases are in `docs/format.rst`); bad arguments stay plain
  `ValueError`/`TypeError`. A time step repeated byte for byte (some NOAA HRRR
  files) is skipped with an `ARLFormatWarning`; a repeat that differs raises.
  `File.check()` finds damage that opens cleanly; it reads headers, not values.
- **The vertical flag is the type.** `VerticalAxis.from_flag(flag, levels,
  offset=)` builds the subclass, mirroring HYSPLIT's `prfcom`: 1 `SigmaAxis`, 2
  `PressureAxis`, 3 `TerrainAxis`, 4 `HybridAxis`, 5 `WrfAxis`, else `ValueError`.
  Sigma/hybrid heights are HYSPLIT's `prfsig`/`prfecm` column (ground at `PRSS`,
  `T02M`; virtual T; a sigma file's `PRES` over nominal sigma); WRF pressure is `PRES`
  (eta needs coefficients the file lacks); pressure files use `HGTS`. No fallback across
  flags; each subclass takes only the keyword-only inputs it needs, else `ValueError`.
- **Rotated grids** (`orientation != 0`) raise `NotImplementedError` in `Projection`.

## Design invariants

1. **`grid.py` is horizontal only**, wind rotation included (`Grid.rotate_winds`,
   `Grid.meridian_convergence`); vertical geometry is in `vertical.py`.
2. **`level` is the stored vertical dimension.** `open_dataset` returns lazy
   NumPy-backed arrays (`ArlVariableArray`; dask only through `xr.open_dataset(...,
   chunks=)`) and adds no derived 3D coordinates (`z_agl`, `z_msl` are helpers).
   Attrs hold plain values only (the grid is the `arl_grid` coordinate's attrs),
   so datasets survive netCDF.
3. **`extract_subset` crops before unpacking.** With a bbox it passes a
   `GridWindow` to `DataRecord.read(window=...)`, which decodes only that tile.
   Without a horizontal crop it does not unpack: `_copy_subset_records` copies
   each record's bytes (DIF records included), renumbers the header's level, and
   rebuilds the index records. Keep both paths writing the same index records.
4. **Windowing matches HYSPLIT's `xtrct_grid`.** On lat/lon grids a cell is kept
   when its center is inside the bbox, edges included, `west > east` crosses the
   dateline, and a window across a global grid's seam has `x_stop > nx`. On projected
   grids the SW and NE corners are mapped to grid indices and rounded like Fortran
   `NINT` (reference: `test_grid.py`). A bbox is `(west, south, east, north)` in degrees.
5. **`write_dataset` handles the common case only:** the flat Dataset that
   `open_dataset` returns, with DIF records written from `attrs["diff"] =
   "DIF..."` on the parent. Per-variable forecast hours and other irregular
   layouts use `File`.
6. **No backward compatibility.** When a metadata representation changes, change
   every read and write path together; no code that accepts old attrs or formats.
7. **Writers stream one time step at a time, and reads do not cache.**
   `DataRecord._flush()` drops its buffers after writing and the writers call
   `File.flush()` after each time step; `File`, `RecordSet` and `DataRecord` form
   reference cycles, so a kept buffer lives until the cycle collector runs.
   `DataRecord.read()` returns a new array each call (lazy Dataset arrays keep
   their records alive); only `.data` caches. `tests/test_memory.py` guards both.
8. **Handles come from xarray's `CachingFileManager`**, so files reopen after
   `close()` or xarray's LRU cache closing them (`File.handle` reacquires). It
   gets bare "r"/"w" through `_open_binary`: xarray turns only "w" into "a" on
   reopen, so "wb" would truncate the output each time.
9. **Outputs appear only when complete.** `extract_subset`, `write_dataset` and
   archive downloads write a hidden `.partial` file and rename it on success
   (`_io.atomic_output`), and the first two refuse an output that is their input.
10. **Metadata types are immutable values**: `Projection`, `Grid`, `GridWindow`,
    `Header` and `IndexRecord` are frozen dataclasses, and `VerticalAxis` blocks
    assignment and keeps `levels` read-only. Hashing and `cached_property`
    (`Grid.crs`, `Grid.origin`) rely on this; build a new value instead.
11. **The public API is explicit.** The three `__init__.py` files re-export it,
    and modules define `__all__` (all but `ops/subset.py` and `ops/sample.py`).
    Options after the obvious leading arguments are keyword-only, except that
    `unpack` keeps six positional arguments (the wheel smoke test in
    `publish.yml` calls it so).
12. **Names.** Outputs are `dest` (a file) or `dest_dir` (a directory).
    `source` means only the 4-character ARL source ID (`File.source`,
    `ds.attrs["source"]`); the download classes are archives and their server is
    the `mirror`. `open_dataset`/`write_dataset` keep xarray's `filename_or_obj`.
13. **Containers follow `File.times`.** `File[int]`, `iter(File)` and
    `File.records` use sorted valid-time order, not on-disk order. `key in rs`
    accepts `rs[key]`'s `(level, variable)` keys and a bare variable name.

## Commands

The environment is uv-managed (`uv sync`, which also builds `_pack`); every task
is a `just` recipe, CI runs the same recipes, and `just` lists them:
`just quality-check` (lint, type check, docstrings, tests), `just test` (offline,
parallel; extra args go to pytest), `just test-network`, `just lint` /
`just format`, `just type-check` (pyrefly, strict), `just build-docs` (runs every
example; warnings are errors), `just docs-serve`, `just pre-commit`, `just dist`,
`just changelog`.

After editing dependencies, run `uv lock` online (`--offline` churns unrelated
pins) and commit `uv.lock` with them. uv rebuilds the editable install when
`_pack.c`, `setup.py`, `pyproject.toml` or git state change (`[tool.uv]
cache-keys`). `ruff format` skips `README.md` and `AGENTS.md` (hand-aligned).
Python 3.11+; runtime dependencies are numpy, pandas, pyproj, typing_extensions
and xarray (fsspec, s3fs for `[archives]`).

## Docs

`docs/api.rst` lists the API by topic; each class and each member gets a page
(`docs/_templates/autosummary/`, `docs/_ext/api_pages.py`; `docs/_autosummary/` is
generated). A property's page holds its See Also and Examples, so give every
public attribute and property a docstring. A new public function gets a user guide
(a gerund page such as `cropping.rst`, listed in `docs/user-guides.rst`), its API
entry, and a README line in the same change. Examples run when the docs build, so
a broken one fails the build:

- A `.. plot::` directive in a docstring's Examples section runs its code
  (doctest `>>>` lines work) and shows the figure.
- A `.. ipython:: python` block in an `.rst` page runs and shows each line's
  output; `@savefig name.png` above a plotting call saves the figure. A block that
  raises or warns fails the build unless marked `:okexcept:` / `:okwarning:`. All
  pages share one session, so each page imports what it uses. Notebooks are off.
- Examples run on GitHub Actions: synthetic data, a small file in the repository,
  or a public download, never a machine-specific path; seed random numbers. The
  repository has no sample ARL file, so an example builds one with
  `arlmet.write_dataset`; it never fetches an archive file (gigabytes) and shows
  that call as a plain `code-block`. An extra it needs goes in the `dev` group.

## Tests

- Tests build small synthetic ARL files whose grid must hold the index record
  (`50 + nx*ny >= len(index.tobytes())`); 20x20 fits a few variables and levels.
- `just test` runs `-m "not network and not slow"`, as CI does; a bare `pytest`
  also downloads from NOAA S3. CI runs the slow network tests (`test_subset_slow.py`,
  parity with HYSPLIT's `xtrct_grid`) only when started by hand.
- Markers are strict and warnings are errors (assert expected ones with `pytest.warns`).
- `test_memory.py` fails with the temp dir on a large-`st_blksize` filesystem.

## Commits, changelog, releases

- Conventional Commits (`fix(record): ...`, `docs: ...`, `!` if breaking).
  User-visible changes go under `## [Unreleased]` in `CHANGELOG.md`.
- The version comes from git tags (setuptools-scm): no version string to bump,
  and `uv.lock` does not change. Steps: [RELEASING.md](RELEASING.md). **Do not
  cut a release, create a tag, or push unless the maintainer asks.**
- The versioned docs live on the gh-pages branch, maintained by
  `.github/scripts/docs_versions.py`; never edit it by hand.
- Tooling comes from [jmineau/python-template](https://github.com/jmineau/python-template)
  via copier; `copier update` pulls in its changes (never edit `.copier-answers.yml`).
