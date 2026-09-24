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

# AGENTS.md — Developer and Agent Guide for arl-met

This document gives a new engineer or agent the context needed to contribute
to `arl-met` without redoing discovery. Read it before touching any source file.

## What This Package Does

`arl-met` reads, writes, and manipulates NOAA ARL binary meteorology files.
These files are the main data format consumed by HYSPLIT and PYSTILT
(the Python STILT implementation). The package provides:

- a low-level file/record model (`File`, `RecordSet`, `DataRecord`)
- an xarray-native Dataset path (`open_dataset`, `write_dataset`) for the common case
- a direct subset extractor (`extract_subset`) — crops before unpack, HYSPLIT-style
- a point-sampling API (`sample_points`)
- a file concatenator (`concat`) — joins ARL files into one, byte-level append
- a batch concatenator (`concat_by_time`) — groups a directory by valid time
- vertical helpers (`pressure`, `z_agl`, `z_msl`)

The primary caller is PYSTILT. PYSTILT uses `extract_subset` and `sample_points`
in the inner loop of particle trajectory calculations.

## ARL Format Constraints That Bite Developers

These are the format rules that cause silent bugs or confusing failures if ignored.

### Record layout
Each ARL file is a flat sequence of fixed-size binary records. Every record is
`Header.N_BYTES + nx * ny` bytes (`50 + nx*ny`). Records contain one 2D field
(one variable, one level, one time).

### Index record
The first record at each time step is always an index record (`variable="INDX"`).
The index encodes the grid, vertical axis, and a per-level variable manifest. It
must fit in exactly one record block (`50 + nx*ny` bytes). If a cropped grid is
too small, the index record overflows. `validate_subset_record_size()` in
`subset.py` checks this before writing.

### Differential records
Some variable names begin with `DIF` (e.g., `DIFZ`). These are in-stream
correction fields. The documented behavior is in the HYSPLIT user guide. Do not
assume every `DIF*` field is an automatic correction for the prior record. Only
implement the documented subset of DIFF behavior. In arl-met's low-level model,
a DIF record belongs to the immediately preceding non-DIF record in the stream
at the same time and level.

### Packing
ARL packs floats as unsigned bytes relative to the previous row or column value.
The packer and unpacker are in `packing.py`. The checksum is XOR of all packed
bytes. Do not modify `pack`/`unpack` without pack/unpack round-trip test coverage.

### Large-grid letter codes
When `nx` or `ny` exceeds 999, ARL encodes the overflow digit as a letter (`A`=1,
`B`=2, ...). `split_grid_component` and the inverse in `header.py` handle this.

### Vertical flag and the polymorphic axis model
`VerticalAxis` is an **abstract base class**. The ARL vertical flag determines
the subclass, constructed via `VerticalAxis.from_flag(flag, levels, offset=)`:

| Flag | Subclass | `to_pressure` | `to_height_agl` |
|------|----------|---------------|-----------------|
| 1 | `SigmaAxis` | sigma → pressure formula | hypsometric integration (PRSS + TEMP) |
| 2 | `PressureAxis` | returns stored levels | HGTS − terrain (requires HGTS) |
| 3 | `TerrainAxis` | raises ValueError | returns stored levels (already AGL) |
| 4 | `HybridAxis` | hybrid → pressure formula | hypsometric integration (PRSS + TEMP) |
| 5 | (not implemented) | raises NotImplementedError | raises NotImplementedError |

**This mirrors HYSPLIT's `prfcom` dispatcher exactly.** Each coordinate system
has one correct method — the flag IS the type. Key consequences:

- **Sigma/hybrid files never have HGTS.** Heights are always hypsometric.
- **Pressure files always have HGTS.** `z_agl`/`z_msl` require it — no
  hypsometric fallback.
- **Do not add HGTS-first fallback paths.** The previous code tried this; it
  was removed to match HYSPLIT.

Direct `VerticalAxis(...)` construction is not allowed (abstract). Use
`VerticalAxis.from_flag(...)` or a subclass constructor directly (e.g.
`PressureAxis(levels=[...])`).

## Module Layout

```
src/arlmet/
  __init__.py      public exports
  record.py        require_mode decorator + DataRecord (single 2D packed field)
  recordset.py     RecordCollection, VariableView, VariableAccessor, RecordSet
  file.py          File — top-level file handle, scan, factories
  grid.py          Grid, GridWindow, Projection — horizontal geometry only
  vertical.py      VerticalAxis (ABC), SigmaAxis, PressureAxis, TerrainAxis, HybridAxis
  header.py        Header, helper functions — 50-byte record header codec
  index.py         IndexRecord, VarInfo, LvlInfo — index record codec
  packing.py       pack(), unpack(), calculate_checksum()
  xarray/          open_dataset(), write_dataset(), vertical helpers
    __init__.py    re-exports all public xarray symbols
    _backend.py    ArlVariableArray (xarray BackendArray)
    _coords.py     arl_grid coord codec + Dataset vertical-axis helpers
    _accessor.py   ARLDatasetAccessor (ds.arl.grid, .vertical_axis, .source)
    _vertical.py   pressure(), z_agl(), z_msl()
    dataset.py     open_dataset() / write_dataset() — flat Dataset read/write path
  ops/             operations on ARL files (subset, sample, concat)
    __init__.py    re-exports extract_subset, sample_points, concat, concat_by_time
    subset.py      extract_subset(), resolve_window(), normalize_levels()
    sample.py      sample_points()
    concat.py      concat(), concat_by_time() — join ARL files into one
  sources.py       HrrrSource, NamSource, GdasSource, GfsSource — NOAA S3 downloads
tests/
  test_grid.py
  test_low_level.py
  test_memory.py
  test_metadata.py
  test_packing.py
  test_pkg.py
  test_sample.py
  test_concat.py
  test_sources.py
  test_subset.py
  test_vertical.py
  test_writer.py
data/              real ARL sample files (not committed — local only)
  RP202407.gbl     73x144 lat/lon, 124 times, 18 levels — good benchmark target
  gdas1.sep25.w1
  hysplit.t01z.hrrrf
  ... (various NAM, GFS, HRRR samples)
```

## Dependency Graph (runtime)

Arrows show runtime imports. TYPE_CHECKING-only imports are not shown.

```
grid, packing                       ← leaf nodes
header → grid
index  → grid, header, vertical
vertical  (no imports — leaf node)
record    → grid, header, packing, vertical
recordset → grid, header, index, record, vertical
             (delayed import: xarray, inside VariableView.to_xarray only)
file      → grid, header, index, record, recordset, vertical
             (delayed: ops.subset in File.extract_subset, ops.sample in
              File.sample_points — ops sits on top of file, so file's use of
              ops is lazy to keep the file↔ops dependency one-way at import time)
ops/__init__ → ops.concat, ops.subset, ops.sample   (re-exports the public ops)
ops.subset   → file, grid, header, index, vertical
ops.sample   → grid, vertical   (TYPE_CHECKING: file, record, recordset)
ops.concat   → file, index
xarray/   → file, grid, ops.subset, vertical  (TYPE_CHECKING: record, recordset)
sources   → ops.subset
```

Delayed (in-function-body) imports: `VariableView.to_xarray()` → `xarray/`
(xarray imports from recordset); `File.extract_subset()` → `ops.subset` and
`File.sample_points()` → `ops.sample` (ops depends on file, so file imports ops
lazily to avoid a cycle). All other inter-module dependencies are explicit
top-level imports.

## Design Invariants — Do Not Violate

1. **`grid.py` is horizontal-only.** `Grid` owns projection, shape, and
   coordinate generation. Vertical geometry belongs in `vertical.py`.
   Wind rotation (`Grid.meridian_convergence`, `Grid.rotate_winds`) lives here
   too: it is a property of the horizontal projection, computed from
   `pyproj.Proj(grid.crs).get_factors(...).meridian_convergence`.

2. **`level` is the stored vertical dimension.** Do not store derived 3D
   coordinates (`z_agl`, `z_msl`) as default xarray coordinates. They are
   opt-in helpers.

3. **NumPy first.** Default `open_dataset` uses NumPy-backed lazy
  arrays (`ArlVariableArray`).

4. **No live Python objects in xarray attrs.** All attrs must be JSON-serializable
   so datasets survive netCDF round-trips.

5. **`extract_subset` crops before unpack.** The performance gain comes from
   passing a `GridWindow` to `DataRecord.read(window=...)`, which decodes only
   the requested tile. Do not revert to "open full dataset, then subset".

6. **`window_from_bbox` uses HYSPLIT-inclusive windowing.** The grid index
   selection matches `xtrct_grid.f`: pixels are included if their center falls
   within the bbox (inclusive on both edges). See `grid.py:window_from_bbox`
   and `test_grid.py` for the reference behavior.

7. **bbox input is lat/lon only.** `extract_subset` and `open_dataset` accept
   `bbox=(west, south, east, north)` in degrees EPSG:4326. Projected inputs are
   not supported yet. A future `bbox_crs` parameter should raise on anything
   other than `"EPSG:4326"` until full reprojection is implemented.

8. **`write_dataset` is intentionally common-case only.** It writes the flat
  Dataset representation returned by `open_dataset`: surface variables have no
  `level` dimension, upper-air variables share one `level` coordinate, and
  `forecast_hour(time)` represents only the index-record forecast. DIF writing
  is supported through explicit parent metadata (`DataArray.attrs["diff"] =
  "DIF..."`), but per-variable forecast heterogeneity and other irregular
  layouts should still use the low-level `File` API instead.

9. **No backward compatibility.** When metadata representations change, update
   all write/read paths together. Do not add fallback code paths that accept old
   formats or legacy attrs. Datasets produced by older versions of the library
   are not supported; users must re-open files with the current library version.

10. **The flag IS the type — no cross-flag fallbacks.** Vertical coordinate
    dispatch is handled by polymorphic `VerticalAxis` subclasses, mirroring
    HYSPLIT's `prfcom`. Each coordinate system has exactly one correct method
    for deriving pressure and height. Do not add "if HGTS present, use it"
    fallbacks for sigma/hybrid, or hypsometric fallbacks for pressure files.

11. **Writers stream one time step at a time.** `DataRecord._flush()` drops
    `_unpacked`/`_packed`/`_bytes` after writing (the record flips to mode
    "r"), and `extract_subset`/`write_dataset` call `File.flush()` after each
    time step. Do not reintroduce whole-file buffering: `File`, `RecordSet`,
    and `DataRecord` form reference cycles, so any buffer a written File keeps
    lives until the cycle collector runs (this was the extract_subset
    "leak"). `tests/test_memory.py` guards peak and retained memory.

12. **`DataRecord.read()` never caches; `.data` is the cache.** `read()` is
    a pure disk read (new array each call). Lazy `open_dataset()` arrays keep
    their source records alive, so a caching `read()` made them accumulate
    the whole file. Only `DataRecord.data` stores the full field.

13. **File handle management.** `File` gets its handle from xarray's
    `CachingFileManager` so read-mode files can reopen after `close()` (lazy
    `open_dataset` arrays read after the File is closed). The manager is given
    bare mode "r"/"w" through `_open_binary` because xarray only switches "w"
    to "a" on reopen — passing "wb" made any reopen truncate the output.
    `File.handle` reacquires when xarray's global LRU cache
    (`file_cache_maxsize`, default 128) has closed the cached handle.

## Public API Summary

```python
import arlmet

# Read — analysis view
ds = arlmet.open_dataset("file.arl")
ds = arlmet.open_dataset("file.arl", bbox=(-112, 40, -111, 41), levels=[0, 1, 2])

# Read/write — common-case Dataset path
ds = arlmet.open_dataset("file.arl", squeeze=False)
ds["TEMP"] -= 273.15
ds["WWND"].attrs["diff"] = "DIFW"
arlmet.write_dataset(ds, "out.arl")

# Direct subset extraction (fast, crop-before-unpack)
# (returns the new File opened for reading — close it; an unclosed File holds
# its file handle until garbage collection)
arlmet.extract_subset("in.arl", "out.arl", bbox=(-130, 20, -60, 60)).close()
arlmet.extract_subset("in.arl", "out.arl", levels=[0, 1, 2], variables=["UWND", "VWND"]).close()

# Point sampling
import pandas as pd
points = pd.DataFrame({"lon": [-111.9], "lat": [40.7], "z": [850.0]})
result = arlmet.sample_points("file.arl", points, ["UWND", "VWND"], z_kind="pressure")
# Grid-relative -> east/north winds on projected grids (both components required)
result = arlmet.sample_points("file.arl", points, ["UWND", "VWND"], earth_relative=True)

# Concatenate ARL files into one (byte-level append; orders by valid time)
arlmet.concat(["20240101_00_hrrr", "20240101_06_hrrr"], "20240101_hrrr")

# Batch concat: group a directory of files into chunks by valid time
# (time read from each file's index record, not its name)
arlmet.concat_by_time("hrrr/", "daily/", freq="1D", pattern="*_hrrr",
                        template="{time:%Y%m%d}_hrrr")

# Vertical helpers (operate on open_dataset output)
p = arlmet.pressure(ds)           # DataArray of pressure levels
h_agl = arlmet.z_agl(ds)         # height above ground
h_msl = arlmet.z_msl(ds)         # height above MSL

# Low-level
with arlmet.File("file.arl") as f:
    print(f.grid, f.vertical_axis, f.times)
    rs = f[f.times[0]]          # RecordSet
    rec = rs.records[0]         # DataRecord
    arr = rec.read()            # np.ndarray (full grid)
    arr = rec.read(window=...)  # np.ndarray (cropped tile)

# Low-level writing for irregular files
with arlmet.File("out.arl", mode="w", source="TEST", grid=grid, vertical_axis=vaxis) as f:
  rs = f.create_recordset(times[0], forecast=0)
  rs.create_datarecord("PRSS", level=0, forecast=0, data=prss)
  rs.create_datarecord("TEMP", level=1, forecast=3, data=temp)
  rs.create_datarecord("WWND", level=1, forecast=3, data=wwnd, diff="DIFW")
  f.flush()  # write this time step and free its buffers; call once per time step
```

## sample_points z_kind Reference

| `z_kind` | `z` units | flag=1/4 (sigma/hybrid) requires | flag=2 (pressure) requires | flag=3 (terrain) |
|----------|-----------|----------------------------------|----------------------------|-------------------|
| `"native"` | level index | — | — | — |
| `"pressure"` | hPa | PRSS | — (stored levels) | raises ValueError |
| `"agl"` | meters | PRSS + TEMP (hypsometric) | HGTS + SHGT | — (stored levels) |
| `"msl"` | meters | PRSS + TEMP + SHGT | HGTS | SHGT |

## Development Workflow

```bash
# install with dev dependencies
uv sync --dev

# run tests
uv run pytest -q

# lint and format
uv run ruff check .
uv run ruff format .

# pre-commit (runs ruff + other hooks)
pre-commit run --all-files
```

Python ≥ 3.10 required. Runtime dependencies: `numpy`, `pandas`, `pyproj`,
`xarray`.

**Keep `uv.lock` in sync.** `arlmet` is an editable install, so its own
version is pinned in `uv.lock`. When you bump the version in `pyproject.toml`
(e.g. during a release), re-run `uv lock` (or `uv sync`) and commit the updated
`uv.lock` in the same commit. Otherwise the editable-install pre-commit hook
re-syncs the lockfile on every run and blocks commits with a dirty tree. Run
`uv lock` online — `uv lock --offline` re-resolves packages from the local
cache and can churn many unrelated pins. A correct release bump changes
exactly one line of `uv.lock`.

## Testing Notes

- Tests use synthetic ARL fixtures, not the large files in `data/`.
- When writing a synthetic file, the grid must be large enough that
  `50 + nx*ny >= len(index_record.tobytes())`. A 20x20 grid is usually safe
  for a few variables and levels.
- `test_subset.py`, `test_sample.py`, and `test_concat.py` cover the newest APIs.
- `test_writer.py` covers round-trip correctness.
- `test_memory.py` bounds writer memory with `tracemalloc` (peak during
  `extract_subset`/`write_dataset`, and memory retained after return with the
  GC disabled, and a lazy-Dataset `write_dataset`). Keep its grids ~200x200
  so array buffers dominate Python object overhead.
- Run offline tests with `-m "not network and not slow"` (as CI does). A bare
  `pytest` also runs the network tests, which hit NOAA S3.
- For performance benchmarks, use `data/RP202407.gbl` (73x144, 124 times, 18
  levels). The direct `extract_subset` path takes ~3.8s for a 25x53 North
  America crop on that file.

## Known Limitations and Open Work

- **Flag 5 (WRF) vertical axes are not implemented.** Flags 1–4 are fully
  supported via polymorphic `VerticalAxis` subclasses. Adding flag 5 means
  adding a `WrfAxis` subclass with `to_pressure` and `to_height_agl` methods.
- **No xarray `BackendEntrypoint`.** The current `ArlVariableArray` is
  backend-style but is not registered as a proper xarray engine. This is
  intentional until the subset path matures enough to support lazy slice
  indexing efficiently.
- **No `bbox_crs` support yet.** The parameter should be accepted and raise on
  non-EPSG:4326 inputs as a future extension point.
- **DIFF variable handling is partial.** Read/rewrite/subset preservation is
  supported for in-stream `DIF*` records, and generated DIF writing follows the
  documented residual-after-pack pattern. Complex multi-record DIFF chains are
  not tested.
- **`write_dataset` is intentionally conservative.** It supports the common
  Dataset shape only. DIF generation is supported via `attrs["diff"]`, but
  files with per-variable forecast heterogeneity or other irregular layouts
  should be authored through `File`.

## Format References

- HYSPLIT ARL format overview: https://www.ready.noaa.gov/hysplitusersguide/S141.htm
- HYSPLIT "Compilation Limits" (max 12 met files per simulation with a single
  grid — the reason `concat`/`concat_by_time` exist):
  https://www.ready.noaa.gov/hysplitusersguide/S441.htm
- GDAS packing details: https://www.ready.noaa.gov/gdas1.php
- READY archive catalog: https://www.ready.noaa.gov/archives.php
- ARLreader notes: https://github.com/martin-rdz/ARLreader/blob/master/working_with_ARLformat.md
