# arl-met

[![Tests](https://github.com/jmineau/arl-met/actions/workflows/tests.yml/badge.svg)](https://github.com/jmineau/arl-met/actions/workflows/tests.yml)
[![Documentation](https://github.com/jmineau/arl-met/actions/workflows/docs.yml/badge.svg)](https://github.com/jmineau/arl-met/actions/workflows/docs.yml)
[![Code Quality](https://github.com/jmineau/arl-met/actions/workflows/quality.yml/badge.svg)](https://github.com/jmineau/arl-met/actions/workflows/quality.yml)
[![codecov](https://codecov.io/gh/jmineau/arl-met/branch/main/graph/badge.svg)](https://codecov.io/gh/jmineau/arl-met)
[![PyPI version](https://badge.fury.io/py/arlmet.svg)](https://badge.fury.io/py/arlmet)
[![Python Version](https://img.shields.io/pypi/pyversions/arlmet.svg)](https://pypi.org/project/arlmet/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pyright](https://img.shields.io/badge/pyright-checked-brightgreen.svg)](https://github.com/microsoft/pyright)

Read, write, subset, and sample NOAA ARL meteorological files.

`arl-met` provides a Python-first interface to the ARL packed meteorology
format used by HYSPLIT and related workflows. It supports:

- low-level record-preserving reads and writes through `File`, `RecordSet`, and `DataRecord`
- xarray Dataset reads and common-case writes through `open_dataset()` and `write_dataset()`
- NOAA source fetching helpers for common ARL archives
- crop-before-unpack subset extraction with `extract_subset()`
- vertical helper functions such as `pressure()`, `z_agl()`, and `z_msl()`
- point sampling with `sample_points()`

## Alpha status

This is an alpha release. The core read/write/subset APIs are usable, but the
package is still tightening its high-level contracts and release surface.

Current strengths:

- low-level ARL fidelity, including preservation of trailing `DIF*` records
- xarray-native analysis workflow for common ARL files
- direct subset extraction and point sampling
- tested support for Python 3.10 through 3.12

Current limitations:

- `write_dataset()` is intentionally conservative and targets the flat common-case Dataset contract
- complex multi-record DIFF chains are not tested
- WRF vertical flag 5 is not implemented

## Installation

Install the core package:

```bash
pip install arlmet
```

Install the optional source-fetching dependencies:

```bash
pip install "arlmet[sources]"
```

For development:

```bash
git clone https://github.com/jmineau/arl-met.git
cd arl-met
uv sync --dev
```

## Quick examples

Open an ARL file as a Dataset:

```python
import arlmet

ds = arlmet.open_dataset("met.arl")
print(ds)
```

Modify a Dataset and write it back:

```python
import arlmet

ds = arlmet.open_dataset("met.arl")
ds["TEMP"] = ds["TEMP"] - 273.15
ds["WWND"].attrs["diff"] = "DIFW"
arlmet.write_dataset(ds, "edited.arl")
```

Extract a subset without unpacking the full file first:

```python
import arlmet

arlmet.extract_subset(
    "met.arl",
    "subset.arl",
    bbox=(-114.0, 39.0, -110.0, 42.0),
    levels=[0, 1, 2],
)
```

Use the low-level writer for irregular layouts:

```python
import numpy as np
import pandas as pd
import arlmet

grid = arlmet.Grid(
    projection=arlmet.Projection(
        pole_lat=90.0,
        pole_lon=0.0,
        tangent_lat=1.0,
        tangent_lon=1.0,
        grid_size=0.0,
        orientation=0.0,
        cone_angle=0.0,
        sync_x=1.0,
        sync_y=1.0,
        sync_lat=-10.0,
        sync_lon=20.0,
    ),
    nx=20,
    ny=20,
)
vertical_axis = arlmet.VerticalAxis(flag=2, levels=[0.0, 1000.0])
time = pd.Timestamp("2024-07-18 00:00")

prss = np.ones((grid.ny, grid.nx), dtype=np.float32)
wwnd = np.ones((grid.ny, grid.nx), dtype=np.float32)

with arlmet.File("custom.arl", mode="w", source="TEST", grid=grid, vertical_axis=vertical_axis) as arl:
    rs = arl.create_recordset(time, forecast=0)
    rs.create_datarecord("PRSS", level=0, forecast=0, data=prss)
    rs.create_datarecord("WWND", level=1, forecast=0, data=wwnd, diff="DIFW")
```

## Resources

Documentation is available at https://jmineau.github.io/arl-met/

Useful ARL/HYSPLIT references:

- [HYSPLIT User Guide](https://www.arl.noaa.gov/documents/reports/hysplit_user_guide.pdf) for the broader model and file-format context.
- [HYSPLIT meteorology page](https://www.ready.noaa.gov/hysplitusersguide/S141.htm) for the ARL meteorology format overview.
- [READY archive](https://www.ready.noaa.gov/archives.php) for the available meteorology archives.
- [GDAS1 packing notes](https://www.ready.noaa.gov/gdas1.php) for a concrete example of ARL packing behavior.

Related project:

- [ARLreader](https://github.com/martin-rdz/ARLreader), which focuses on GDAS1 files, while `arl-met` targets a broader ARL/xarray workflow.

## Release notes

See [CHANGELOG.md](CHANGELOG.md) for release history.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for development
setup and contribution guidelines.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
