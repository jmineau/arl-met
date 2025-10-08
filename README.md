# arl-met
NOAA ARL meteorological files

## 🚧 Work in Progress

This project aims to create a Python package for reading and analyzing NOAA ARL (Air Resources Laboratory) meteorological files.

### Status update

- Current capability: ARLMet can parse ARL files and load records into xarray DataArray/Dataset structures. Core data ingest and record unpacking are functional, but higher-level features remain limited.
- Immediate tasks:
    - Calculate vertical coordinates in meters above ground level (m AGL) from the ARL vertical axis information.
    - Calculate mean sea level (MSL) heights. This requires terrain elevation; if terrain is not present in the ARLMet data, provide and document a default terrain file (or a configurable fallback) to compute MSL.
    - Update dataset and variable attributes.
    - Make attributes CF-compliant (standard names, units, axis/coordinates, and global metadata).
- Next steps: add tests and examples for vertical coordinate handling, document the default-terrain behaviour, and expand CF attribute coverage across variables and coordinates.

## Installation

Install the package using pip:

```bash
pip install git+https://github.com/jmineau/arl-met.git
```

For development, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Documentation

Full documentation is available in the `docs/` directory. To build and view the documentation locally:

```bash
pip install -e ".[docs]"
make docs
```

Then open `docs/build/html/index.html` in your web browser.

## Goals

- **Interpolate at a point**: Extract meteorological data at specific geographic locations
- **Get profiles**: Retrieve vertical atmospheric profiles
- **Timeseries**: Extract time series data for variables of interest
- **Maps**: Generate spatial maps of meteorological fields
