# arl-met
NOAA ARL meteorological files

## 🚧 Work in Progress

This project aims to create a Python package for reading and analyzing NOAA ARL (Air Resources Laboratory) meteorological files.

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
