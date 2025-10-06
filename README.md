# arl-met
NOAA ARL meteorological files

## 🚧 Work in Progress

This project aims to create a Python package for reading and analyzing NOAA ARL (Air Resources Laboratory) meteorological files.

## Installation

Install the package using pip:

```bash
pip install git+https://github.com/jmineau/arl-met.git
```

For development, clone the repository and install in editable mode:

```bash
git clone https://github.com/jmineau/arl-met.git
cd arl-met
pip install -e ".[dev]"
```

## Development

This project uses [Black](https://black.readthedocs.io/) for code formatting.

### Code Formatting

Format code automatically:
```bash
make format
# or directly with black
black arlmet/
```

Check code formatting:
```bash
make check
# or directly with black
black --check arlmet/
```

### Pre-commit Hooks

Install pre-commit hooks to automatically format code before commits:
```bash
pre-commit install
```

The pre-commit hook will automatically run Black and other checks on your code before each commit.

## Goals

- **Interpolate at a point**: Extract meteorological data at specific geographic locations
- **Get profiles**: Retrieve vertical atmospheric profiles
- **Timeseries**: Extract time series data for variables of interest
- **Maps**: Generate spatial maps of meteorological fields
