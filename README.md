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

Read and analyze ARL meteorological files.

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

### From Source

```bash
git clone https://github.com/jmineau/arl-met.git
cd arl-met
pip install -e .
```

## Usage

```python
import arlmet

# Add usage example here
```

## Documentation

Full documentation is available at [https://jmineau.github.io/arl-met/](https://jmineau.github.io/arl-met/)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**James Mineau** - [jmineau](https://github.com/jmineau)
