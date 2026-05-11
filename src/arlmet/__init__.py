"""
arlmet: Python package for reading and analyzing NOAA ARL meteorological files.

This package provides tools to read, parse, and work with ARL (Air Resources Laboratory)
packed meteorological data files used by HYSPLIT and other atmospheric transport models.
"""

from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
import re


def _detect_version() -> str:
    try:
        return package_version("arlmet")
    except PackageNotFoundError:
        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        match = re.search(
            r'^version\s*=\s*"([^"]+)"',
            pyproject.read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
        if match is None:
            return "0+unknown"
        return match.group(1)


__version__ = _detect_version()
__author__ = "James Mineau"
__email__ = "jmineau@gmail.com"

from .file import File
from .grid import Grid, Projection
from .header import Header
from .index import IndexRecord
from .packing import calculate_checksum, pack, unpack
from .record import DataRecord
from .recordset import RecordSet
from .sampling import sample_points
from .subset import extract_subset
from .vertical import VerticalAxis
from .xarray import open_dataset, pressure, write_dataset, z_agl, z_msl

__all__ = [
    "File",
    "RecordSet",
    "DataRecord",
    "open_dataset",
    "write_dataset",
    "pressure",
    "z_agl",
    "z_msl",
    "Projection",
    "Grid",
    "VerticalAxis",
    "Header",
    "IndexRecord",
    "calculate_checksum",
    "pack",
    "unpack",
    "extract_subset",
    "sample_points",
]
