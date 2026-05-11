"""
arlmet: Python package for reading and analyzing NOAA ARL meteorological files.

This package provides tools to read, parse, and work with ARL (Air Resources Laboratory)
packed meteorological data files used by HYSPLIT and other atmospheric transport models.
"""

__version__ = "2025.10.1"
__author__ = "James Mineau"
__email__ = "jameskmineau@gmail.com"

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
