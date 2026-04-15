"""
arlmet: Python package for reading and analyzing NOAA ARL meteorological files.

This package provides tools to read, parse, and work with ARL (Air Resources Laboratory)
packed meteorological data files used by HYSPLIT and other atmospheric transport models.
"""

__version__ = "2025.10.1"
__author__ = "James Mineau"
__email__ = "jameskmineau@gmail.com"

from .core import DataRecord, File, RecordSet
from .grid import Grid, Projection
from .metadata import Header, IndexRecord
from .packing import calculate_checksum, pack, unpack
from .vertical import Grid3D, Surface, VerticalAxis
from .xarray_io import open_dataset, write_dataset

__all__ = [
    "File",
    "RecordSet",
    "DataRecord",
    "open_dataset",
    "write_dataset",
    "Projection",
    "Grid",
    "Grid3D",
    "Surface",
    "VerticalAxis",
    "Header",
    "IndexRecord",
    "calculate_checksum",
    "pack",
    "unpack",
]
