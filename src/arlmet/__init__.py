"""
arlmet: Python package for reading and analyzing NOAA ARL meteorological files.

This package provides tools to read, parse, and work with ARL (Air Resources Laboratory)
packed meteorological data files used by HYSPLIT and other atmospheric transport models.
"""

__version__ = "2025.10.1"
__author__ = "James Mineau"
__email__ = "jameskmineau@gmail.com"

from .core import DataRecord, File, RecordSet, open_dataset
from .grid import Grid, Projection, Surface, VerticalAxis
from .metadata import Header, IndexRecord
from .packing import calculate_checksum, pack, unpack

__all__ = [
    "File",
    "RecordSet",
    "DataRecord",
    "open_dataset",
    "Projection",
    "Grid",
    "Surface",
    "VerticalAxis",
    "Header",
    "IndexRecord",
    "calculate_checksum",
    "pack",
    "unpack",
]
