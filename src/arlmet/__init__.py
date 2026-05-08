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
from .metadata import Header, IndexRecord
from .packing import calculate_checksum, pack, unpack
from .record import DataRecord
from .recordset import RecordSet
from .sampling import sample_points, terrain
from .subset import extract_subset
from .surface import Surface
from .vertical import (
    Grid3D,
    VerticalAxis,
    interp_vertical,
    pressure,
    z_agl,
    z_msl,
)
from .xarray import open_dataset, write_dataset

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
    "pressure",
    "z_agl",
    "z_msl",
    "interp_vertical",
    "Header",
    "IndexRecord",
    "calculate_checksum",
    "pack",
    "unpack",
    "extract_subset",
    "terrain",
    "sample_points",
]
