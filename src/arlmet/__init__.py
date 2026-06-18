"""
arlmet: Python package for reading and analyzing NOAA ARL meteorological files.

This package provides tools to read, parse, and work with ARL (Air Resources Laboratory)
packed meteorological data files used by HYSPLIT and other atmospheric transport models.
"""

import importlib.metadata

from .concat import concat, concat_by_time
from .file import File
from .grid import Grid, Projection
from .header import Header
from .index import IndexRecord
from .packing import calculate_checksum, pack, unpack
from .record import DataRecord
from .recordset import RecordSet
from .sampling import sample_points
from .subset import extract_subset
from .vertical import HybridAxis, PressureAxis, SigmaAxis, TerrainAxis, VerticalAxis
from .xarray import open_dataset, pressure, write_dataset, z_agl, z_msl

__version__ = importlib.metadata.version("arlmet")
__author__ = "James Mineau"
__email__ = "jmineau@gmail.com"

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
    "SigmaAxis",
    "PressureAxis",
    "TerrainAxis",
    "HybridAxis",
    "Header",
    "IndexRecord",
    "calculate_checksum",
    "pack",
    "unpack",
    "extract_subset",
    "sample_points",
    "concat",
    "concat_by_time",
]
