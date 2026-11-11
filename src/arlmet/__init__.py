"""
arlmet: Python package for reading and analyzing NOAA ARL meteorological files.

This package provides tools to read, parse, and work with ARL (Air Resources Laboratory)
packed meteorological data files used by HYSPLIT and other atmospheric transport models.

The byte-level codec is not imported here: ``Header`` and ``IndexRecord`` are in
:mod:`arlmet.record`, and ``pack``, ``unpack`` and ``calculate_checksum`` in
:mod:`arlmet.packing`.
"""

from . import converters
from ._inventory import Inventory, inventory
from ._version import __version__ as __version__
from .exceptions import ARLFormatError, ARLFormatWarning
from .file import File
from .grid import Grid, GridWindow, Projection
from .ops import concat, concat_by_time, extract_subset, sample_points
from .probe import Probe, probe
from .record import DataRecord
from .recordset import RecordSet
from .vertical import (
    HybridAxis,
    PressureAxis,
    SigmaAxis,
    TerrainAxis,
    VerticalAxis,
    WrfAxis,
)
from .xarray import open_dataset, pressure, write_dataset, z_agl, z_msl

__author__ = "James Mineau"
__email__ = "jameskmineau@gmail.com"

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
    "GridWindow",
    "VerticalAxis",
    "SigmaAxis",
    "PressureAxis",
    "TerrainAxis",
    "HybridAxis",
    "WrfAxis",
    "extract_subset",
    "sample_points",
    "concat",
    "concat_by_time",
    "inventory",
    "Inventory",
    "probe",
    "Probe",
    "ARLFormatError",
    "ARLFormatWarning",
    "converters",
]
