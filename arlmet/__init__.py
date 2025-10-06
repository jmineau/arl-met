"""
ARL-MET: Python package for reading and analyzing NOAA ARL meteorological files.

This package provides tools to read, parse, and work with ARL (Air Resources Laboratory)
packed meteorological data files used by HYSPLIT and other atmospheric transport models.
"""

from .arlmet import ARLMet, open_dataset
