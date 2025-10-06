arl-met Documentation
=====================

Welcome to the **arl-met** documentation!

This project provides a Python package for reading and analyzing NOAA ARL (Air Resources Laboratory) meteorological files.

.. note::
   🚧 This project is currently under active development.

Installation
------------

Install the package using pip:

.. code-block:: bash

   pip install git+https://github.com/jmineau/arl-met.git

For development, see :doc:`contributing`.

Quick Start
-----------

.. code-block:: python

   from arlmet import open_dataset

   # Open an ARL meteorological file
   ds = open_dataset("path/to/file.arl")

Goals
-----

- **Interpolate at a point**: Extract meteorological data at specific geographic locations
- **Get profiles**: Retrieve vertical atmospheric profiles
- **Timeseries**: Extract time series data for variables of interest
- **Maps**: Generate spatial maps of meteorological fields

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   contributing

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api

.. toctree::
   :maxdepth: 1
   :caption: Additional Information

   format

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
