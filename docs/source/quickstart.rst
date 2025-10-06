Quick Start
===========

This guide will help you get started with arl-met.

Opening ARL Files
-----------------

The main entry point is the :func:`~arlmet.open_dataset` function:

.. code-block:: python

   from arlmet import open_dataset

   # Open an ARL meteorological file
   ds = open_dataset("path/to/file.arl")

Using the ARLMet Class
----------------------

You can also use the :class:`~arlmet.arlmet.ARLMet` class directly:

.. code-block:: python

   from arlmet import ARLMet

   # Create an ARLMet instance
   arl = ARLMet("path/to/file.arl")

   # Access records
   records = arl.records

Working with Data
-----------------

The package reads ARL packed meteorological data and provides access to:

- **Surface variables**: Pressure, temperature, winds, fluxes, precipitation
- **Upper-air variables**: Winds, temperature, heights, moisture

Example workflow:

.. code-block:: python

   from arlmet import open_dataset

   # Open the file
   ds = open_dataset("data.arl")

   # Access variables, coordinates, and metadata
   # (specific API details depend on implementation)

Next Steps
----------

- See the :doc:`api` for detailed API documentation
- Check out the :doc:`format` page to understand the ARL file format
