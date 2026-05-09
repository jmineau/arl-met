Downloading Archived Meteorology
================================

arl-met includes source classes for downloading ARL meteorology files from the
NOAA ARL public archives.

Install the source dependencies first:

.. code-block:: bash

   pip install "arlmet[sources]"

Choose a source class
---------------------

Each class knows the filename and archive layout for one product.

.. list-table:: Available source classes
   :header-rows: 1

   * - Class
     - Product
     - Typical coverage
   * - :class:`arlmet.sources.HRRRSource`
     - HRRR 3 km
     - CONUS, 6-hour files
   * - :class:`arlmet.sources.NAMSource`
     - NAM 12 km
     - North America, daily files
   * - :class:`arlmet.sources.GDASSource`
     - GDAS 1 degree
     - Global, weekly files
   * - :class:`arlmet.sources.GFSSource`
     - GFS 0.25 degree
     - Global, daily files
   * - :class:`arlmet.sources.ReanalysisSource`
     - NCEP/NCAR Reanalysis
     - Global, monthly files

Download files for a time range
-------------------------------

.. code-block:: python

   from arlmet.sources import HRRRSource

   source = HRRRSource()
   files = source.fetch(
       "2024-07-18 00:00",
       "2024-07-19 00:00",
       local_dir="./met",
   )

``fetch()`` returns the local paths in chronological order. Duplicate archive
keys are removed automatically when the requested time range spans multiple
hours within the same ARL file.

Crop on download
----------------

For large global products, pass ``bbox=`` so the downloaded file is cropped
before it is cached locally.

.. code-block:: python

   from arlmet.sources import GFSSource

   source = GFSSource()
   files = source.fetch(
       "2024-07-18 00:00",
       "2024-07-19 00:00",
       local_dir="./met",
       bbox=(-114.0, 39.0, -110.0, 42.0),
   )

This uses :func:`arlmet.extract_subset` internally after the raw file is
downloaded.

Choose a backend
----------------

The default backend is ``"s3"`` and is usually the fastest choice.

.. code-block:: python

   files = source.fetch(
       "2024-07-18",
       "2024-07-19",
       local_dir="./met",
       backend="ftp",
   )

Supported backends are:

- ``"s3"``: NOAA public S3 bucket via ``s3fs``
- ``"ftp"``: NOAA ARL FTP archive
- ``"http"``: READY web archive

Caching and overwrite behavior
------------------------------

Downloaded files are reused if a matching local file already exists. Pass
``overwrite=True`` to force a fresh download.

.. code-block:: python

   files = source.fetch(
       "2024-07-18",
       "2024-07-19",
       local_dir="./met",
       overwrite=True,
   )