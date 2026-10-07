Downloading Archived Meteorology
================================

arl-met includes archive classes for downloading ARL meteorology files from the
NOAA ARL public archives.

Install the archive dependencies first:

.. code-block:: bash

   pip install "arlmet[archives]"

Choose an archive
-----------------

Each class knows the filename and archive layout for one product, and is
registered under a short name.

.. list-table:: Available archives
   :header-rows: 1

   * - Name
     - Class
     - Product
     - Typical coverage
   * - ``"hrrr"``
     - :class:`arlmet.archives.HRRRArchive`
     - HRRR 3 km
     - CONUS, 6-hour files, June 2019–present
   * - ``"hrrr.v1"``
     - :class:`arlmet.archives.HRRRv1Archive`
     - HRRR 3 km, version 1
     - CONUS, 6-hour files, June 2015–2019
   * - ``"nam12"``
     - :class:`arlmet.archives.NAMArchive`
     - NAM 12 km
     - North America, daily files
   * - ``"nams"``
     - :class:`arlmet.archives.NAMSArchive`
     - NAM hybrid sigma-pressure
     - CONUS, Alaska, or Hawaii (``domain=``), daily files
   * - ``"gdas1"``
     - :class:`arlmet.archives.GDASArchive`
     - GDAS 1 degree
     - Global, weekly files
   * - ``"gdas0p5"``
     - :class:`arlmet.archives.GDAS0p5Archive`
     - GDAS 0.5 degree
     - Global, daily files, 2007–2019
   * - ``"gfs0p25"``
     - :class:`arlmet.archives.GFSArchive`
     - GFS 0.25 degree
     - Global, daily files
   * - ``"narr"``
     - :class:`arlmet.archives.NARRArchive`
     - North American Regional Reanalysis 32 km
     - North America, monthly files, 1979–2019
   * - ``"reanalysis"``
     - :class:`arlmet.archives.ReanalysisArchive`
     - NCEP/NCAR Reanalysis 2.5 degree
     - Global, monthly files

Choose an archive by name
~~~~~~~~~~~~~~~~~~~~~~~~~

:data:`arlmet.archives.ARCHIVES` maps each name to its class, and
:func:`arlmet.archives.get_archive` builds an archive from its name plus any
options, which is convenient when the product comes from a config file:

.. code-block:: python

   from arlmet.archives import ARCHIVES, get_archive

   sorted(ARCHIVES)  # ['gdas0p5', 'gdas1', 'gfs0p25', 'hrrr', ...]
   archive = get_archive("nams", domain="ak")
   files = archive.fetch("2024-07-18", "2024-07-19", dest_dir="./met/")

An unknown name raises ``ValueError`` listing the available ones. A subclass of
:class:`arlmet.archives.Archive` that sets ``name`` is registered
automatically when it is defined, so your own archives work with
``get_archive()`` too.

Download files for a time range
-------------------------------

.. code-block:: python

   from arlmet.archives import HRRRArchive

   archive = HRRRArchive()
   files = archive.fetch(
       "2024-07-18 00:00",
       "2024-07-19 00:00",
       dest_dir="./met",
   )

``fetch()`` returns the local paths in chronological order. Duplicate archive
keys are removed automatically when the requested time range spans multiple
hours within the same ARL file.

Crop on download
----------------

For large global products, pass ``bbox=`` so the downloaded file is cropped
before it is cached locally.

.. code-block:: python

   from arlmet.archives import GFSArchive

   archive = GFSArchive()
   files = archive.fetch(
       "2024-07-18 00:00",
       "2024-07-19 00:00",
       dest_dir="./met",
       bbox=(-114.0, 39.0, -110.0, 42.0),
   )

This uses :func:`arlmet.extract_subset` internally after the raw file is
downloaded.

Pass ``levels=`` to keep only some vertical levels, counted from 0 at the
surface. It works with or without ``bbox``.

.. code-block:: python

   files = archive.fetch(
       "2024-07-18 00:00",
       "2024-07-19 00:00",
       dest_dir="./met",
       bbox=(-114.0, 39.0, -110.0, 42.0),
       levels=range(20),   # the lowest 20 levels
   )

The crop is part of each cached file's name (for example
``...crop_-114.00_39.00_-110.00_42.00.levels_0-19``), so files cropped
differently are cached separately. Bbox values are written with two decimals
unless they have more, which are kept in full (``...crop_-111.925_...``).

The full, uncropped file is downloaded into ``dest_dir`` (not the system temp
directory) and deleted once the crop is written, so ``dest_dir`` needs room
for one full file at a time (about 3 GB for HRRR).

Choose a mirror
---------------

Each archive is served from three mirrors. The default, ``"s3"``, is usually
the fastest choice.

.. code-block:: python

   files = archive.fetch(
       "2024-07-18",
       "2024-07-19",
       dest_dir="./met",
       mirror="ftp",
   )

The mirrors are:

- ``"s3"``: NOAA public S3 bucket via ``s3fs``
- ``"ftp"``: NOAA ARL FTP archive
- ``"http"``: READY web archive

Caching and overwrite behavior
------------------------------

Downloaded files are reused if a matching local file already exists. Pass
``overwrite=True`` to force a fresh download.

Files are written to a hidden ``.<name>.<random>.partial`` file first and
renamed into place only once complete, so an interrupted fetch never leaves a
truncated file that a later call would reuse. A ``.partial`` file left behind
by a killed process is never used and can be deleted.

Requesting a time range that begins before an archive's ``start_date`` (the
start of its archive) raises ``ValueError``.

.. code-block:: python

   files = archive.fetch(
       "2024-07-18",
       "2024-07-19",
       dest_dir="./met",
       overwrite=True,
   )
