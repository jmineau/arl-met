Downloading Archived Meteorology
================================

arl-met includes archive classes for downloading ARL meteorology files from the
NOAA ARL public archives.

Install the archive dependencies first:

.. code-block:: bash

   pip install "arlmet[archives]"

The examples on this page download files from NOAA, so they are shown but not run
when the docs are built, except the one that only picks an archive by name.

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

.. ipython:: python

   from arlmet.archives import ARCHIVES, get_archive

   sorted(ARCHIVES)
   archive = get_archive("nams", domain="ak")
   archive

.. code-block:: python

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

Several callers can fetch into one folder at once: threads, processes such as
PYSTILT's workers, or jobs on other nodes that share the folder. Each file is
downloaded once. The first caller to reach a file creates a hidden
``.<name>.lock`` file beside it and keeps touching it while it downloads; the
others wait until the finished file appears and use it. The lock is removed
when the download ends, also after an error or ``Ctrl-C``. If its holder is
killed without a chance to clean up (``kill -9``, a lost node), nobody waits
forever: a lock untouched for five minutes is taken over, and a caller that has
waited four hours downloads the file itself. With ``overwrite=True`` a caller
waits for a download in progress, then downloads again. The lock saves
duplicate work only; a file is always renamed into place whole, so a lost
race costs a second download and never a damaged file. A ``.lock`` file left
by a killed process is harmless and can be deleted.

Requesting a time range that begins before an archive's ``start_date`` (the
start of its archive) raises ``ValueError``.

.. code-block:: python

   files = archive.fetch(
       "2024-07-18",
       "2024-07-19",
       dest_dir="./met",
       overwrite=True,
   )

Keep a local mirror in step with NOAA
-------------------------------------

``fetch()`` gets the files for a time range. :meth:`~arlmet.archives.Archive.sync`
keeps a whole folder complete and checked, so a daily job can replace a
mirroring tool such as ``lftp mirror``:

.. code-block:: python

   from arlmet.archives import HRRRArchive

   result = HRRRArchive().sync("/data/hrrr")
   result.downloaded   # files fetched this run
   result.mismatched   # local files whose MD5 differs from NOAA's: reported, left alone

NOAA publishes ``listing.md5.txt``, one ``md5  name`` line per file, at the top
of every product on the S3 and FTP mirrors, and rewrites it when a new file is
complete and when NOAA corrects one. ``sync()`` reads it and, for each file in
the archive's range:

- **No local copy:** downloads it to a hidden ``.partial`` file, computes its
  MD5 as it streams, and renames it into place only if it matches NOAA's.
  Otherwise it deletes it and lists the file in ``result.failed``.
- **A copy the manifest records as NOAA's:** does nothing, and does not read the
  file.
- **A copy the manifest records as differing from NOAA's:** lists it in
  ``result.mismatched``, without reading it, and leaves it alone. It stays
  there on every run until NOAA's file changes.
- **A copy NOAA has corrected** (NOAA's MD5 is not the one recorded when the
  copy was checked): downloads it again the same way and replaces it where it
  is, whether the old copy matched or not.
- **A copy the manifest does not list** (a folder filled another way), **or
  whose size or modification time changed since it was recorded:** computes its
  MD5 and records it. If it is NOAA's it is in ``result.adopted``; if not, in
  ``result.mismatched``, and left alone.

Nothing is deleted. Files NOAA does not list, including the ``name~date~``
backups ``lftp`` leaves, are not touched.

The folder is the product's own (for HRRR, the one that holds ``2024/07/...``),
and new files go where :meth:`~arlmet.archives.Archive.paths_for_range` puts
them. A local copy is found by its *name*, anywhere under the folder, so a
folder in another layout (all the HRRR v1 files in one folder, say) is
checked and recorded, never downloaded again. Run ``dry_run=True`` first on
a folder you care about: it reads nothing and writes nothing, and
``result.downloaded`` shows what a real run would fetch.

.. code-block:: python

   # Which files would be fetched, without reading or writing anything?
   result = HRRRArchive().sync("/data/hrrr", dry_run=True)
   len(result.downloaded), len(result.adopted)

   # Only part of the archive
   HRRRArchive().sync("/data/hrrr", start="2024-07-01", end="2024-07-31")

The manifest
~~~~~~~~~~~~

``arlmet-manifest.tsv`` in the folder keeps apart what each local file is and
what NOAA's copy was when it was checked, as ``rsync`` does with its file
list. It is a plain text file with one line per file and five fields separated
by tabs: the local file's MD5, NOAA's MD5 when it was checked, the file's size in
bytes, its modification time in nanoseconds, and its path relative to the folder
(last, so it may contain spaces). For example (tabs shown as spaces):

.. code-block:: text

   0f4c...e2a1   0f4c...e2a1   3377102848   1718150400000000000   2024/07/20240718_00-05_hrrr

A run reads a file only if it is not in the manifest or its size or modification
time differ from the line, and then updates the line. A file whose MD5 differs
from NOAA's is recorded too, so it is reported on every run without being read;
it is replaced only when NOAA's MD5 changes. To have a file checked again,
delete its line. The manifest is written as the run goes, so a run that is
killed keeps what it checked (the first run over a big mirror reads every file
once, which takes a while).

One run at a time
~~~~~~~~~~~~~~~~~

A folder ``.arlmet-sync.lock`` made with ``mkdir`` (which is atomic also on
NFS, where a ``flock`` may not be seen by another host) stops a second run
with a ``RuntimeError`` naming the host and start time of the first. It is
removed at the end of a run, also after an error. A run that is killed without
a chance to clean up (``kill -9``, a lost node) leaves it; once you are sure no
sync is running, delete the folder.

A file that cannot be fetched, or whose download does not match NOAA's MD5, is
logged and listed in ``result.failed``, and the run goes on; the next run tries
it again. Turn on logging (``logging.basicConfig(level=logging.INFO)``) to see
each file as it is fetched.
