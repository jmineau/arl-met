Cataloging a folder
===================

Use :func:`arlmet.catalog` to find the ARL files that cover a time window by what
their headers say, not by how they are named. Each file's index records hold its
valid times, the model that wrote it (the source), and its grid. ``catalog`` reads
them once for a whole folder and keeps them in a small table.
:meth:`~arlmet.Catalog.covering` then returns the files for a window, and
:meth:`~arlmet.Catalog.missing` the hours no file holds.

.. code-block:: python

   import arlmet

   cat = arlmet.catalog("hrrr")
   files = cat.covering("2021-01-14 06:00", "2021-01-15 06:00")

A program that finds files by a name pattern (``%Y%m%d_%H``) finds nothing
when the pattern is wrong or the archive is renamed, and cannot tell two
products named alike apart. The catalog needs only the folder.

An example
----------

The examples on this page use four small files written with
:func:`arlmet.write_dataset`: six hourly time steps each, from 2024-07-01 00:00
to 23:00, named after their first hour. Any names would do.

.. ipython:: python

   import tempfile
   from pathlib import Path

   import numpy as np
   import pandas as pd
   import xarray as xr

   import arlmet

   folder = Path(tempfile.mkdtemp()) / "met"
   folder.mkdir()
   grid = xr.Variable((), 0, attrs={
       "nx": 20, "ny": 20, "pole_lat": 90.0, "pole_lon": 0.0,
       "tangent_lat": 1.0, "tangent_lon": 1.0, "grid_size": 0.0,
       "orientation": 0.0, "cone_angle": 0.0, "sync_x": 1.0, "sync_y": 1.0,
       "sync_lat": 30.0, "sync_lon": -120.0,
   })
   surface = arlmet.SigmaAxis(levels=[1.0])
   day = xr.Dataset(
       {"PRSS": (("time", "lat", "lon"), np.full((24, 20, 20), 850.0, "float32"))},
       coords={"time": pd.date_range("2024-07-01", periods=24, freq="h"), "arl_grid": grid},
       attrs={"source": "TEST"},
   )
   for first in range(0, 24, 6):
       block = day.isel(time=slice(first, first + 6))
       arlmet.write_dataset(block, folder / f"20240701_{first:02d}", vertical_axis=surface)

   cat = arlmet.catalog(folder)
   cat.table[["source", "start", "end", "step"]]

:attr:`~arlmet.Catalog.table` has one row per file: its ``path``, its
``source``, a short description of its ``grid``, its first and last valid time
(``start`` and ``end``), the time between its valid times (``step``), and an
``error`` column, empty for a file that can be used.

Finding the files for a window
------------------------------

HYSPLIT interpolates between the valid times around each moment of a run, so a
window needs every valid time from the one at or before its start to the one at
or after its end. :meth:`~arlmet.Catalog.covering` returns the files that
hold them, in time order:

.. ipython:: python

   [p.name for p in cat.covering("2024-07-01 05:30", "2024-07-01 12:00")]

The window starts at 05:30, so it needs 05:00, which is in the first file, and
it ends at 12:00, which is in the third. A window from 06:00 to 11:00 needs only
the second file. A time without a time zone is UTC, and one with a time zone
is converted to UTC, as in the archives.

The valid times are expected every :attr:`~arlmet.Catalog.step`, the
most common spacing between the valid times of the folder. When a file is
missing, :meth:`~arlmet.Catalog.missing` lists the times no file holds,
and ``covering`` raises ``ValueError`` with them:

.. ipython:: python

   (folder / "20240701_06").unlink()
   cat = arlmet.catalog(folder)
   cat.missing("2024-07-01 05:30", "2024-07-01 08:00")

.. ipython:: python

   try:
       cat.covering("2024-07-01 05:30", "2024-07-01 08:00")
   except ValueError as exc:
       print(exc)


Times past the last file or before the first count as missing in the same way,
one every step.

To use what the folder has and find the rest elsewhere, such as crops that
hold part of a run beside an archive that holds all of it, pass
``missing_ok=True``. ``covering`` then returns the files for the times that are
held, and ``missing`` lists the others:

.. ipython:: python

   [p.name for p in cat.covering("2024-07-01 05:30", "2024-07-01 12:00", missing_ok=True)]

When two files hold the same valid time (a re-download kept under a new name,
an older copy kept beside a newer one), ``covering`` raises ``ValueError`` and
names both files, rather than choosing one. Move one of them out of the folder.
``missing`` does not list such a time: it is held, twice. The one exception is
a file whose every valid time a longer file of the same source and grid also
holds, such as a few hours cut from a file and kept beside it: the longer file
is used and the short one is left out.

Which files are read
--------------------

- **The folder's own files, or with** ``recursive=True`` **every folder below
  it.** Archives come in two layouts. A plain copy of NOAA's archive keeps
  files in ``YYYY/MM/`` folders (``hrrr/2021/01/20210114_00-05_hrrr``):
  take its catalog with ``recursive=True``. A shared archive often keeps one flat
  folder of links into such a tree: take that folder as it is, without
  ``recursive``.
- **Links to files are followed, and each real file is one row**, shown by its
  real path, so a link and the file it points to, or two links to one file,
  never count as two files. A link whose target is gone, or a loop of links, is
  a row with the reason in ``error``. Links to folders are not followed.
- **Subfolders are read in full with** ``recursive=True``, **old copies
  included.** An archive that keeps superseded files in a subfolder such as
  ``deprecated/`` holds two files for those hours, and ``covering`` raises for
  them. Take the folder of live files instead (a year, or the flat folder of
  links), or the archive without ``recursive``.
- **Hidden files and folders, backup copies and copies set aside are
  skipped**: names that start with ``.`` (arl-met's own partial outputs and the
  cache file), end in ``~``, ``.partial``, ``.bad`` or ``.lock``, or hold ``.lock.``. An
  archive that keeps a damaged file as ``name.bad`` beside its replacement is
  read as if only the replacement were there.
- **Every other file is kept in the table.** One that is not ARL, is cut short,
  holds no time step, or whose time steps are not evenly spaced (HYSPLIT stops
  where the spacing changes) gets the reason in ``error``, and ``covering`` does
  not use it:

  .. code-block:: python

     bad = cat.table[cat.table["error"] != ""]

  A folder kept by :meth:`~arlmet.archives.Archive.sync` holds
  ``arlmet-manifest.tsv`` (see :doc:`downloading`), which is not ARL, so it
  appears here with an ``error``; ``covering`` ignores it.

- **Opening a file is all that is checked.** A time step written partway, or a
  record overwritten with other bytes, opens cleanly. Use
  :meth:`arlmet.File.check` on the files ``covering`` returns to find those
  (see :doc:`checking`).

Opening a file may warn: a time step that some NOAA HRRR files repeat byte for
byte is read once, with an :class:`arlmet.ARLFormatWarning`.

The cache
---------

The table is saved to a CSV file, so the files are opened only once. A later
call to ``catalog`` lists the folder, reads again only the files that
are new or whose size or modification time changed, and drops the files that
are gone. :attr:`~arlmet.Catalog.cache` says where it is:

- ``.arlmet-catalog.csv`` in the folder itself, when you can write to it
  (``.arlmet-catalog-recursive.csv`` for a recursive catalog);
- otherwise, as for a shared read-only archive, a file named after the folder
  in ``$ARLMET_CACHE_DIR``, or in the user cache folder when that variable is
  not set: ``$XDG_CACHE_HOME/arlmet/catalog`` (``~/.cache/arlmet/catalog``) on
  Linux, ``~/Library/Caches/arlmet/catalog`` on macOS,
  ``%LOCALAPPDATA%\arlmet\catalog`` on Windows.

When the cache is not in the folder and the folder holds
``.arlmet-catalog.csv`` (or ``-recursive``), that table is read, never
changed, as the starting point: files whose size and modification time match
its rows are not opened again, and only the others are read before the full table
is saved to your own cache. Rows in your own cache win over it. Whoever syncs a
shared archive keeps this table current by running ``arlmet.catalog(folder)``
as the last step of the sync. Rows are keyed by real path, so it helps readers
who see the archive at the same real path.

Pass ``cache_dir=`` to choose the folder yourself:

.. code-block:: python

   cat = arlmet.catalog("/archive/hrrr", cache_dir="caches")

How long it takes
-----------------

Opening a file reads one index record per time step, and files are opened 16 at
a time, since on a network filesystem each one mostly waits on its reads. The
first call on a large archive is slow. On two archives of about 16,000 six-hour
HRRR files it took 10 minutes on a fast filesystem and 44 minutes on a busy shared
one (where one file at a time would have taken about 11 hours). A later call only
lists the folder and compares sizes and times: 4 to 34 seconds there. The cache
is saved every 500 files, so an interrupted first call resumes where it stopped.

Run ``catalog`` once before starting many jobs on the same folder, such as a
Slurm job array. Jobs that start together on a folder with no cache do not
each open every file: the first opens them and the others wait for its table,
so the whole array waits as long as one first call. With the cache written
first, each job only lists the folder.
