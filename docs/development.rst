:orphan:

Developer notes
===============

Reference material for people changing arl-met itself. The rules a change must
keep (the ARL format rules and the design invariants) and the commands are in
`AGENTS.md <https://github.com/jmineau/arl-met/blob/main/AGENTS.md>`_; how to
send a change is in :doc:`contributing`.

HYSPLIT routines arl-met mirrors
--------------------------------

arl-met follows HYSPLIT's own reader and writer wherever the format leaves room
for choice. The Fortran routines it matches:

.. list-table::
   :header-rows: 1

   * - HYSPLIT
     - What it does
     - In arl-met
   * - ``pakout`` / ``pakinp``
     - pack and unpack one field as byte differences; compute the checksum
     - :func:`arlmet.pack`, :func:`arlmet.unpack`, :func:`arlmet.calculate_checksum`
   * - ``pakrec`` / ``pakndx``
     - write records and the index record; the checksum table is reset to 0 at
       each time step and filled only for the records packed
     - :meth:`arlmet.DataRecord.verify_checksum` treats a stored 0 as "none"
   * - ``metinp``
     - read a met file; never compares checksums
     - :meth:`arlmet.File.check` reads headers only
   * - ``metset.f``
     - read the 12 projection values raw (``12F7.0``)
     - only the longitudes are wrapped to [-180, 180]
   * - ``prfcom``
     - choose the vertical conversion from the vertical flag
     - :meth:`arlmet.VerticalAxis.from_flag` and its subclasses
   * - ``xtrct_grid.f``
     - crop a met file to a grid window
     - :meth:`arlmet.Grid.window_from_bbox`, :func:`arlmet.extract_subset`
       (``tests/test_subset_slow.py`` compares the two on real files)

Vertical-axis conversions
-------------------------

Each subclass of :class:`arlmet.VerticalAxis` converts its levels with
keyword-only inputs (``to_pressure(*, surface_pressure=None)`` and
``to_height_agl(*, surface_pressure=None, temperature=None, hgts=None,
terrain=None)``), uses only the ones its coordinate system needs, and raises
``ValueError`` naming a missing one.

.. list-table::
   :header-rows: 1

   * - Flag
     - Class
     - ``to_pressure``
     - ``to_height_agl``
   * - 1
     - :class:`~arlmet.SigmaAxis`
     - ``offset + (surface_pressure - offset) * sigma``
     - hypsometric, from ``surface_pressure`` (PRSS) and ``temperature`` (TEMP)
   * - 2
     - :class:`~arlmet.PressureAxis`
     - the stored levels
     - ``hgts - terrain`` (HGTS minus SHGT)
   * - 3
     - :class:`~arlmet.TerrainAxis`
     - raises ``ValueError`` (no pressure coordinate)
     - the stored levels (already above ground)
   * - 4
     - :class:`~arlmet.HybridAxis`
     - ``surface_pressure * sigma + floor(level)``; the first level is the surface
     - hypsometric, as for flag 1

Any other flag, including 5 (WRF), raises ``ValueError`` in ``from_flag``.

Module layout
-------------

.. code-block:: text

   src/arlmet/
     __init__.py    public exports
     file.py        File: file handle, scan of index records, check(), factories
     recordset.py   RecordSet: the records of one valid time
     record.py      DataRecord (one packed 2D field) and the _require_mode helper
     collection.py  RecordCollection protocol, VariableView, VariableAccessor
     header.py      Header: the 50-byte record header codec, grid letter helpers
     index.py       IndexRecord, VarInfo, LvlInfo: the index record codec
     grid.py        Projection, Grid, GridWindow: horizontal geometry only
     vertical.py    VerticalAxis (abstract) and SigmaAxis, PressureAxis,
                    TerrainAxis, HybridAxis
     packing.py     pack(), unpack(), calculate_checksum()
     _pack.c        C extension arlmet._pack: the packing encoder (pack_core)
     exceptions.py  ARLFormatError, ARLFormatWarning
     _io.py         atomic_output(), reject_same_file(): safe output paths
     _time.py       ensure_timestamp()
     archives.py    Archive, HRRRArchive, NAMArchive, GDASArchive, ...;
                    ARCHIVES registry, get_archive(): NOAA archive downloads
     ops/
       subset.py    extract_subset(), resolve_window(), normalize_levels()
       sample.py    sample_points()
       concat.py    concat(), concat_by_time()
     xarray/
       dataset.py     open_dataset(), write_dataset(): the flat Dataset path
       _entrypoint.py ARLBackendEntrypoint: xr.open_dataset(path, engine="arl")
       _backend.py    ArlVariableArray (xarray BackendArray)
       _coords.py     the arl_grid coordinate codec, vertical-axis coordinates
       _accessor.py   ds.arl.grid, .vertical_axis, .source
       _vertical.py   pressure(), z_agl(), z_msl()

Module imports
--------------

Runtime imports between modules (``TYPE_CHECKING`` imports left out; *lazy*
means inside a function body):

.. code-block:: text

   vertical, exceptions, _io, _time        no arlmet imports
   grid           (pyproj)
   packing      → _pack (C extension), grid
   header       → _time, exceptions, grid
   index        → _time, exceptions, grid, header, vertical
   record       → exceptions, grid, header, packing, vertical; lazy: xarray
   collection   → grid, record, vertical
   recordset    → collection, grid, header, index, record, vertical
   file         → _time, collection, exceptions, grid, header, index, record,
                  recordset, vertical, xarray.backends.CachingFileManager;
                  lazy: ops.sample, ops.subset, xarray.dataset
   ops.subset   → _io, exceptions, file, grid, header, index, packing, record, vertical
   ops.sample   → _time, grid, vertical; lazy: file
   ops.concat   → exceptions, file
   xarray.*     → _io, _time, grid, ops.subset, vertical; lazy: file
   archives     → _time; lazy: fsspec, ops.subset

``file`` imports the operations and the Dataset builder lazily because they
import ``file``. ``archives`` is not imported by ``import arlmet``, and imports
``fsspec`` only when it downloads.

Memory tests
------------

``tests/test_memory.py`` measures the writers with ``tracemalloc``: the peak
during :func:`~arlmet.extract_subset` (both the byte-copy path, without a bbox,
and the repack path, with one) and :func:`~arlmet.write_dataset` (including from
a lazy Dataset), and what is still held after they return with the garbage
collector disabled. Its grids are about 200x200, so array buffers dominate Python
object overhead.

Python sizes each open file's buffer from the filesystem's ``st_blksize``. On a
filesystem with a large block size (some network filesystems report 128 KB,
where a local disk reports 4 KB) each open file adds that much, and three of
these tests fail. Run them with the temporary directory on a local disk.

Benchmarks
----------

Use real files from the archives for performance work, for example the July 2024
NCEP/NCAR Reanalysis month:

.. code-block:: python

   from arlmet.archives import get_archive

   get_archive("reanalysis").fetch("2024-07-01", "2024-07-01", dest_dir="met/")

This gives ``RP202407.gbl`` (73x144, 124 times, 18 levels). Measured on one
development machine, :func:`~arlmet.extract_subset` took about 2.1 s for
``bbox=(-170, 10, -40, 70)`` (repack path) and about 0.3 s for
``levels=[0, 1, 2, 3, 4]`` without a bbox (byte-copy path). A 614x428 NAM12 day
(``get_archive("nam12")``) took about 0.27 s for the same levels.

Format references
-----------------

- `HYSPLIT meteorology files <https://www.ready.noaa.gov/hysplitusersguide/S141.htm>`_:
  the ARL format overview.
- `HYSPLIT CONTROL file <https://www.ready.noaa.gov/hysplitusersguide/S310.htm>`_,
  the "number of input data grids" line: one number ``N`` makes each file its own
  grid; two numbers ``1 N`` are one grid with ``N`` files in time.
- `HYSPLIT compilation limits <https://www.ready.noaa.gov/hysplitusersguide/S441.htm>`_:
  at most 12 met files with the one-number form and 128 files per grid with
  ``1 N``, the reason :func:`~arlmet.concat` and :func:`~arlmet.concat_by_time`
  exist.
- `GDAS1 packing notes <https://www.ready.noaa.gov/gdas1.php>`_ and the
  `READY archive catalog <https://www.ready.noaa.gov/archives.php>`_.
- `ARLreader's notes on the format
  <https://github.com/martin-rdz/ARLreader/blob/master/working_with_ARLformat.md>`_.
