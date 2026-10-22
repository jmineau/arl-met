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
     - read and written raw too; only :meth:`arlmet.Grid.calculate_coords`
       wraps longitudes
   * - ``gblset`` / ``gbl2xy``
     - place a longitude on a lat/lon grid: in 0-360, measured from
       ``sync_lon``, unless ``sync_lon < 0 <= pole_lon`` (``PRIME``)
     - :meth:`arlmet.Grid.subset` writes a crop as ``xtrct_grid`` does:
       ``sync_*`` its first cell with ``sync_lon`` in 0-360, ``pole_*`` its last
   * - ``stlmbr`` / ``stcm1p`` (set up in ``metini.f``)
     - the map projection of a projected grid: ``cone_angle`` alone picks
       polar stereographic (±90), Mercator (0) or a Lambert cone tangent there;
       ``grid_size`` holds at the reference latitude, ``tangent_lat``
     - :attr:`arlmet.Projection.params`: ``lat_ts``, or a Lambert ``k_0`` that
       makes the scale 1 at ``tangent_lat``; agrees with the compiled routines
       within 0.001 cells on HRRR, NAM, NARR, NAMS and WRF files
   * - ``prfcom``
     - choose the vertical conversion from the vertical flag
     - :meth:`arlmet.VerticalAxis.from_flag` and its subclasses
   * - ``prfsig`` / ``prfecm``
     - the sigma (flag 1, and flag 5 without WRF's hybrid coefficients) and
       hybrid (flag 4) profiles: heights from the ground
       (``PRSS``, ``T02M``) with the virtual temperature; ``prfsig`` takes each
       level's pressure from ``PRES`` when the file has it (``ZFLG``), ``prfecm``
       always from the levels; ``ln p`` linear in height
     - the column in ``ops/sample.py`` and :func:`arlmet.z_agl`, through
       ``vertical._hysplit_z_agl``; on two HRRR columns it agrees with the
       compiled ``prfsig`` within 0.006 hPa
   * - ``prfprs``
     - the pressure-level (flag 2) profile: heights from ``HGTS``, the ground
       (``SHGT``, ``PRSS``, ``T02M``, ``U10M``) at the bottom, levels at or
       below the ground skipped; ``ln p`` linear in height
     - the column in ``ops/sample.py``; ``TEMP`` and winds agree with the
       compiled ``prfprs`` within 0.001 on NAM12 and GDAS columns, pressure
       within 0.0006 hPa (float32 rounding of ``ln p``)
   * - ``prfwrf``
     - the WRF (flag 5) profile: pressure from the ``PRES`` field, heights from
       WRF's hybrid coefficients (read from a side file, ``hybrid_wrfvcoords.txt``)
     - not used: the coefficients are not in the ARL file, so
       :class:`arlmet.WrfAxis` files take the ``prfsig`` path, pressure from
       ``PRES``
   * - ``xtrct_grid.f``
     - crop a met file to a grid window
     - :meth:`arlmet.Grid.window_from_bbox`, :func:`arlmet.extract_subset`
       (``tests/test_subset_slow.py`` compares the two on real files). The same
       cells, but a projected crop keeps the source's sync point, where
       ``xtrct_grid`` writes the corner cell's latitude and longitude to 0.01
       degree and so moves its crop by up to 0.15 cells (:meth:`arlmet.Grid.subset`)

Vertical-axis conversions
-------------------------

Each subclass of :class:`arlmet.VerticalAxis` converts its levels with
keyword-only inputs (``to_pressure(*, surface_pressure=None)`` and
``to_height_agl(*, surface_pressure=None, temperature=None, pressure=None,
hgts=None, terrain=None)``), uses only the ones its coordinate system needs, and
raises ``ValueError`` naming a missing one.

.. list-table::
   :header-rows: 1

   * - Flag
     - Class
     - ``to_pressure``
     - ``to_height_agl``
   * - 1
     - :class:`~arlmet.SigmaAxis`
     - ``offset + (surface_pressure - offset) * sigma`` (nominal; a file's
       ``PRES`` is the real pressure)
     - hypsometric, from ``surface_pressure`` (PRSS) through ``pressure`` (PRES)
       when given, else the nominal pressure, with ``temperature``
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
   * - 5
     - :class:`~arlmet.WrfAxis`
     - raises ``ValueError`` (needs WRF's hybrid coefficients; use ``PRES``)
     - hypsometric, as for flag 1 with ``pressure`` (PRES); the ground
       temperature is the first level's brought down from its eta

Any other flag raises ``ValueError`` in ``from_flag``.

Flag 5 is what HYSPLIT's ``arw2arl`` writes for a WRF run with
``hybrid_opt=2``: the levels are WRF's mass-level eta (``ZNU``) with 1.0 for
the surface, and the offset is WRF's model top ``P_TOP`` in hPa. HYSPLIT's
``prfcom`` sends a flag-5 file to ``prfwrf`` only when the file has ``MUBA`` and
``MUPE`` and ``hybrid_wrfvcoords.txt`` sits beside it; otherwise ``metini``
treats it as flag 1 and ``prfsig`` integrates heights between the ``PRES``
levels. ``WrfAxis`` files follow that ``prfsig`` path exactly as sigma files
with ``PRES`` do (``vertical._hysplit_z_agl`` and the column in
``ops/sample.py``); on two HRRR columns written as flag-5 files they agree with
the compiled ``prfsig`` within 0.007 hPa.

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
     indexrecord.py IndexRecord, VarInfo, LvlInfo: the index record codec
     grid.py        Projection, Grid, GridWindow: horizontal geometry only
     vertical.py    VerticalAxis (abstract) and SigmaAxis, PressureAxis,
                    TerrainAxis, HybridAxis, WrfAxis
     packing.py     pack(), unpack(), calculate_checksum()
     _pack.c        C extension arlmet._pack: pack_core, HYSPLIT's pakout
     exceptions.py  ARLFormatError, ARLFormatWarning
     _io.py         atomic_output(), reject_same_file(): safe output paths
     _time.py       ensure_timestamp()
     archives.py    Archive, HRRRArchive, NAMArchive, GDASArchive, ...;
                    ARCHIVES registry, get_archive(): NOAA archive downloads
     directoryindex.py index(), DirectoryIndex: a folder's files by their headers
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

Examples in the docs
--------------------

The user guides run their examples when the docs are built (``just build-docs``,
as CI does), so they show real output and the build fails when one breaks. There
is no sample ARL file in the repository, and an example never downloads one, so
``docs/_ext/docs_examples.py`` writes small files of made-up values with
:func:`arlmet.write_dataset`. A page that reads a file starts with a block that
is run but not shown:

.. code-block:: rst

   .. ipython:: python
      :suppress:

      import os

      import docs_examples

      docs_cwd = docs_examples.temp_workdir()
      docs_examples.sample_file("met.arl")

``temp_workdir()`` changes to a new temporary folder, so the examples can write
files with short names, and the page ends with another such block,
``os.chdir(docs_cwd)``. ``sample_file()`` takes the valid times, and
``upper=False`` for surface variables only or ``projected=True`` for a Lambert
grid. A few things to know:

- An expression is echoed only at the top level of a block: ``print`` one inside
  a ``with`` or ``for``.
- A block that raises or warns fails the build, unless it has ``:okexcept:`` or
  ``:okwarning:``.
- All pages share one session, so each page imports what it uses.
- An example that downloads stays a plain ``code-block``, and its page says so.

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
