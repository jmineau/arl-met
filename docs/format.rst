ARL File Format
===============

This page describes the ARL (Air Resources Laboratory) packed data format used by HYSPLIT and related atmospheric transport models.

Overview
--------

The ARL packed data format is a hierarchical binary structure designed for efficient storage and retrieval of meteorological data.

Useful references
-----------------

- `HYSPLIT meteorology page <https://www.ready.noaa.gov/hysplitusersguide/S141.htm>`_ for the ARL meteorology format overview.
- `GDAS1 packing notes <https://www.ready.noaa.gov/gdas1.php>`_ for a concrete example of ARL packing behavior.

File Structure
--------------

The ARL file format consists of sequential record sets, each representing one forecast time::

   ARL FILE
   ├── RecordSet 1 (Time 1)
   │   ├── INDEX RECORD (metadata, grid definition, variable catalog)
   │   ├── SURFACE LEVEL (Level 0)
   │   │   ├── Variable 1: PRSS (50-byte header + packed data)
   │   │   ├── Variable 2: T02M (50-byte header + packed data)
   │   │   ├── Variable 3: TPP1 (50-byte header + packed data)
   │   │   └── ... (all surface variables)
   │   │
   │   ├── UPPER LEVEL 1 (e.g., 1000 mb)
   │   │   ├── Variable 1: UWND (50-byte header + packed data)
   │   │   ├── Variable 2: VWND (50-byte header + packed data)
   │   │   ├── Variable 3: TEMP (50-byte header + packed data)
   │   │   └── ... (all variables for this level)
   │   │
   │   ├── UPPER LEVEL 2 (e.g., 925 mb)
   │   └── ... (continues for all vertical levels)
   │
   ├── RecordSet 2
   │   ├── Index Record
   │   ├── Data Record(s)
   │
   └── ... (additional record sets for other grids or time periods)

Key Components
--------------

1. File Level Structure
~~~~~~~~~~~~~~~~~~~~~~~

- **Sequential Time Periods**: Each represents one forecast time
- **Direct Access**: Fixed record length allows random access
- **Platform Independent**: Binary format works across systems

2. Record Set Structure
~~~~~~~~~~~~~~~~~~~~~~~

Each record set contains:

- **Index Record**: Metadata for entire time period, grid definition, variable catalog
- **Data Records**: One record per variable per level
  - Surface data first (Level 0)
  - Then upper levels (1 to nz-1)

Differential records
~~~~~~~~~~~~~~~~~~~~

Some ARL streams include ``DIF*`` records. These are separate on-disk records
that correct a data record of the same time and level, their parent: a DIF
record holds what packing lost from its parent (the original values minus the
packed ones). NOAA's HRRR files have one for the precipitation (``DIFR`` after
``TPP1``) and one for the vertical velocity on every level (``DIFW`` after
``WWND``).

A DIF record is paired with its parent by name, as HYSPLIT's reader
(``metinp``) pairs it:

.. list-table::
   :header-rows: 1

   * - DIF record
     - Level
     - Parent
   * - ``DIFW``
     - above the surface
     - ``WWND`` or ``DZDT``
   * - ``DIFT``
     - above the surface
     - ``TKEN`` or ``TKEU``
   * - any ``DIF*``
     - surface (level 0)
     - ``TPP*`` or ``PRT*``

The parent is the nearest record before the DIF on its level that the table
allows, whatever lies between them: HYSPLIT adds the DIF into a field that holds
the last such record it read. NOAA's ``era52arl`` lists ``DIFW`` and ``DIFR``
last on each level, after ``RELH`` and ``USTR``, and they still correct ``WWND``
and ``TPP1``. A DIF record with another name above the surface, or with no
parent before it on its level, is applied to nothing, as HYSPLIT skips it, and
opening the file warns where it is (:class:`arlmet.ARLFormatWarning`).

A parent can have several DIF records in a row (``WWND``, ``DIFW``, ``DIFW``).
HYSPLIT's reader adds each one it reads into its field, one after another, so
arl-met reads them the same way: the parent's value is
``(parent + DIF) + DIF``, added in stream order in ``float32``. No producer we
know of writes more than one. When one does, its index record may list the same
``DIF*`` name more than once on a level. A level that lists a variable other
than a ``DIF*`` twice is malformed (:class:`arlmet.ARLFormatError`).

arl-met preserves that stream structure in the low-level API. When reading a
parent record through :class:`arlmet.DataRecord`, the returned values are the
parent plus each of its :attr:`~arlmet.DataRecord.diffs`. An uncropped
:func:`~arlmet.extract_subset` copies every DIF record's bytes; a crop or a
rewrite packs the combined values again, writing the same DIF records after
each parent, whose values agree with the source's within the DIFs' precision.
Both write each DIF record right after its parent, so a DIF that the source
lists later on its level (as ``era52arl`` does) moves up, and the index record
lists the new order; a DIF record that corrects nothing is left out. A subset
that leaves out level 0 makes its lowest level the surface, where a ``DIFW`` or
``DIFT`` would correct nothing, so :func:`~arlmet.extract_subset` refuses one
that would move a DIF record there (HYSPLIT's ``xtrct_grid`` always keeps
level 0).

When arl-met generates a DIF record while writing, it follows the producer-side
HYSPLIT pattern:

.. code-block:: text

   write parent -> unpack packed parent -> diff = target - unpacked_parent -> write DIF

With several DIF names, each DIF is the target minus the unpacked parent and
the unpacked DIF records before it. The writer refuses a DIF name the table
does not pair with the parent at its level (``ValueError``), so what it writes
reads back with the same pairs.

3. Individual Data Record Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each data record consists of::

   ┌──────────────────────┬──────────────────────────────┐
   │ HEADER (50 bytes)    │ PACKED DATA (nx × ny bytes)  │
   │ - Variable ID (4)    │ - One byte per grid point    │
   │ - Date/Time          │ - Difference encoding        │
   │ - Level indicator    │ - Precision maintained       │
   │ - Packing scale      │                              │
   └──────────────────────┴──────────────────────────────┘

4. Packing Algorithm
~~~~~~~~~~~~~~~~~~~~

The packing process::

   Original Data → Difference Encoding → Scale/Quantize → Pack to bytes
       ↓                    ↓                  ↓              ↓
   [Real Array]    [Δ from neighbors]    [Scale to 0-255]  [1 byte/point]

Reading a record reverses this: the first column is rebuilt down the rows from
the header's initial value, then each row across from its first cell, adding
one difference at a time in single precision. A value smaller in magnitude than
the record's precision (``2**exponent / 254``) is read as zero, except in the
first column, which HYSPLIT's ``pakinp`` leaves as it is. :func:`arlmet.unpack`
follows ``pakinp`` step for step and gives the same ``float32`` values.

Writing is HYSPLIT's ``pakout``: the exponent is the smallest that holds the
largest difference between neighbors, each byte is the difference from the
previous value as a reader will rebuild it, and values are packed as they are
(the initial value is the first cell's, even when it is below the precision).
:func:`arlmet.pack` gives the same bytes, exponent, precision and initial value.

Malformed files
~~~~~~~~~~~~~~~

Opening a file scans its index records. :class:`arlmet.ARLFormatError` (a
``ValueError`` subclass) is raised when the content is not valid ARL:

- a record header or index record cannot be parsed (e.g. a non-ARL file)
- the file size is not a whole number of records, or an index record declares
  more data records than remain in the file (a truncated download)
- index records disagree on source, grid, or vertical axis
- a ``DIF*`` record has no parent before it on its level
- a level lists a variable other than a ``DIF*`` record twice
- a valid time is repeated with different content

A file can also be damaged in ways that open cleanly, such as a time step
written partway. :meth:`arlmet.File.check` finds them (:doc:`checking`).

Some NOAA archive files (e.g. HRRR) repeat a whole time step, index record and
data records, byte for byte. arl-met keeps the first copy, ignores the repeat,
and emits an :class:`arlmet.ARLFormatWarning`. Rewriting such a file (e.g. with
:func:`arlmet.extract_subset`) produces a clean file without the repeat.

Missing records
~~~~~~~~~~~~~~~

A data record whose header has a negative forecast hour, or the name ``NULL``,
is flagged missing. HYSPLIT's writer starts every record of a time step that
way (``pakini.f``: forecast hour -1, no precision, exponent or first value, the
packed values spaces) before a converter fills it, so a field the converter
never writes stays flagged. HYSPLIT's reader (``metinp.f``) skips such a record
with a warning, and stops when a time step has more than 20. This is the only
use HYSPLIT makes of a data record's forecast hour: the forecast hour it carries
is the index record's, which is why :func:`arlmet.open_dataset` keeps only that
one, as the ``forecast_hour`` coordinate.

arl-met reads a missing record as NaN, through :meth:`arlmet.DataRecord.read`,
:func:`arlmet.open_dataset` and :func:`arlmet.sample_points`; a missing
``DIF*`` record adds nothing to its parent, as in HYSPLIT.
:meth:`arlmet.File.check` lists missing records. A field that is NaN
everywhere is written as a missing record, as ``pakini.f`` leaves one, so
:func:`arlmet.extract_subset` keeps a missing record missing.
:func:`arlmet.write_dataset` writes an all-NaN slice as a missing record when
that variable and level hold data at another time step, so it keeps a missing
record missing too; a variable and level that are NaN at every time step get no
record.

Checksums
~~~~~~~~~

Each time step's index record holds a checksum for each data record: the sum of
the record's packed bytes, less 255 each time the sum reaches 256
(:func:`arlmet.calculate_checksum`). :meth:`arlmet.DataRecord.verify_checksum`
compares a record's packed bytes with it.

A stored checksum of 0 means none was recorded, and ``verify_checksum()``
returns ``True`` for it. HYSPLIT's writer fills in a checksum only for the
records it packs in a time step, and HYSPLIT does not compare checksums when it
reads a file. NOAA's GDAS and NAM12 files store 0 for the precipitation and
fluxes taken from a 6-hour forecast, at 00, 06, 12 and 18 UTC; their values are
good.

Variable Organization
---------------------

Surface Variables (Level 0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Pressure fields**: PRSS, MSLP
- **Temperature**: T02M, TMPS
- **Winds**: U10M, V10M
- **Fluxes**: SHTF, LTHF, USTR
- **Precipitation**: TPP1, RAINC, RAINNC

Upper-Air Variables (Levels 1-nz)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Winds**: UWND, VWND, WWND
- **Temperature**: TEMP
- **Heights**: HGTS
- **Moisture**: RELH, SPHU
- **Special**: TKEN, DIFT, DIFW

The presence of a ``DIF*`` record is stream-dependent. Do not assume every
variable has one or that every ``DIF*`` name is universally valid outside the
source workflow that produced it.

Key Features
------------

This hierarchical structure provides:

- **Efficient Storage**: 1 byte per grid point with maintained precision
- **Fast Access**: Direct access to any variable at any level/time
- **Flexibility**: Support for multiple grids and coordinate systems
- **Portability**: Platform-independent binary format
- **Extensibility**: New variables can be added via configuration files

Summary
-------

The key insight is that this is essentially a database structure optimized for meteorological data, where the index records serve as the catalog and the data records are stored in a highly compressed but quickly accessible format.

For more details on how arl-met reads and processes these files, see the :doc:`api` documentation.
