ARL File Format
===============

This page describes the ARL (Air Resources Laboratory) packed data format used by HYSPLIT and related atmospheric transport models.

Overview
--------

The ARL packed data format is a hierarchical binary structure designed for efficient storage and retrieval of meteorological data.

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

3. Individual Data Record Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each data record consists of::

   ┌──────────────────────┬──────────────────────────────┐
   │ HEADER (50 bytes)    │ PACKED DATA (nx × ny bytes)  │
   │ - Variable ID (4)    │ - One byte per grid point    │
   │ - Date/Time          │ - Difference encoding        │
   │ - Level indicator    │ - Precision maintained       │
   │ - Checksum info      │                              │
   └──────────────────────┴──────────────────────────────┘

4. Packing Algorithm
~~~~~~~~~~~~~~~~~~~~

The packing process::

   Original Data → Difference Encoding → Scale/Quantize → Pack to bytes
       ↓                    ↓                  ↓              ↓
   [Real Array]    [Δ from neighbors]    [Scale to 0-255]  [1 byte/point]

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
