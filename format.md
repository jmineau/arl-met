# ARL Packed Data Format - Hierarchical Structure

## Overview

The ARL (Air Resources Laboratory) packed data format is a hierarchical binary structure designed for efficient storage and retrieval of meteorological data. This format is used by HYSPLIT and related atmospheric transport models.

## File Structure

```
ARL METEOROLOGICAL DATA FILE
├── RecordSet 1
│   ├── Index Record
│   │   ├── Header (50 bytes ASCII)
│   │   │   ├── Year (2 digits)
│   │   │   ├── Month (2 digits)
│   │   │   ├── Day (2 digits)
│   │   │   ├── Hour (2 digits)
│   │   │   ├── Forecast Hour (2 digits)
│   │   │   ├── Level (2 digits)
│   │   │   ├── Grid ID (2 chars)
│   │   │   ├── Variable Label (4 chars)
│   │   │   ├── Exponent (1 float)
│   │   │   ├── Precision (1 float)
│   │   │   └── Initial Value (1 float)
│   │   ├── Data Source (4 chars)
│   │   ├── Forecast Hour (3 digits)
│   │   ├── Minutes (2 digits)
│   │   ├── Grid Definition (12 floats)
│   │   ├── Grid Dimensions (nx, ny, nz)
│   │   ├── Vertical Coordinate System (1-4)
│   │   └── For Each Level:
│   │       ├── Level Height
│   │       ├── Number of Variables
│   │       └── Variable List (4-char IDs + checksums)
│   │
│   ├── Record(s) (Fixed Length: 50 + nx*ny bytes)
│   │   ├── SURFACE LEVEL (Level 0)
│   │   │   ├── Variable 1: PRSS (50-byte header + packed data)
│   │   │   ├── Variable 2: MSLP (50-byte header + packed data)
│   │   │   ├── Variable 3: TPP1 (50-byte header + packed data)
│   │   │   └── ... (all surface variables)
│   │   │
│   │   ├── UPPER LEVEL 1 (e.g., 1000 mb)
│   │   │   ├── Variable 1: UWND (50-byte header + packed data)
│   │   │   ├── Variable 2: VWND (50-byte header + packed data)
│   │   │   ├── Variable 3: TEMP (50-byte header + packed data)
│   │   │   └── ... (all variables for this level)
│   │   │
│   │   ├── UPPER LEVEL 2 (e.g., 925 mb)
│   │   └── ... (continues for all vertical levels)
│
├── RecordSet 2
│   ├── Index Record
│   ├── Data Record(s)
│
└── ... (additional record sets for other grids or time periods)
```

## Key Components

### 1. File Level Structure

- **Sequential Time Periods**: Each represents one forecast time
- **Direct Access**: Fixed record length allows random access
- **Platform Independent**: Binary format works across systems

### 2. Recrord Set Structure

```
┌─────────────────────────────────────────┐
│ INDEX RECORD(S)                         │
│ - Metadata for entire time period       │
│ - Grid definition                       │
│ - Variable catalog                      │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│ DATA RECORDS                            │
│ - One record per variable per level     │
│ - Surface data first (Level 0)          │
│ - Then upper levels (1 to nz-1)         │
└─────────────────────────────────────────┘
```

### 3. Individual Data Record Structure

```
┌──────────────────────┬──────────────────────────────┐
│ HEADER (50 bytes)    │ PACKED DATA (nx × ny bytes)  │
│ - Variable ID (4)    │ - One byte per grid point    │
│ - Date/Time          │ - Difference encoding        │
│ - Level indicator    │ - Precision maintained       │
│ - Checksum info      │                              │
└──────────────────────┴──────────────────────────────┘
```

### 4. Packing Algorithm Concept

```
Original Data → Difference Encoding → Scale/Quantize → Pack to bytes
    ↓                    ↓                  ↓              ↓
[Real Array]    [Δ from neighbors]    [Scale to 0-255]  [1 byte/point]
```

### 5. Multiple Grid Support

```
GRID 1 (High Resolution)
├── Time Period 1
├── Time Period 2
└── ...

GRID 2 (Lower Resolution)  
├── Time Period 1
├── Time Period 2
└── ...
```

- Each grid has separate INDEX records
- Each grid uses different unit numbers
- HYSPLIT selects highest resolution grid at particle location

### 6. Variable Organization

#### Surface Variables (Level 0)
- **Pressure fields**: PRSS, MSLP
- **Temperature**: T02M, TMPS
- **Winds**: U10M, V10M
- **Fluxes**: SHTF, LTHF, USTR
- **Precipitation**: TPP1, RAINC, RAINNC

#### Upper-Air Variables (Levels 1-nz)
- **Winds**: UWND, VWND, WWND
- **Temperature**: TEMP
- **Heights**: HGTS
- **Moisture**: RELH, SPHU
- **Special**: TKEN, DIFT, DIFW

## Key Features

This hierarchical structure provides:

- **Efficient Storage**: 1 byte per grid point with maintained precision
- **Fast Access**: Direct access to any variable at any level/time
- **Flexibility**: Support for multiple grids and coordinate systems
- **Portability**: Platform-independent binary format
- **Extensibility**: New variables can be added via configuration files

## Summary

The key insight is that this is essentially a database structure optimized for meteorological data, where the index records serve as the catalog and the data records are stored in a highly compressed but quickly accessible format.