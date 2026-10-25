Writing ARL Files
=================

arl-met has two supported authoring paths:

- :func:`arlmet.open_dataset` / :func:`arlmet.write_dataset` for the common
  case where surface variables have no ``level`` dimension and upper-air
  variables share one ``level`` coordinate.
- :class:`arlmet.File` for exact low-level control, including per-record
  forecast hours and other irregular layouts.

.. note::

   The documented Dataset writer is intentionally conservative. If you have a
   better idea for handling irregular ARL files at a higher level, open an
   issue or pull request. The current recommendation is to keep one simple
   Dataset contract and route edge cases through :class:`arlmet.File`.

The examples run when the docs are built, on ``met.arl``, a small file of made-up
values that the build writes with :func:`arlmet.write_dataset`
(:doc:`quickstart` describes it). Use your own file's name in its place.

.. ipython:: python
   :suppress:

   import os

   import docs_examples

   docs_cwd = docs_examples.temp_workdir()
   docs_examples.sample_file("met.arl")

Round-trip with Dataset
-----------------------

.. ipython:: python

   import arlmet

   ds = arlmet.open_dataset("met.arl")
   ds["TEMP"] = ds["TEMP"] - 273.15
   arlmet.write_dataset(ds, "edited.arl")
   arlmet.open_dataset("edited.arl")["TEMP"].max().item()  # degrees C now

To request a trailing DIF record when writing a parent variable, set the
parent DataArray's ``diff`` attr to the DIF record name:

.. ipython:: python

   ds = arlmet.open_dataset("met.arl")
   ds["WWND"].attrs["diff"] = "DIFW"
   arlmet.write_dataset(ds, "edited-with-diff.arl")
   with arlmet.File("edited-with-diff.arl") as met:
       record = met[met.times[0]][(1, "WWND")]
       print([dif.variable for dif in record.diffs])

Dataset write contract
----------------------

- ``time`` and ``arl_grid`` must be present
- ``source`` must be in ``ds.attrs``
- ``forecast_hour(time)``, a coordinate, is the ARL index-record forecast hour
  (0 when absent)
- surface variables use dims ``(time, lat, lon)`` or ``(time, y, x)``
- upper-air variables use dims ``(time, level, lat, lon)`` or
  ``(time, level, y, x)``
- all upper-air variables must share the Dataset ``level`` coordinate

The ``forecast_hour`` coordinate holds one forecast hour per time step, the one
the ARL index record stores, and :func:`arlmet.write_dataset` writes it into the
index record and into every data record of that time step. A data record's own
forecast hour can differ from its index record's (an accumulated field taken from
a 6-hour forecast may say 6 where the index says 0); that is a fact about the
file's records, read with :attr:`arlmet.DataRecord.forecast`, and the Dataset
does not represent it. HYSPLIT reads a data record's forecast hour only as the
missing flag (a negative hour marks the record missing); the forecast hour it
uses is the index record's. So a round trip through a Dataset changes no value
HYSPLIT reads, and a file that must keep per-record hours is written with
:class:`arlmet.File` (below).

.. ipython:: python

   ds = arlmet.open_dataset("met.arl")
   ds["forecast_hour"]

Write constraints
-----------------

- ``forecast_hour`` must be a coordinate with dims ``("time",)`` when present
- variable names must be 4 characters or fewer
- each 2D slice must be finite, or entirely NaN; an all-NaN slice is written
  as no record (this is how :func:`arlmet.open_dataset` represents a variable
  missing from some levels or times), while a partly-NaN slice raises
- upper-air levels are written in ``level`` coordinate order and renumbered
  ``1..N``, so a Dataset holding a subset of a file's levels (from
  ``open_dataset(levels=...)`` or ``ds.sel(level=...)``) is written as a
  compact file
- the output must not be the file the Dataset was opened from; write to a new
  path. The file is written to a temporary name and renamed into place when
  complete, so an interrupted write never leaves a truncated file
- generated DIF names must be declared on the parent variable with
  ``attrs["diff"]``, must start with ``DIF``, and must be one HYSPLIT's name
  table pairs with the parent (see below)
- Dataset DIF writing is parent-led; do not add ``DIF*`` variables as separate
  Dataset data variables
- surface-only datasets must pass ``vertical_axis=`` explicitly to
  :func:`arlmet.write_dataset`

Differential records
--------------------

arl-met supports two DIF authoring paths:

- low-level: pass ``diff="DIF..."`` to :meth:`arlmet.RecordSet.create_datarecord`
- Dataset: set ``DataArray.attrs["diff"] = "DIF..."`` on the parent variable

In both cases the caller supplies only the intended parent field values.
arl-met writes the parent record first, immediately unpacks the packed parent,
computes the residual ``target - unpacked_parent``, and writes that residual as
the trailing DIF record.

The low-level writer also takes several names, ``diff=["DIFW", "DIFW"]``, and
writes that many DIF records after the parent, in order. Each holds what
packing left after the parent and the DIF records before it, so a reader that
adds them all, as HYSPLIT and arl-met do, gets back to the target. This is how
a crop or rewrite keeps a parent's DIF records when a file has several; one is
the usual case, and the Dataset path writes one.

Validation rules:

- DIF names must start with ``DIF``
- each DIF name must be one that HYSPLIT's name table pairs with the parent at
  its level: ``DIFW`` with ``WWND`` or ``DZDT`` and ``DIFT`` with ``TKEN`` or
  ``TKEU`` above the surface, any ``DIF*`` name with ``TPP*`` or ``PRT*`` at
  the surface (level 0); any other pair raises ``ValueError``, since a reader
  would not pair it (*ARL File Format* has the table)
- generated DIF records inherit the parent time, level, and forecast
- the same DIF name cannot be rebound to a different parent variable during one
  write
- existing on-disk ``DIF*`` records are preserved on low-level rewrite and
  subset extraction

Analysis with open_dataset
--------------------------

:func:`arlmet.open_dataset` gives a flat view suited for analysis. Surface
variables (e.g. ``PRSS``, ``SHGT``) have dimensions ``(time, lat, lon)``; upper-
air variables (e.g. ``UWND``, ``VWND``, ``TEMP``) have dimensions
``(time, level, lat, lon)``.

.. ipython:: python

   ds = arlmet.open_dataset("met.arl", bbox=(-114, 39, -110, 42), levels=[0, 1, 2])
   print(ds["PRSS"].dims)
   print(ds["TEMP"].dims)

   p = arlmet.pressure(ds)
   z = arlmet.z_agl(ds)
   p.values
   z.sizes

Advanced authoring with File
----------------------------

Use :class:`arlmet.File` when you need exact control over the records written to
disk.

.. ipython:: python

   import numpy as np
   import pandas as pd

   grid = arlmet.Grid(
       projection=arlmet.Projection(
           pole_lat=90.0,
           pole_lon=0.0,
           tangent_lat=1.0,
           tangent_lon=1.0,
           grid_size=0.0,
           orientation=0.0,
           cone_angle=0.0,
           sync_x=1.0,
           sync_y=1.0,
           sync_lat=-10.0,
           sync_lon=20.0,
       ),
       nx=20,
       ny=20,
   )
   vertical_axis = arlmet.PressureAxis(levels=[0.0, 1000.0])
   time = pd.Timestamp("2024-07-18 00:00")

   prss = np.ones((grid.ny, grid.nx), dtype=np.float32)
   temp = np.ones((grid.ny, grid.nx), dtype=np.float32) * 280.0

   with arlmet.File(
       "custom.arl",
       mode="w",
       source="TEST",
       grid=grid,
       vertical_axis=vertical_axis,
   ) as arl:
       rs = arl.create_recordset(time, forecast=0)
       rs.create_datarecord("PRSS", 0, forecast=0, data=prss)
       rs.create_datarecord("TEMP", 1, forecast=3, data=temp)
       rs.create_datarecord("WWND", 1, forecast=3, data=temp, diff="DIFW")

This low-level path is the recommended workflow whenever different variables at
the same time need different forecast hours or explicit record-level control.
``File`` metadata (``source``, ``grid``, ``vertical_axis``) and the record
options (``forecast``, ``data``, ``diff``) are keyword-only.
:meth:`arlmet.File.add_record` is a shorthand that creates the time step's
record set on first use:

.. ipython:: python

   with arlmet.File(
       "added.arl", mode="w", source="TEST", grid=grid, vertical_axis=vertical_axis
   ) as arl:
       arl.add_record(time, "TEMP", 1, forecast=3, data=temp)

Records are held in memory until the file is flushed or closed. When writing
many time steps, call :meth:`arlmet.File.flush` after filling each one so memory
stays bounded by a single time step instead of the whole file.
:func:`arlmet.write_dataset` and :func:`arlmet.extract_subset` already do this.
Flushed record sets are on disk and can no longer be modified.

.. ipython:: python

   fields = {("PRSS", 0): prss, ("TEMP", 1): temp}
   steps = [(time + pd.Timedelta(hours=h), fields) for h in range(3)]

   with arlmet.File(
       "long.arl", mode="w", source="TEST", grid=grid, vertical_axis=vertical_axis
   ) as arl:
       for time, fields in steps:
           rs = arl.create_recordset(time, forecast=0)
           for (name, level), data in fields.items():
               rs.create_datarecord(name, level=level, forecast=0, data=data)
           arl.flush()  # write this time step and free its memory

   with arlmet.File("long.arl") as met:
       print(met.times)

.. ipython:: python
   :suppress:

   os.chdir(docs_cwd)

Contributing ideas
------------------

If you have a better high-level pattern for irregular ARL authoring,
contributions are welcome. The project intentionally documents one primary
Dataset workflow plus the exact low-level fallback.
