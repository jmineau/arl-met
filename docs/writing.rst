Writing ARL Files
=================

arl-met has two supported authoring paths:

- :func:`arlmet.open_dataset` / :func:`arlmet.write_dataset` for the common
  case where surface variables have no ``level`` dimension and upper-air
  variables share one ``level`` coordinate.
- :class:`arlmet.File` for exact low-level control, including per-variable
  forecast hours and other irregular layouts.

.. note::

   The documented Dataset writer is intentionally conservative. If you have a
   better idea for handling irregular ARL files at a higher level, open an
   issue or pull request. The current recommendation is to keep one simple
   Dataset contract and route edge cases through :class:`arlmet.File`.

Round-trip with Dataset
-----------------------

.. code-block:: python

   import arlmet

  ds = arlmet.open_dataset("input.arl")
   ds["TEMP"] = ds["TEMP"] - 273.15
   arlmet.write_dataset(ds, "edited.arl")

To request a trailing DIF record when writing a parent variable, set the
parent DataArray's ``diff`` attr to the DIF record name:

.. code-block:: python

  ds = arlmet.open_dataset("input.arl")
  ds["WWND"].attrs["diff"] = "DIFW"
  arlmet.write_dataset(ds, "edited-with-diff.arl")

Dataset write contract
----------------------

- ``time`` and ``arl_grid`` must be present
- ``source`` must be in ``ds.attrs``
- ``forecast_hour(time)`` is the ARL index-record forecast hour
- surface variables use dims ``(time, lat, lon)`` or ``(time, y, x)``
- upper-air variables use dims ``(time, level, lat, lon)`` or
  ``(time, level, y, x)``
- all upper-air variables must share the Dataset ``level`` coordinate

The ``forecast_hour`` data variable is intentionally limited in scope. Its attrs
clarify that it stores the forecast hour written to the ARL index record for
each time step. Individual variable record forecasts may differ and are not
represented in the Dataset API.

Write constraints
-----------------

- ``forecast_hour`` must have dims ``("time",)`` when present
- variable names must be 4 characters or fewer
- slices must be complete and finite; missing values are not written
- generated DIF names must be declared on the parent variable with
  ``attrs["diff"]`` and must start with ``DIF``
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

Validation rules:

- DIF names must start with ``DIF``
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

.. code-block:: python

   import arlmet

   ds = arlmet.open_dataset("input.arl", bbox=(-114, 39, -110, 42), levels=[0, 1, 2])
   print(ds["PRSS"].dims)  # ('time', 'lat', 'lon')
   print(ds["TEMP"].dims)  # ('time', 'level', 'lat', 'lon')

   p = arlmet.pressure(ds)
   z = arlmet.z_agl(ds)

Advanced authoring with File
----------------------------

Use :class:`arlmet.File` when you need exact control over the records written to
disk.

.. code-block:: python

   import numpy as np
   import pandas as pd
   import arlmet

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
   vertical_axis = arlmet.VerticalAxis(flag=2, levels=[0.0, 1000.0])
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
       rs.create_datarecord("PRSS", level=0, forecast=0, data=prss)
    rs.create_datarecord("TEMP", level=1, forecast=3, data=temp)
    rs.create_datarecord("WWND", level=1, forecast=3, data=temp, diff="DIFW")

This low-level path is the recommended workflow whenever different variables at
  the same time need different forecast hours or explicit record-level control.

Contributing ideas
------------------

If you have a better high-level pattern for irregular ARL authoring,
contributions are welcome. The project intentionally documents one primary
Dataset workflow plus the exact low-level fallback.
