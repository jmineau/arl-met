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

   ds = arlmet.open_dataset("input.arl", squeeze=False)
   ds["TEMP"] = ds["TEMP"] - 273.15
   arlmet.write_dataset(ds, "edited.arl")

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
- ``DIF*`` variables are not currently writable
- surface-only datasets must pass ``vertical_axis=`` explicitly to
  :func:`arlmet.write_dataset`

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

This low-level path is the recommended workflow whenever different variables at
the same time need different forecast hours.

Contributing ideas
------------------

If you have a better high-level pattern for irregular ARL authoring,
contributions are welcome. The project intentionally documents one primary
Dataset workflow plus the exact low-level fallback.