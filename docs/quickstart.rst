Quick Start
===========

This page shows the shortest useful end-to-end workflow: open an ARL file,
inspect a few variables, crop it to a smaller domain, and write the result
back to disk.

The examples run when the docs are built, on ``met.arl``, a small file of made-up
values that the build writes with :func:`arlmet.write_dataset`: a July day over
Utah, four time steps six hours apart, on a 61 by 51 grid of 0.1 degree cells
and four pressure levels. Use your own file's name in its place.

.. ipython:: python
   :suppress:

   import os

   import docs_examples

   docs_cwd = docs_examples.temp_workdir()
   docs_examples.sample_file("met.arl")

Open a file lazily with xarray
------------------------------

The main entry point is :func:`arlmet.open_dataset`. It returns an
``xarray.Dataset`` backed by lazy ARL reads, so variables are unpacked only
when you actually access them.

Surface variables (e.g. ``PRSS``, ``SHGT``) have dimensions
``(time, lat, lon)`` and upper-air variables (e.g. ``UWND``, ``VWND``,
``TEMP``) have dimensions ``(time, level, lat, lon)``. The ``level``
coordinate contains only the upper-air pressure levels. ``ds.isel(level=0)``
selects the first upper-air level for upper-air variables and leaves surface
variables unchanged.

.. ipython:: python

   import arlmet

   ds = arlmet.open_dataset("met.arl")
   print(ds)
   print(ds["PRSS"].dims)
   print(ds["UWND"].dims)
   ds.isel(level=0)  # selects first upper level; surface vars unchanged

arlmet also registers itself as an xarray backend, so the same Dataset is
available through :func:`xarray.open_dataset` with ``engine="arl"``. That route
adds xarray's generic options, such as ``chunks=`` for dask-backed arrays:

.. ipython:: python

   import xarray as xr

   ds = xr.open_dataset("met.arl", engine="arl")
   ds = xr.open_dataset("met.arl", engine="arl", chunks={"time": 1})
   ds["TEMP"]

ARL files have no standard extension, so xarray recognizes them by their first
record: ``xr.open_dataset("met.arl")`` works without ``engine=`` too,
as long as no other installed backend claims the file first.

Select a smaller domain while reading
-------------------------------------

Pass ``bbox=`` and ``levels=`` to crop before unpacking.

.. ipython:: python

   ds = arlmet.open_dataset(
       "met.arl",
       bbox=(-114.0, 39.0, -110.0, 42.0),
       levels=[0, 1, 2],
   )
   ds.sizes

Write a cropped copy to a new ARL file
--------------------------------------

Use :func:`arlmet.extract_subset` when you want a new ARL file on disk rather
than an in-memory xarray subset.

.. ipython:: python

   arlmet.extract_subset(
       "met.arl",
       "cropped.arl",
       bbox=(-114.0, 39.0, -110.0, 42.0),
       levels=[0, 1, 2],
   )

Modify a file and write it back
-------------------------------

Use :func:`arlmet.open_dataset` and :func:`arlmet.write_dataset` for the
common case where surface variables have no ``level`` dimension and upper-air
variables share one ``level`` coordinate.

.. ipython:: python

   ds = arlmet.open_dataset("met.arl")
   ds["TEMP"] = ds["TEMP"] - 273.15
   arlmet.write_dataset(ds, "edited.arl")

If you need a trailing DIF record for a parent variable, set the parent
DataArray's ``diff`` attr before writing:

.. ipython:: python

   ds["WWND"].attrs["diff"] = "DIFW"
   arlmet.write_dataset(ds, "edited-with-diff.arl")

For irregular files, including per-variable forecast hours, see :doc:`writing`
for the low-level :class:`arlmet.File` workflow.

Sample variables at arbitrary points
------------------------------------

Use :func:`arlmet.sample_points` for trajectory or receptor-style sampling.

.. ipython:: python

   import pandas as pd

   points = pd.DataFrame(
       {
           "lon": [-111.9],
           "lat": [40.7],
           "z": [850.0],
           "time": ["2024-07-18 00:00"],
       }
   )
   arlmet.sample_points("met.arl", points, ["UWND", "VWND"])

.. ipython:: python
   :suppress:

   os.chdir(docs_cwd)

Where to go next
----------------

- :doc:`downloading` for NOAA archive download helpers
- :doc:`cropping` for lazy reads versus writing cropped ARL files
- :doc:`writing` for Dataset writer requirements and low-level file creation
- :doc:`api` for the full reference
