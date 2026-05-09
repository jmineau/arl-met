Quick Start
===========

This page shows the shortest useful end-to-end workflow: open an ARL file,
inspect a few variables, crop it to a smaller domain, and write the result
back to disk.

Open a file lazily with xarray
------------------------------

The main entry point is :func:`arlmet.open_dataset`. It returns an
``xarray.Dataset`` backed by lazy ARL reads, so variables are unpacked only
when you actually access them.

.. code-block:: python

   import arlmet

   ds = arlmet.open_dataset("path/to/file.arl")
   print(ds)
   print(ds.data_vars)

Select a smaller domain while reading
-------------------------------------

Pass ``bbox=`` and ``levels=`` to crop before unpacking.

.. code-block:: python

   import arlmet

   ds = arlmet.open_dataset(
       "path/to/file.arl",
       bbox=(-114.0, 39.0, -110.0, 42.0),
       levels=[0, 1, 2],
   )

Write a cropped copy to a new ARL file
--------------------------------------

Use :func:`arlmet.extract_subset` when you want a new ARL file on disk rather
than an in-memory xarray subset.

.. code-block:: python

   import arlmet

   arlmet.extract_subset(
       "path/to/file.arl",
       "path/to/cropped.arl",
       bbox=(-114.0, 39.0, -110.0, 42.0),
       levels=[0, 1, 2],
   )

Modify a dataset and write it back
----------------------------------

The easiest way to write an ARL file is to start from a dataset produced by
``open_dataset()``, modify values, and then call :func:`arlmet.write_dataset`.

.. code-block:: python

   import arlmet

   ds = arlmet.open_dataset("path/to/file.arl", levels=[0, 1, 2])
   ds = ds[["UWND", "VWND", "TEMP"]].load()
   ds["TEMP"] = ds["TEMP"] - 273.15
   arlmet.write_dataset(ds, "path/to/edited.arl")

Sample variables at arbitrary points
------------------------------------

Use :func:`arlmet.sample_points` for trajectory or receptor-style sampling.

.. code-block:: python

   import pandas as pd
   import arlmet

   points = pd.DataFrame(
       {
           "lon": [-111.9],
           "lat": [40.7],
           "z": [850.0],
           "time": ["2024-07-18 00:00"],
       }
   )
   samples = arlmet.sample_points(arlmet.File("path/to/file.arl"), points, ["UWND", "VWND"])

Where to go next
----------------

- :doc:`downloading` for NOAA archive download helpers
- :doc:`cropping` for lazy reads versus writing cropped ARL files
- :doc:`writing` for xarray-to-ARL requirements and round-trip workflows
- :doc:`api` for the full reference
