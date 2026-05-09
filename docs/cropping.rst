Cropping ARL Data
=================

arl-met supports two related workflows:

- crop or subset lazily while reading into xarray
- write a new ARL file that contains only the selected domain, levels, or variables

Lazy crop while reading
-----------------------

Pass ``bbox=`` to :func:`arlmet.open_dataset` when you want an in-memory subset
for analysis.

.. code-block:: python

   import arlmet

   ds = arlmet.open_dataset(
       "input.arl",
       bbox=(-114.0, 39.0, -110.0, 42.0),
   )

This path crops before unpacking, so it avoids reading the full grid when you
only need a smaller region.

Select levels while reading
---------------------------

ARL levels are selected by integer level index.

.. code-block:: python

   ds = arlmet.open_dataset(
       "input.arl",
       bbox=(-114.0, 39.0, -110.0, 42.0),
       levels=[0, 1, 2],
   )

Write a smaller ARL file
------------------------

Use :func:`arlmet.extract_subset` when you want a new ARL file on disk.

.. code-block:: python

   import arlmet

   arlmet.extract_subset(
       "input.arl",
       "cropped.arl",
       bbox=(-114.0, 39.0, -110.0, 42.0),
   )

Keep only the levels and variables you need
-------------------------------------------

``extract_subset()`` can keep only the levels and variables you need.

.. code-block:: python

   arlmet.extract_subset(
       "input.arl",
       "subset.arl",
       bbox=(-114.0, 39.0, -110.0, 42.0),
       levels=[0, 1, 2],
       variables=["UWND", "VWND", "TEMP"],
   )

What changes in the output file
-------------------------------

When levels are selected, the output file is renumbered from zero upward while
preserving the original level values in the vertical axis metadata.

When variables are selected, only those variables are written into each output
time step. Forecast hours from the source index records are preserved.

Limitations
-----------

- ``bbox`` is always interpreted as ``(west, south, east, north)`` in degrees
- projected-grid bounding boxes must not cross the dateline
- the cropped grid must still be large enough to hold the ARL index record

If the selected domain is too small to hold the output index record,
``extract_subset()`` raises a ``ValueError`` explaining how many grid cells are
required.