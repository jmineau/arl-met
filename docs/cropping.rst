Cropping ARL Data
=================

arl-met supports two related workflows:

- crop or subset lazily while reading into xarray
- write a new ARL file that contains only the selected domain, levels, or variables

The examples run when the docs are built, on ``met.arl``, a small file of made-up
values that the build writes with :func:`arlmet.write_dataset`
(:doc:`quickstart` describes it). Use your own file's name in its place.

.. ipython:: python
   :suppress:

   import os

   import docs_examples

   docs_cwd = docs_examples.temp_workdir()
   docs_examples.sample_file("met.arl")

Lazy crop while reading
-----------------------

Pass ``bbox=`` to :func:`arlmet.open_dataset` when you want an in-memory subset
for analysis.

.. ipython:: python

   import arlmet

   ds = arlmet.open_dataset(
       "met.arl",
       bbox=(-114.0, 39.0, -110.0, 42.0),
   )
   ds.sizes

This path crops before unpacking, so it avoids reading the full grid when you
only need a smaller region.

Select levels while reading
---------------------------

ARL levels are selected by integer level index.

.. ipython:: python

   ds = arlmet.open_dataset(
       "met.arl",
       bbox=(-114.0, 39.0, -110.0, 42.0),
       levels=[0, 1, 2],
   )
   ds.sizes

Write a smaller ARL file
------------------------

Use :func:`arlmet.extract_subset` when you want a new ARL file on disk.

.. ipython:: python

   arlmet.extract_subset(
       "met.arl",
       "cropped.arl",
       bbox=(-114.0, 39.0, -110.0, 42.0),
   )

``extract_subset()`` returns the output path as a :class:`pathlib.Path`, so you
can pass it straight to :func:`arlmet.open_dataset` or :class:`arlmet.File`.

The subset is written one time step at a time, so memory use is bounded by a
single cropped time step rather than the whole output file.

.. ipython:: python

   out = arlmet.extract_subset(
       "met.arl", "cropped.arl", bbox=(-114.0, 39.0, -110.0, 42.0)
   )
   ds = arlmet.open_dataset(out)
   ds.sizes

When you already hold an open :class:`arlmet.File`, the same operation is
available as a method:

.. ipython:: python

   with arlmet.File("met.arl") as met:
       out = met.extract_subset("cropped.arl", bbox=(-114.0, 39.0, -110.0, 42.0))
   ds = arlmet.open_dataset(out)

Keep only the levels and variables you need
-------------------------------------------

``extract_subset()`` can keep only the levels and variables you need.

.. ipython:: python

   arlmet.extract_subset(
       "met.arl",
       "subset.arl",
       bbox=(-114.0, 39.0, -110.0, 42.0),
       levels=[0, 1, 2],
       variables=["UWND", "VWND", "TEMP"],
   )
   list(arlmet.open_dataset("subset.arl").data_vars)

Without a ``bbox``, selecting levels or variables does not unpack anything:
the selected records are copied byte for byte and only their level numbers
and the index records are rewritten, so it runs at close to file-copy speed
and the output values are exactly the input's.

.. ipython:: python
   :suppress:

   os.chdir(docs_cwd)

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
