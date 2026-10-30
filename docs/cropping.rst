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

ARL levels are selected by integer level index, counted from 0 at the surface,
so ``levels=[0, 1, 2]`` keeps the surface fields and the first two upper-air
levels (the Dataset's ``level`` coordinate is then 1 and 2).

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

On a lat-lon grid, a ``bbox`` with ``west > east`` crosses the dateline. On a
global grid a box may also cross the grid's seam, the column where its longitudes
start again (0 E on GDAS or GFS, which start at 0 E; 180 on a grid that starts at
180 W). The crop then holds only the box's columns, in order from west to east
across the seam: ``bbox=(-10.0, 35.0, 30.0, 60.0)`` on GFS 0.25 is 161 columns
wide, not all 1440. HYSPLIT's ``xtrct_grid`` writes the same header for such a
crop, but its values east of the seam come from the column after the right one.
A box with columns at both ends of a lat-lon grid that does not span 360 degrees
raises ``ValueError``, since no single grid holds them.

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
time step. Forecast hours are preserved, the index records' and each data
record's own.

A crop of a projected grid keeps the source's sync point, the latitude and
longitude the index record ties to one grid point, and moves only that grid
point (``sync_x``, ``sync_y``) by where the crop starts, so each cell of the crop
sits exactly where the source file puts it. The grid point can then lie outside
the crop (``sync_y = -326`` for a crop of HRRR that starts at row 328); HYSPLIT
reads it so. HYSPLIT's own ``xtrct_grid`` writes the corner cell's latitude and
longitude instead, which the index record stores to 0.01 degree, so its crops move
by up to 0.15 cells of a 3 km grid. A crop of a lat-lon grid has its corner cell
as its sync point, as HYSPLIT requires there; it is exact when the spacing is a
multiple of 0.01 degree. Its longitudes are written east of Greenwich (0-360), as
``xtrct_grid`` writes them and NOAA's global files store them, because HYSPLIT
places a lat-lon grid by the sign of its stored longitudes; arl-met's ``lon``
coordinate is the same either way.

Limitations
-----------

- ``bbox`` is always interpreted as ``(west, south, east, north)`` in degrees
- projected-grid bounding boxes must not cross the dateline
- on a lat-lon grid short of 360 degrees, a box must not select columns at both
  of its ends
- the cropped grid must still be large enough to hold the ARL index record

If the selected domain is too small to hold the output index record,
``extract_subset()`` raises a ``ValueError`` explaining how many grid cells are
required. The index record lists every level and variable, so the minimum grows
with them. On a NAM 12 km file with all its levels and variables it is 2,012
cells, and the 4 by 3 degree box used on this page is 33 by 26 = 858 cells
there: pass ``levels=`` or ``variables=``, or use a larger box. On a GFS 0.25
degree file with all levels it is 3,340 cells.
