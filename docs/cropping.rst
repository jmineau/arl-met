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

A crop with ``bbox`` unpacks each record's window and packs it again, so its
values can move by about one packing step of the new record (its precision,
see :doc:`format`): up to 0.2 m/s for ``UWND`` on a reanalysis crop, where the
crop's own first column holds the jet's gradient. Without ``bbox`` (only
``levels`` or ``variables``) the records are copied byte for byte and no value
changes.

.. ipython:: python

   out = arlmet.extract_subset(
       "met.arl", "cropped.arl", bbox=(-114.0, 39.0, -110.0, 42.0)
   )
   ds = arlmet.open_dataset(out)
   ds.sizes

``extract_subset()`` replaces an existing output. In a batch you may rerun after a
failure, pass ``overwrite=False`` to keep the crops already written: an existing
``dest`` is returned as it is, without reading the input. Nothing checks that it
was cropped with the same arguments, so use it only when the output name says
what went into it.

.. ipython:: python

   arlmet.extract_subset(
       "met.arl", "cropped.arl", bbox=(-114.0, 39.0, -110.0, 42.0), overwrite=False
   )

When you already hold an open :class:`arlmet.File`, the same operation is
available as a method:

.. ipython:: python

   with arlmet.File("met.arl") as met:
       out = met.extract_subset("cropped.arl", bbox=(-114.0, 39.0, -110.0, 42.0))
   ds = arlmet.open_dataset(out)

A ``bbox`` with ``west > east`` crosses the dateline, on any grid:
``(170.0, 50.0, -170.0, 60.0)`` is the box ``(170.0, 50.0, 190.0, 60.0)``. On a
global grid a box may also cross the grid's seam, the column where its longitudes
start again (0 E on GDAS or GFS, which start at 0 E; 180 on a grid that starts at
180 W). The crop then holds only the box's columns, in order from west to east
across the seam: ``bbox=(-10.0, 35.0, 30.0, 60.0)`` on GFS 0.25 is 161 columns
wide, not all 1440. HYSPLIT's ``xtrct_grid`` writes the same header for such a
crop, but its values east of the seam come from the column after the right one.
A box with columns at both ends of a lat-lon grid that does not span 360 degrees
raises ``ValueError``, since no single grid holds them.

On a lat-lon grid longitudes in a ``bbox`` count modulo 360: 180 and -180 are
the same meridian, so an east edge at 180 keeps the column at 180 whether the
grid writes it as 180 or -180, and a box may reach past 180 (HYSPLIT's
``xtrct_grid`` agrees for ``(-190, -170)``, ``(190, 200)`` and ``(-10, 10)``, but
not for a box that wraps past 360 or spans 360 degrees: for ``(350, 370)``,
``(-180, 180)`` and ``(10, 370)`` it writes a longer, a half or an unreadable
file, where arl-met returns the box's columns). A box 360 degrees wide or more
holds every column. ``bbox=(170.0, 0.0, 190.0, 10.0)`` and ``(170.0, 0.0, -170.0, 10.0)``
select the same columns. A box that reaches past the grid's own columns only
crops to the ones it has (a regional grid from 160 E to 200 E holds
``(170.0, 25.0, 190.0, 30.0)`` whole). A column or row within a millionth of a cell
of an edge counts as on it, so ``bbox=(-112.3, 39.0, -110.0, 42.0)`` keeps the
column at -110 whether the grid stores it as -110 or as 250, and an edge at
41.4 keeps the row at 41.4 on a 0.1 degree grid, where 0.1 is not exact in
binary and the stored latitude is 41.400000000000006.

A box in another coordinate system
----------------------------------

A ``bbox`` is in degrees of longitude and latitude unless you pass ``bbox_crs=``,
the coordinate reference system it is in: an EPSG code (``"EPSG:32612"`` or
``32612``), a PROJ string, a :class:`pyproj.CRS`, anything
:meth:`pyproj.CRS.from_user_input` takes. The box is then
``(xmin, ymin, xmax, ymax)`` in that system's units, easting before northing. Here
a box of 80 by 160 km in UTM zone 12N, around Salt Lake City:

.. ipython:: python

   utm = arlmet.open_dataset(
       "met.arl",
       bbox=(380_000.0, 4_400_000.0, 460_000.0, 4_560_000.0),
       bbox_crs="EPSG:32612",
   )
   dict(utm.sizes)

``extract_subset()``, ``File.extract_subset()``, ``File.to_dataset()`` and
``xr.open_dataset(..., engine="arl")`` take ``bbox_crs=`` too. The box is a
rectangle in its own system, which on the grid is a curved shape, so the crop is
found as a projected grid's is (next section), on either kind of grid: the smallest
rectangle of cells that holds every point of the box on the grid. A global lat-lon
grid is cut at its seam there, so a box across the seam gets every column.
``bbox_crs="EPSG:4326"`` is the same as none, in either axis order
(``"OGC:CRS84"`` too). A projected grid's own system, ``grid.crs``, is in meters
from the center of its first cell, so ``bbox_crs=ds.arl.grid.crs`` takes a box in
cells times the grid spacing.

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

Variable names must be spelled as the file spells them (``"TEMP"``, not
``"temp"``); a single string such as ``variables="TEMP"`` is one name. A
``DIF*`` name the file holds (``"DIFW"``) is accepted but selects nothing: that
record is written with its parent (``"WWND"``) whenever the parent is selected.
A name the file does not hold raises a ``ValueError`` that lists
every unknown name and the variables the file has, also when the other names are
fine, so a typo never gives a file that is missing a field. A selection that
leaves no record in the file, such as ``variables=[]``, or a variable that is
only on levels you left out, raises too: ``extract_subset()`` never writes an
empty file.

.. ipython:: python
   :suppress:

   os.chdir(docs_cwd)

Which cells a crop keeps
------------------------

On a lat-lon grid a crop keeps the cells whose centers lie inside the box, edges
included. ``xtrct_grid`` takes the column nearest each edge, so it keeps the same
columns when the edges are on columns and can keep one more at an edge between two
(on GDAS 1, ``bbox=(-112.6, -70, -100.4, 70)`` is 14 columns there and 12 here).

On a projected grid (Lambert conformal, polar stereographic, Mercator) a crop is
the smallest rectangle of the grid's cells that holds the box's part on the grid:
every point inside the box that lies on the grid lies in one of the crop's cells.
Parallels and meridians curve on these grids, so the box is not a rectangle in
grid cells. arl-met maps the box's outline, points along its edges at most half a
cell apart on the grid, to grid indices, and keeps the points inside the grid's
cells, the points where the outline crosses the grid's border, and the grid's
corners that lie inside the box. It rounds the lowest and highest index on each
axis to the nearest cell, as Fortran's ``NINT`` rounds. Points of the box off the
grid do not widen the crop: a box that reaches past the grid, across the meridian
opposite the projection's center or to a pole the projection cannot map, is cropped
to its part on the grid (the whole grid, for ``bbox=(-180.0, 0.0, 180.0, 90.0)`` on
HRRR), and a box with no point on the grid raises ``ValueError``.

This departs from ``xtrct_grid``, which maps all four corners but keeps only the
rectangle between the south-west and north-east ones, wherever they fall. Its crop
can leave out the box's south-east and north-west corners, so a point there, such
as a receptor inside the box, can be off the cropped grid; and across the cut its
corners' indices mean nothing. For a box inside the grid, arl-met's crop holds
``xtrct_grid``'s and is often a few cells larger: for ``bbox=(-113.5, 39.5, -110.5, 42.0)`` on HRRR
it is 99 x 106 cells where ``xtrct_grid``'s is 99 x 79, with 14 more rows to the
south and 13 to the north. Crops of lat-lon grids (GFS, GDAS) are the same as
``xtrct_grid``'s.

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
