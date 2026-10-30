Sampling Points
===============

Use :func:`arlmet.sample_points` to interpolate meteorological variables at
arbitrary ``(lon, lat, z, time)`` points. This is the direct path when you have
trajectory points, station locations, or receptors and want field values at
each one without building a full Dataset.

The examples run when the docs are built, on ``met.arl``, a small file of made-up
values that the build writes with :func:`arlmet.write_dataset`
(:doc:`quickstart` describes it), and on a few more like it, named where they
are used.

.. ipython:: python
   :suppress:

   import os

   import docs_examples

   docs_cwd = docs_examples.temp_workdir()
   docs_examples.sample_file("met.arl")

Basic usage
-----------

Provide the points as a DataFrame (or dict) with ``lon``, ``lat``, ``z``, and
``time`` columns, plus the variables you want sampled. The first argument,
``files``, can be a path to an ARL file — :func:`arlmet.sample_points` opens and
closes it for you.

.. ipython:: python

   import pandas as pd

   import arlmet

   points = pd.DataFrame(
       {
           "lon": [-111.9, -112.0],
           "lat": [40.7, 40.8],
           "z": [850.0, 700.0],
           "time": ["2024-07-18 00:00", "2024-07-18 00:00"],
       }
   )

   result = arlmet.sample_points("met.arl", points, ["UWND", "VWND", "TEMP"])
   result

The returned DataFrame is a copy of ``points`` with one column added per
requested variable. Every column of ``points`` (IDs, observations, anything
else) and its index are preserved, so you can compare samples with
observations row by row. A requested variable whose name matches an existing
column raises ``ValueError`` rather than overwriting it.

If you already have an open file, pass it instead of a path. The
:class:`arlmet.File` object also exposes the same operation as a method:

.. ipython:: python

   with arlmet.File("met.arl") as met:
       result = met.sample_points(points, ["UWND", "VWND", "TEMP"])
   result

A file you open yourself is left open for you to manage; only paths that
``sample_points`` opens internally are closed for you.

Choosing the vertical coordinate
---------------------------------

The ``z`` column is interpreted according to ``z_kind`` (default
``"pressure"``):

.. list-table::
   :header-rows: 1

   * - ``z_kind``
     - Meaning of ``z``
     - Requirements
   * - ``"native"``
     - fractional level index
     - none
   * - ``"pressure"``
     - hPa
     - ``PRSS`` for sigma/hybrid files (sigma files use ``PRES`` too when they
       have it); ``PRES`` for WRF files (flag=5)
   * - ``"agl"``
     - meters above ground level
     - flag=2: ``HGTS`` + ``SHGT``; flag=1/4: ``PRSS`` + ``TEMP``; flag=3: none;
       flag=5: ``PRSS`` + ``PRES`` + ``TEMP``
   * - ``"msl"``
     - meters above mean sea level
     - flag=2: ``HGTS`` + ``SHGT``; flag=1/4: ``PRSS`` + ``TEMP`` + ``SHGT``; flag=3: ``SHGT``;
       flag=5: ``PRSS`` + ``PRES`` + ``TEMP`` + ``SHGT``

On sigma, hybrid and WRF files (flags 1, 4 and 5) the heights are HYSPLIT's:
from the ground (``PRSS``, ``T02M``) through each level's pressure (``PRES`` on
sigma files that have it and on WRF files) with the virtual temperature
(``SPHU`` or ``RELH``), as HYSPLIT's ``prfsig`` and ``prfecm`` build them. Level 0 is the ground, so a point
between the ground and the first level is interpolated from the ground values
(``PRSS`` for ``PRES``, ``T02M`` for ``TEMP``, ``U10M`` for ``UWND``, ...)
instead of being NaN, and pressure is interpolated as ``ln p``. The
:doc:`vertical` guide has the details.

On pressure-level files (flag 2) the heights are ``HGTS``, and the column also
starts at the ground, as HYSPLIT's ``prfprs`` builds it: over high ground the
lowest pressure levels are below the surface (their values extrapolated by the
model), and they are left out, so a point between the ground and the first
level above it is interpolated from the ground values. A point under the
lowest level at a low site gets them too, instead of NaN. At heights pressure
is interpolated as ``ln p`` here too.

.. ipython:: python

   above_ground = points.assign(z=[500.0, 1500.0])  # meters above ground level
   with arlmet.File("met.arl") as met:
       result = arlmet.sample_points(
           met, above_ground, ["TEMP"], z_kind="agl"
       )
   result

The virtual ``"pressure"`` variable, the pressure of the levels interpolated to
each point, can also be requested as a sampled column. On sigma files with
``PRES`` it is the same as sampling ``PRES``:

.. ipython:: python

   result = arlmet.sample_points("met.arl", points, ["pressure", "TEMP"])
   result

Horizontal interpolation
------------------------

``method`` controls horizontal interpolation between grid cells:

- ``"linear"`` (default) — bilinear interpolation
- ``"nearest"`` — nearest grid cell

.. ipython:: python

   result = arlmet.sample_points("met.arl", points, ["TEMP"], method="nearest")
   result

Earth-relative winds
--------------------

Winds in ARL files on projected grids (Lambert conformal, polar
stereographic, Mercator) are stored relative to the grid axes, which is what
HYSPLIT expects. Sampled ``UWND``/``VWND`` and ``U10M``/``V10M`` are therefore
not directly comparable with observed winds. Pass ``earth_relative=True`` to
rotate each pair so that ``u`` points east and ``v`` north, using the meridian
convergence of the file's grid at each point. Both components of a pair must
be requested. Lat/lon grids are unaffected.

Here ``lambert.arl`` is a made-up file on a Lambert conformal grid, and the
same points are sampled both ways:

.. ipython:: python
   :suppress:

   docs_examples.sample_file("lambert.arl", projected=True)

.. ipython:: python

   arlmet.sample_points("lambert.arl", points, ["UWND", "VWND"])
   arlmet.sample_points("lambert.arl", points, ["UWND", "VWND"], earth_relative=True)

The same rotation is available on the grid for winds you obtained another way,
for example from :func:`arlmet.open_dataset`:

.. ipython:: python

   with arlmet.File("lambert.arl") as met:
       u_earth, v_earth = met.grid.rotate_winds(u=5.0, v=-3.0, lon=-111.9, lat=40.7)
   print(u_earth, v_earth)

WRF winds
---------

In a WRF file (source ``AWRF``, written by HYSPLIT's ``arw2arl``) ``UWND`` and
``VWND`` are on WRF's staggered grid, not at the cell centers where the other
fields are: column ``i`` of ``UWND`` is the west face of cell ``i``, and row
``j`` of ``VWND`` its south face. HYSPLIT reads ``UWND`` half a cell east of a
position and ``VWND`` half a cell north, and so does ``sample_points``: at the
center of a cell, ``UWND`` is the mean of that cell's west and east faces.

Each wind's whole column is taken there, its ground value (``U10M`` or
``V10M``) and its levels' heights included, as HYSPLIT builds a profile at each
grid point before interpolating; an ``msl`` height is still measured from the
terrain at the point itself. With ``method="nearest"`` the winds are those at
the nearest cell's center. The file holds no east face for the last column of
cells and no north face for the last row, so ``UWND`` (``VWND``) is NaN there.
Other sources are not affected, nor are the readers (:class:`arlmet.File`,
:func:`arlmet.open_dataset`), which return each field on the grid it is stored
on.

Some WRF files hold time-averaged winds coupled to the column mass: those
``arw2arl`` writes with ``MUU0`` and ``MUV0`` (the column mass at the U and V
points) and the map factors ``MSFU``, ``MSFV`` and ``MSFT``, as STILT's WRF
runs use. Their ``UWND`` is of order 1e5, and they store potential temperature
``THET`` instead of ``TEMP``. ``sample_points`` converts them as HYSPLIT does,
at each grid point before interpolating:

- ``UWND * MSFU / MUU0`` and ``VWND * MSFV / MUV0`` (m/s), and ``WWND * MSFT /
  100`` (its ``DIFW`` included); ``U10M`` and ``V10M`` are plain winds and stay
  as stored.
- ``TEMP`` is ``THET * (PRES / 1000) ** 0.286`` on any file that has ``THET``
  and ``PRES`` but no ``TEMP``, and the heights for ``agl`` and ``msl`` sampling
  are built from it.

The readers keep the stored values; on such a file the ``description``
attribute of ``UWND``, ``VWND`` and ``WWND`` in :func:`arlmet.open_dataset`
gives the conversion. A Dataset or a crop written from them stays coupled, as
HYSPLIT expects of a file that still holds ``MUU0`` and ``MUV0``.

Point times
-----------

Each point's valid time comes from exactly one place:

- the ``time`` column of ``points``, if it has one;
- otherwise the ``time=`` argument, applied to every point;
- otherwise, when the input file(s) hold a single valid time, that time.

Passing both a ``time`` column and ``time=`` is ambiguous and raises
``ValueError``, as does giving neither for inputs that hold more than one time.
A point time that no input file contains raises ``ValueError`` naming the
missing times.

Here ``single_time.arl`` is a made-up file like ``met.arl`` with one time step,
and ``met.arl`` holds four:

.. ipython:: python
   :suppress:

   docs_examples.sample_file("single_time.arl", docs_examples.TIMES[:1])

.. ipython:: python

   points = pd.DataFrame({"lon": [-111.9], "lat": [40.7], "z": [850.0]})

   arlmet.sample_points("single_time.arl", points, ["TEMP"])
   arlmet.sample_points("met.arl", points, ["TEMP"], time="2024-07-18 06:00")

Sampling across multiple files
------------------------------

Pass a sequence of paths (or open files) when your points span time periods
stored in different files. Each timestamp must appear in at most one file;
arl-met routes each point to the file that contains its time and closes any
paths it opened.

.. ipython:: python
   :suppress:

   for i, hour in enumerate(["00", "06", "12"]):
       docs_examples.sample_file(f"met_{hour}.arl", docs_examples.TIMES[i : i + 1])

.. ipython:: python

   paths = ["met_00.arl", "met_06.arl", "met_12.arl"]
   points = pd.DataFrame(
       {
           "lon": [-111.9, -111.9, -111.9],
           "lat": [40.7, 40.7, 40.7],
           "z": [850.0, 850.0, 850.0],
           "time": ["2024-07-18 00:00", "2024-07-18 06:00", "2024-07-18 12:00"],
       }
   )
   arlmet.sample_points(paths, points, ["UWND", "VWND"])

You can mix open files and paths in the same sequence; files you opened stay
open, and paths are closed for you.

.. ipython:: python
   :suppress:

   os.chdir(docs_cwd)

Limitations
-----------

- Each timestamp must be present in at most one input file; overlapping times
  raise ``ValueError``.
- Point times not covered by any input file raise ``ValueError``.
- ``z_kind="agl"`` and ``"msl"`` use different field requirements depending on
  the vertical flag: pressure files (flag=2) require ``HGTS`` and ``SHGT``; sigma/hybrid
  files (flag=1/4) use hypsometric integration from ``PRSS`` and ``TEMP``;
  terrain-following files (flag=3) use the stored level heights directly; WRF
  files (flag=5) integrate as sigma files do, through the ``PRES`` field. See
  the :doc:`vertical` guide.
- Only sigma, pressure-level, hybrid and WRF files take the ground as a level
  (pressure-level files only at heights); on terrain-following files a point below
  the lowest level with data is NaN.
