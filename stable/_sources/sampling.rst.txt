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
     - flag=2: ``HGTS`` + ``SHGT``; flag=1/4: ``PRSS`` + ``TEMP``; flag=3: ``SHGT``
       or ``PRSS`` + ``TEMP`` (none with ``model_top=None``);
       flag=5: ``PRSS`` + ``PRES`` + ``TEMP``
   * - ``"msl"``
     - meters above mean sea level
     - flag=2: ``HGTS`` + ``SHGT``; flag=1/4: ``PRSS`` + ``TEMP`` + ``SHGT``; flag=3: ``SHGT``
       (and ``TEMP`` without ``PRSS``);
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

Between grid points at heights (``agl``, ``msl``) the order is HYSPLIT's: the
column at each of the four grid points around a point, with that grid point's
own ground and level heights, is interpolated to the point's ``z``, and the
four values are then blended bilinearly. (The column is the one on HYSPLIT's
internal levels by default and the file's own levels with ``model_top=None``;
the section below says which is which.) Where the levels' heights differ between neighboring
grid points (over uneven ground, or wherever the pressure field varies) this
differs from blending each level first and interpolating the blended column:
by up to 2 K and 1 m/s in the lowest 50 m on a 27 km WRF grid, and less on
finer grids and higher up. At a grid point the two are the same.

A grid point whose column ends below the height does not count: the other
grid points' weights are rescaled to sum to 1, and the point is NaN only where
none of them has a value. At pressures, which HYSPLIT does not sample, and at
native levels, each level is blended first and the blended column
interpolated, so a pressure just above the ground at the point is read near
the ground of the point's own column. An ``msl`` height is a height above the ground at the point itself
(``z`` less the ``SHGT`` there), and each grid point's column is read at that
height above its own ground.

.. ipython:: python

   result = arlmet.sample_points("met.arl", points, ["TEMP"], method="nearest")
   result

Matching a HYSPLIT run
----------------------

HYSPLIT does not read the met at a particle's own height. It first fills its
own internal levels at every grid point, then reads the particle between two
of them. Its value at a point therefore depends on a setting of the run: the
top of the model domain, the line of the ``CONTROL`` file after the vertical
motion option (PYSTILT's ``z_top``, 25000 m by default). At heights (``agl``,
``msl``) ``sample_points`` gives the value a run with that top uses, and
``model_top`` defaults to 25000, so a run with PYSTILT's default top needs no
argument. A run with another top passes it as ``model_top``:

- **Internal levels.** For the run's top and the file's vertical grid,
  HYSPLIT's ``metlvl`` picks one of six grids of heights ``a k**2 + b k + c``
  (k = 1, 2, ...): on an HRRR file with a 25 km top, 41 levels at 10, 50, 120,
  220 m, ...; on GFS 0.25°, 71 levels at 10, 30, 60 m, .... Between two levels
  the value is linear, and below the first level (10 m, or 5 m on the finest
  grid) it is that level's value.
- **Terrain-following sigma.** The levels follow the terrain: above a grid
  point with terrain ``zsfc`` each is ``(ZMDL - zsfc) / ZMDL`` times as high,
  where ``ZMDL`` is ``max(model_top, 25000)``, and a point's height is a
  fraction of ``ZMDL`` less the terrain at the point. A point 500 m above
  ground is read about 496 m above a neighboring grid point whose ground is
  200 m higher.
- **Above a file's top.** HYSPLIT fills its levels above a column's top with
  the ``TEMP``, ``UWND`` and ``VWND`` of the last level below it, and with a
  pressure that falls from there hydrostatically. Other variables are NaN
  there.
- On pressure-level files a level between the ground and the first internal
  level is left out, as ``prfprs`` leaves it out; a sigma, hybrid or WRF file
  without ``SHGT`` takes HYSPLIT's terrain estimate from ``PRSS`` and the
  first level's temperature.

.. ipython:: python

   low = points.assign(z=[5.0, 300.0])  # meters above ground level
   arlmet.sample_points("met.arl", low, ["TEMP", "pressure"], z_kind="agl")
   arlmet.sample_points(
       "met.arl", low, ["TEMP", "pressure"], z_kind="agl", model_top=10000
   )

The default applies at heights only. At pressures and native levels
``model_top`` plays no part and the default is ignored; passing a value there
raises ``ValueError``.

**Terrain-following (flag 3) files.** Their levels are heights over flat
ground, ``Z*``. HYSPLIT's ``prfter`` puts a level ``Z* (1 - zsfc / ZMDLT)``
above a grid point with terrain ``zsfc``, where ``ZMDLT`` is 20000 m (RAMS's
model top; on a file whose source is ``RAMS``, ``ZMDL``), and ``sample_points``
does the same. Every variable is linear in height there, pressure too, and the
virtual ``pressure`` comes from the file's ``PRES`` field (NaN without one).
A file without ``PRSS`` takes HYSPLIT's surface pressure from ``SHGT``. No
real flag 3 file was on hand: on 9 made-up ones (tops of 3, 10, 25 and 40 km)
``sample_points`` agrees with HYSPLIT's routines to 0.0002 K and m/s and 0.002
hPa. Before arl-met followed ``prfter``, it read a flag 3 file's stored levels
as heights above the ground, which is what ``model_top=None`` still does. On
the made-up file with terrain up to 3300 m and a 25 km top that was off by up
to 5.2 K and 3.9 m/s between 10 m and 5 km (0.5 K and 0.45 m/s on average) and
9.2 K above 5 km, with no value below the first stored level or for pressure.

**The default matches a run whose top is 25000 m, not one with a lower top.**
``ZMDL``, where the sigma surfaces go flat, is 25000 m for every top up to
25 km, but the internal levels are not the same: ``metlvl`` counts the file's
levels up to the run's top and picks the first grid with more levels than that,
so a lower top usually gives a different grid (other heights, not just fewer
levels). On 11 real files (200 points between grid points, compared with
HYSPLIT's routines run with that top), a 10 km or 3 km top picks a different
grid than 25 km on 9 of them; HRRR, for example, has levels at 10, 30, 70, 130
m, ... rather than 10, 50, 120, 220 m, ..., and the default then differs from
that run's value by up to 3.4 K, 3.4 m/s and 0.7 hPa (the largest of
the 9 files, below the run's top layer). On the other 2 (GDAS 0.5° and GFS 0.25°,
whose grid is the same at every top), the lower top's levels are the first of the 25 km
grid, and the default agrees with that run to 0.007 (K, m/s, hPa) below the
top layer. A run whose top is not 25000 m should pass it as ``model_top``.

``model_top=None`` reads each grid point's column at the point's own height
above that grid point's ground, between the file's own levels, and above a
file's top gives NaN. That was what ``sample_points`` did at heights before it
followed HYSPLIT's internal levels, and it is no value HYSPLIT would give.
Against HYSPLIT's own routines run on real files with a 25 km top (200 points
between grid points of one time step, at heights from 2 m up), the largest
differences of ``model_top=None`` are:

.. list-table::
   :header-rows: 1

   * - File
     - Below 10 m: K, m/s, hPa
     - 10 m to 5 km: K, m/s, hPa
   * - GDAS 1° (pressure levels)
     - 3.7, 0.47, 1.9
     - 3.8, 1.5, 1.1
   * - GDAS 0.5° (hybrid, no ``SHGT``)
     - 0.30, 0.63, 1.1
     - 0.49, 0.48, 0.44
   * - GFS 0.25° (hybrid)
     - 0.67, 0.79, 1.1
     - 0.73, 0.70, 0.13
   * - NAM 12 km (pressure levels)
     - 1.0, 0.79, 1.2
     - 2.1, 1.7, 0.62
   * - NAMS CONUS (hybrid)
     - 0.49, 0.22, 1.0
     - 2.2, 1.9, 0.35
   * - WRF 27 km (two days)
     - 3.1, 2.2, 1.1
     - 3.8, 3.5, 0.43
   * - HRRR (2020) and HRRR v1 (2017)
     - 4.6, 0.94, 1.1
     - 1.0, 2.4, 0.39
   * - WRF 1 km
     - 0.30, 1.9, 0.99
     - 0.14, 0.59, 0.084

The means are smaller: 0.05 to 0.6 K and 0.03 to 0.3 m/s below 10 m, 0.007
to 0.1 K and 0.01 to 0.1 m/s from 10 m to 5 km. Pressure below 10 m is off by 0.6 to 0.8 hPa
on average, because HYSPLIT's value there is the one at 10 m. Most of the
difference comes from the internal levels; below 5 km the terrain-following
sigma adds at most 0.07 K and 0.57 m/s (on the 1° GDAS grid) and less on finer
grids. With the default the same points agree with HYSPLIT's routines to
about 0.01 K, 0.01 m/s and 0.01 hPa on most files.

What the default does not cover:

- The internal levels come from the sampled file's vertical grid. A run on
  several met grids takes them from the grid with the most levels, and a run
  with a ``ZSG_LEVS.IN`` file takes its own.
- On a file without ``U10M`` and ``V10M`` HYSPLIT puts a log-law wind at its
  first internal level, which depends on the run's roughness length;
  ``sample_points`` does not.
- In HYSPLIT's top internal layer (from about 24 km up with a 25 km top) its
  advection takes the top level's value while its particle output
  interpolates; ``sample_points`` interpolates.
- A run with a negative top, which HYSPLIT takes as ``ZMDLT`` for a
  terrain-following file (COAMPS's 34800 m): ``model_top`` must be positive.
  HYSPLIT's own ``metlvl`` picks a one-level grid for such a top.

Earth-relative winds
--------------------

Winds in ARL files on projected grids (Lambert conformal, polar
stereographic, Mercator) are stored relative to the grid axes, which is what
HYSPLIT expects. Sampled ``UWND``/``VWND`` and ``U10M``/``V10M`` are therefore
not directly comparable with observed winds. Pass ``earth_relative=True`` to
rotate each pair so that ``u`` points east and ``v`` north, using the meridian
convergence of the file's grid at each point (a rotated grid's ``orientation``
included). Both components of a pair must
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

Each wind is blended from the columns of the four U (or V) grid points around
that position, their ground values (``U10M`` or ``V10M``) and their levels'
heights included, as HYSPLIT builds a profile at each grid point before
interpolating; an ``msl`` height is still measured from the terrain at the
point itself. With ``method="nearest"`` the winds are those at
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
  terrain-following files (flag=3) use the stored level heights, scaled as
  HYSPLIT's ``prfter`` scales them (with ``model_top=None``, as they are); WRF
  files (flag=5) integrate as sigma files do, through the ``PRES`` field. See
  the :doc:`vertical` guide.
- A point outside the grid is NaN. A point within 1e-6 cells of the grid's edge
  counts as on it, so the edge nodes of a projected grid, sampled at the
  longitude and latitude arl-met reports for them, have values even though
  mapping them back to the grid lands about 1e-12 cells outside.
- Sigma, hybrid and WRF files take the ground as a level, pressure-level files
  at heights, and terrain-following files on HYSPLIT's internal levels; on a
  terrain-following file with ``model_top=None`` a point below the lowest level
  with data is NaN.
