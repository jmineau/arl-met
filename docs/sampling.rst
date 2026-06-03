Sampling Points
===============

Use :func:`arlmet.sample_points` to interpolate meteorological variables at
arbitrary ``(lon, lat, z, time)`` points. This is the direct path when you have
trajectory points, station locations, or receptors and want field values at
each one without building a full Dataset.

Basic usage
-----------

Provide the points as a DataFrame (or dict) with ``lon``, ``lat``, ``z``, and
``time`` columns, plus the variables you want sampled. ``source`` can be a path
to an ARL file — :func:`arlmet.sample_points` opens and closes it for you.

.. code-block:: python

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

The returned DataFrame is a copy of ``points`` with one column added per
requested variable. The original index is preserved.

If you already have an open file, pass it instead of a path. The
:class:`arlmet.File` object also exposes the same operation as a method:

.. code-block:: python

   with arlmet.File("met.arl") as met:
       result = met.sample_points(points, ["UWND", "VWND", "TEMP"])

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
     - ``PRSS`` for sigma/hybrid files
   * - ``"agl"``
     - metres above ground level
     - flag=2: ``HGTS`` + ``SHGT``; flag=1/4: ``PRSS`` + ``TEMP``; flag=3: none
   * - ``"msl"``
     - metres above mean sea level
     - flag=2: ``HGTS``; flag=1/4: ``PRSS`` + ``TEMP`` + ``SHGT``; flag=3: ``SHGT``

.. code-block:: python

   with arlmet.File("met.arl") as met:
       result = arlmet.sample_points(
           met, points, ["TEMP"], z_kind="agl"
       )

The virtual ``"pressure"`` variable can also be requested as a sampled column:

.. code-block:: python

   result = arlmet.sample_points(met, points, ["pressure", "TEMP"])

Horizontal interpolation
------------------------

``method`` controls horizontal interpolation between grid cells:

- ``"linear"`` (default) — bilinear interpolation
- ``"nearest"`` — nearest grid cell

.. code-block:: python

   result = arlmet.sample_points(met, points, ["TEMP"], method="nearest")

Single-time files
-----------------

When the file holds a single time, the ``time`` column may be omitted and that
time is used for every point. You can also pass ``time=`` to supply or override
a single timestamp.

.. code-block:: python

   points = pd.DataFrame({"lon": [-111.9], "lat": [40.7], "z": [850.0]})

   result = arlmet.sample_points("single_time.arl", points, ["TEMP"])

Sampling across multiple files
------------------------------

Pass a sequence of paths (or open files) when your points span time periods
stored in different files. Each timestamp must appear in at most one file;
arl-met routes each point to the file that contains its time and closes any
paths it opened.

.. code-block:: python

   paths = ["met_00.arl", "met_06.arl", "met_12.arl"]
   result = arlmet.sample_points(paths, points, ["UWND", "VWND"])

You can mix open files and paths in the same sequence; files you opened stay
open, and paths are closed for you.

Limitations
-----------

- Each timestamp must be present in at most one source file; overlapping times
  raise ``ValueError``.
- Point times not covered by any source raise ``ValueError``.
- ``z_kind="agl"`` and ``"msl"`` use different field requirements depending on
  the vertical flag: pressure files (flag=2) require ``HGTS``; sigma/hybrid
  files (flag=1/4) use hypsometric integration from ``PRSS`` and ``TEMP``;
  terrain-following files (flag=3) use the stored level heights directly. See
  the :doc:`vertical` guide.
