Vertical Coordinates
====================

ARL files store levels in one of several native coordinate systems, recorded by
the vertical flag in the file header. arl-met exposes three helpers that derive
physical vertical coordinates from a Dataset returned by
:func:`arlmet.open_dataset`:

- :func:`arlmet.pressure` — pressure (hPa) at each level
- :func:`arlmet.z_agl` — height above ground level (m)
- :func:`arlmet.z_msl` — height above mean sea level (m)

Each helper takes the Dataset and returns an :class:`xarray.DataArray`. Their
behavior depends on the file's vertical flag, which you can inspect through the
``arl`` accessor:

.. code-block:: python

   import arlmet

   ds = arlmet.open_dataset("met.arl")
   vaxis = ds.arl.vertical_axis
   print(type(vaxis).__name__)  # e.g. 'PressureAxis'
   print(vaxis.flag)            # e.g. 2
   print(vaxis.coord_system)    # e.g. 'pressure'

Vertical coordinate flags
-------------------------

.. list-table::
   :header-rows: 1

   * - Flag
     - Coordinate system
   * - 1
     - sigma
   * - 2
     - pressure
   * - 3
     - terrain-following
   * - 4
     - hybrid sigma-pressure
   * - 5
     - WRF (not implemented)

Pressure
--------

.. code-block:: python

   p = arlmet.pressure(ds)

The result depends on the vertical flag:

- **flag=2 (pressure):** returns the stored 1D ``pressure`` coordinate
  ``(level,)``.
- **flag=1 / flag=4 (sigma / hybrid):** computes a spatially varying field
  ``(time, level, y, x)`` from surface pressure (``PRSS``) using the same
  conversion as HYSPLIT's ``metlvl.f``. ``PRSS`` must be present in the dataset.
- **flag=3 (terrain-following):** returns the ``PRES`` data variable if present,
  and raises ``ValueError`` otherwise.

Height above ground level
--------------------------

.. code-block:: python

   z = arlmet.z_agl(ds)

The method used depends on the vertical flag, matching HYSPLIT's ``prfcom``
dispatcher — each coordinate system has exactly one correct method:

- **flag=2 (pressure):** ``HGTS - SHGT``. Requires ``HGTS`` and ``SHGT`` in
  the dataset. Pressure-level files always store ``HGTS``.
- **flag=1 / flag=4 (sigma / hybrid):** hypsometric integration from the
  surface to each level using ``PRSS`` (hPa) and ``TEMP`` (K). Sigma and hybrid
  files do not store ``HGTS``.
- **flag=3 (terrain-following):** returns the stored 1D ``height`` coordinate —
  terrain-following levels are already heights AGL.

The hypsometric integration uses the temperature at the lowest level for the
surface-to-first-level layer, and the mean temperature of the bounding levels
for each layer above.

Height above mean sea level
---------------------------

.. code-block:: python

   z = arlmet.z_msl(ds)

``z_msl()`` is ``z_agl(ds) + ds["SHGT"]`` (surface terrain height in metres) and
requires ``SHGT`` in the dataset.

NumPy arrays
------------

The same conversions are available on the vertical axis itself for plain NumPy
arrays. Inputs are keyword-only, and each axis uses only the ones its
coordinate system needs; a missing one raises ``ValueError`` naming it.

.. code-block:: python

   vaxis = arlmet.SigmaAxis(levels=[1.0, 0.98, 0.95], offset=0.0)
   p = vaxis.to_pressure(surface_pressure=prss)  # prss.shape + (nlev,)
   z = vaxis.to_height_agl(surface_pressure=prss, temperature=temp)

   paxis = arlmet.PressureAxis(levels=[1000.0, 925.0, 850.0])
   z = paxis.to_height_agl(hgts=hgts, terrain=shgt[..., None])

Vertical axes are immutable: ``levels`` is a read-only array and attributes
cannot be reassigned. Build a new axis instead (for example
``arlmet.VerticalAxis.from_flag(vaxis.flag, levels, offset=vaxis.offset)``).

Limitations
-----------

- WRF vertical flag 5 is not implemented; the helpers raise
  ``NotImplementedError`` for it.
- ``pressure()`` for terrain-following files (flag=3) requires a stored ``PRES``
  field; it is not derived.
