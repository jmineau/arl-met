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
     - WRF hybrid sigma-pressure

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
- **flag=5 (WRF):** returns the ``PRES`` data variable, as HYSPLIT does, and
  raises ``ValueError`` if it is absent (see `WRF files`_ below).

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
- **flag=5 (WRF):** hypsometric integration from ``PRSS`` up through the
  ``PRES`` field, with ``TEMP``. Requires all three.

The hypsometric integration uses the temperature at the lowest level for the
surface-to-first-level layer, and the mean temperature of the bounding levels
for each layer above.

WRF files
---------

HYSPLIT's WRF converter, ``arw2arl``, writes vertical flag 5 when WRF ran with
hybrid levels (``hybrid_opt=2``), and flag 1 (sigma) otherwise. The file's
levels are WRF's eta on the mass levels, with 1.0 for the surface, and the
axis ``offset`` is WRF's model-top pressure in hPa:

.. code-block:: python

   ds = arlmet.open_dataset("wrfout_d01.arl")
   ds.arl.vertical_axis     # WrfAxis(n=34)
   ds.coords["eta"]         # eta of each level
   p = arlmet.pressure(ds)  # the PRES field
   z = arlmet.z_agl(ds)     # hypsometric, from PRSS through PRES with TEMP

On hybrid levels, eta alone does not give pressure: that also takes WRF's
hybrid coefficients, which ``arw2arl`` writes to a separate text file
(``hybrid_wrfvcoords.txt``), not to the ARL file. ``arw2arl`` always writes the
3D pressure field ``PRES``, and HYSPLIT reads pressure from it, so arl-met does
too: ``WrfAxis.to_pressure()`` raises ``ValueError`` and ``pressure()`` returns
``PRES``.

Heights come from the same hypsometric integration as sigma files, between the
``PRES`` levels. That is what HYSPLIT does for a flag-5 file unless the file
also holds WRF's dry column mass (``MUBA``, ``MUPE``) and the coefficient file
sits beside it, in which case HYSPLIT's ``prfwrf`` builds heights from the
coefficients. On a made-up WRF column the two differ by less than 1% (less
than half a meter in the lowest 2 km).

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

   waxis = arlmet.WrfAxis(levels=[1.0, 0.995, 0.98], offset=50.0)
   z = waxis.to_height_agl(surface_pressure=prss, temperature=temp, pressure=pres)

Vertical axes are immutable: ``levels`` is a read-only array and attributes
cannot be reassigned. Build a new axis instead (for example
``arlmet.VerticalAxis.from_flag(vaxis.flag, levels, offset=vaxis.offset)``).

Limitations
-----------

- WRF files (flag=5) need the ``PRES`` field for pressure and heights; the
  hybrid coefficients that would give them from eta are not in the ARL file.
- ``pressure()`` for terrain-following files (flag=3) requires a stored ``PRES``
  field; it is not derived.
