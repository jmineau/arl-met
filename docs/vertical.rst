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
- **flag=1 (sigma):** the ``PRES`` data variable when the dataset has it, as
  HYSPLIT's ``prfsig`` uses it (see `Sigma files with PRES`_ below); otherwise
  as for flag 4.
- **flag=4 (hybrid), and sigma without ``PRES``:** computes a spatially varying
  field ``(time, level, y, x)`` from surface pressure (``PRSS``) using the same
  conversion as HYSPLIT's ``metlvl.f``. ``PRSS`` must be present in the
  dataset. A hybrid file's ``PRES``, if any, is not used, as in HYSPLIT's
  ``prfecm``.
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
  ground up through :func:`arlmet.pressure` with the virtual temperature, as
  HYSPLIT's ``prfsig`` and ``prfecm`` do (see `Sigma and hybrid heights`_).
  Requires ``PRSS`` and ``TEMP``; uses ``T02M`` (or ``TMPS``), ``SPHU`` (or
  ``RELH``) and, on sigma files, ``PRES`` when present. Sigma and hybrid files
  do not store ``HGTS``.
- **flag=3 (terrain-following):** returns the stored 1D ``height`` coordinate —
  terrain-following levels are already heights AGL.
- **flag=5 (WRF):** hypsometric integration from ``PRSS`` up through the
  ``PRES`` field, with ``TEMP``. Requires all three.

Each layer of the hypsometric integration is ``(R_d / g) * T * ln(p_below /
p_above)`` thick, with ``T`` the mean of its two bounding levels. For flag 5
the first layer, from ``PRSS`` to the first level, uses that level's ``TEMP``.

Sigma and hybrid heights
------------------------

On sigma and hybrid files arl-met builds each column as HYSPLIT's ``prfsig``
(flag 1) and ``prfecm`` (flag 4) do, so a value sampled at a height comes from
the level HYSPLIT would read:

- **The ground is level 0:** 0 m, at ``PRSS``, with the 2 m temperature
  ``T02M`` (or ``TMPS``). Without either, the ground temperature is the first
  level's ``TEMP`` brought down adiabatically, ``T1 * sigma1 ** -0.286``, with
  ``sigma1`` the first level's sigma (on hybrid files, its pressure over
  ``PRSS``).
- **Pressure** of each level above is the ``PRES`` field on a sigma file that
  has one, else the levels' nominal pressure (see `Sigma files with PRES`_).
- **Virtual temperature** ``T (1 + 0.61 q)`` sets each layer's thickness, with
  ``q`` the specific humidity ``SPHU``, else ``RELH`` converted with HYSPLIT's
  formula (``q = 0.622 * RH/100 * e_sat(T) / p``, ``e_sat = exp(21.4 - 5351 /
  T)`` hPa), else 0. The ground takes the first level's humidity.

On two HRRR columns (January and July 2021, Salt Lake Valley) the heights and
the pressure and temperature interpolated between them agree with HYSPLIT's
own ``prfsig`` within 0.006 hPa and 0.002 K from 1 m to 15 km.

:func:`arlmet.sample_points` also uses the ground as a level: between the
ground and the first level a variable is interpolated from its ground value,
so a point there is not NaN. The ground values are ``PRSS`` for ``PRES`` and
the virtual ``"pressure"``, the ground temperature above for ``TEMP``,
``U10M`` and ``V10M`` for ``UWND`` and ``VWND`` when present, zero for
``WWND`` and ``DZDT``, and the first level's value for any other variable.
Pressure is interpolated as ``ln p``, linear in height, as HYSPLIT does.

Pressure-level heights
----------------------

On pressure-level files (flag 2) :func:`arlmet.sample_points` builds each column
at heights (``z_kind="agl"`` or ``"msl"``) as HYSPLIT's ``prfprs`` does:

- **The ground is level 0:** 0 m above ground (``SHGT`` above sea level), at
  ``PRSS``, with ``T02M`` (or ``TMPS``) and ``U10M``/``V10M``. Without a 2 m
  temperature, the ground temperature is level 1's ``TEMP`` brought down
  adiabatically, ``T1 * (p1 / PRSS) ** -0.286``; without 10 m winds, the
  ground takes level 1's wind. Level 1 is the file's first pressure level,
  which may be below the ground.
- **Levels at or below the ground are left out.** Over high ground the lowest
  pressure levels have ``HGTS`` under ``SHGT`` and values the model
  extrapolated below the surface. ``prfprs`` puts nothing there, so a point
  between the ground and the first level above it is interpolated from the
  ground values and that level.

The ground values are the ones listed above for sigma files, and ``SHGT`` for
``HGTS``. Pressure stays linear in height on these files. Sampling at
pressures (``z_kind="pressure"``) uses the stored levels as they are, below
the ground included.

On NOAA's NAM 12 km and GDAS 1-degree files, at five western-US points from
1 m to 3 km above ground, ``TEMP``, ``UWND`` and ``VWND`` agree with the
compiled ``prfprs`` within 0.001; pressure differs by up to 0.5 hPa, the
difference between linear and ``ln p`` interpolation.

Sigma files with PRES
---------------------

Some sigma files also store the pressure of every level as the ``PRES`` field:
NOAA's HRRR files, and WRF output that ``arw2arl`` converts without hybrid
levels. Their sigma values are nominal: on HRRR, ``PRES`` is within about 1 hPa
of ``sigma * PRSS`` in the lowest levels but about 55 hPa higher near the
middle of the column. HYSPLIT's ``prfsig`` uses ``PRES`` whenever the file has
it, and so do :func:`arlmet.pressure`, :func:`arlmet.z_agl` and
:func:`arlmet.sample_points`. :meth:`arlmet.SigmaAxis.to_pressure` still gives
the nominal pressure; pass ``PRES`` to :meth:`arlmet.SigmaAxis.to_height_agl`
as ``pressure=`` for the real heights.

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

On hybrid levels, eta alone does not give pressure. WRF's hybrid levels follow
the terrain near the ground and flatten toward constant pressure higher up: the
pressure of a level is ``c3 * mu + c4 + p_top``, where ``mu`` is the dry-air
mass of the column (dry surface pressure minus ``p_top``) and each level has its
own pair ``c3`` and ``c4`` (``c3`` is about eta and ``c4`` is zero near the
ground). ``arw2arl`` writes those coefficients to a separate text file
(``hybrid_wrfvcoords.txt``), not to the ARL file. It always writes the 3D
pressure field ``PRES``, and HYSPLIT reads pressure from it, so arl-met does
too: ``WrfAxis.to_pressure()`` raises ``ValueError`` and ``pressure()`` returns
``PRES``.

Heights come from the same hypsometric integration as sigma files, between the
``PRES`` levels: each layer is ``(R_d / g) * T * ln(p_below / p_above)`` thick.
That is what HYSPLIT's ``prfsig`` does for a flag-5 file, unless the file also
holds WRF's dry column mass (``MUBA``, ``MUPE``) and the coefficient file sits
beside it. Then HYSPLIT's ``prfwrf`` builds the heights from the coefficients
and WRF's dry inverse density, with no temperature, the way WRF itself defines
them. On a made-up dry WRF column the two differ by less than 1% (less than half
a meter in the lowest 2 km, 74 m at 10 km). ``prfsig`` uses the virtual
temperature and starts at the ground temperature; for flag 5 arl-met uses
``TEMP`` and the first level's temperature for the first layer (unlike flags 1
and 4, see `Sigma and hybrid heights`_), so in moist air arl-met's flag-5
heights run slightly low (about 0.6% of a layer's thickness for 10 g/kg of
water vapor).

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
   p = vaxis.to_pressure(surface_pressure=prss)  # prss.shape + (nlev,), nominal
   z = vaxis.to_height_agl(surface_pressure=prss, temperature=temp)
   z = vaxis.to_height_agl(surface_pressure=prss, temperature=tv, pressure=pres)

For a sigma or hybrid axis, ``temperature`` covers every level, the ground
(level 0) included; pass the virtual temperature, with the ground temperature
at level 0, and ``PRES`` as ``pressure`` (sigma only) for HYSPLIT's heights.

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
- Only sigma, hybrid and pressure-level files take the ground as a level in
  :func:`arlmet.sample_points` (pressure-level files at heights only); on
  terrain-following (flag=3) and WRF (flag=5) files a point below the lowest
  level with data is NaN.
