Converting Other Models' Output
===============================

HYSPLIT reads only ARL files, and NOAA publishes them for a few products (HRRR,
NAM, GDAS, GFS, ...; see :doc:`downloading`). For any other model, HYSPLIT ships
one Fortran ``data2arl`` program per product. :mod:`arlmet.converters` holds
Python versions of them. Each takes the product's own output, opened with
xarray, and returns a Dataset in the layout :func:`arlmet.write_dataset` writes:
ARL's four-character names, the units HYSPLIT reads, rows from south to north,
and the vertical axis HYSPLIT expects for the product. You can look at it, plot
it and change it before anything is written.

The converters so far:

- :func:`arlmet.converters.from_era5`: ECMWF's ERA5 reanalysis on pressure
  levels, as HYSPLIT's ``era52arl`` converts it.

.. ipython:: python
   :suppress:

   import os

   import docs_examples

   docs_cwd = docs_examples.temp_workdir()

ERA5
----

Download the pressure-level and single-level fields from the Copernicus Climate
Data Store, for example with its ``cdsapi`` client (an account and API key are
needed). Ask for the same hours and area in both:

.. code-block:: python

   import cdsapi

   client = cdsapi.Client()
   when = {
       "product_type": ["reanalysis"],
       "year": ["2024"],
       "month": ["07"],
       "day": ["18"],
       "time": [f"{hour:02d}:00" for hour in range(24)],
       "area": [44, -116, 38, -110],  # north, west, south, east
       "data_format": "netcdf",
   }
   client.retrieve(
       "reanalysis-era5-pressure-levels",
       {
           **when,
           "variable": [
               "geopotential", "temperature", "u_component_of_wind",
               "v_component_of_wind", "vertical_velocity", "relative_humidity",
           ],
           "pressure_level": ["1000", "975", "950", "925", "900", "850", "800",
                              "700", "600", "500", "400", "300", "250", "200"],
       },
       "era5_pressure_levels.nc",
   )
   client.retrieve(
       "reanalysis-era5-single-levels",
       {
           **when,
           "variable": [
               "surface_pressure", "geopotential", "2m_temperature",
               "2m_dewpoint_temperature", "10m_u_component_of_wind",
               "10m_v_component_of_wind", "boundary_layer_height",
               "total_precipitation", "surface_sensible_heat_flux",
               "surface_latent_heat_flux", "surface_solar_radiation_downwards",
           ],
       },
       "era5_single_levels.zip",
   )

Open both with xarray. A single-level netCDF request that mixes instantaneous
fields with accumulated ones (precipitation, the fluxes) comes as a zip of two
files; open each and merge them. ``chunks={}`` keeps the values on disk until
they are written, one time step at a time:

.. code-block:: python

   import xarray as xr

   pressure_levels = xr.open_dataset("era5_pressure_levels.nc", chunks={})
   single_levels = xr.merge(
       [
           xr.open_dataset("data_stream-oper_stepType-instant.nc", chunks={}),
           xr.open_dataset("data_stream-oper_stepType-accum.nc", chunks={}),
       ],
       compat="override",
   )

GRIB files work the same way, opened with ``engine="cfgrib"`` (which needs the
``cfgrib`` package). Here two small made-up Datasets laid out as the netCDF
files are stand in for the download:

.. ipython:: python

   import arlmet

   pressure_levels, single_levels = docs_examples.era5_sample()
   pressure_levels

Convert them, look at the result, and write it:

.. ipython:: python

   ds = arlmet.converters.from_era5(pressure_levels, single_levels)
   ds
   ds["pressure"].values  # level 1 is the highest pressure
   ds["SHTF"].attrs  # W m-2, positive upward

.. ipython:: python

   arlmet.write_dataset(ds, "era5.arl")
   with arlmet.File("era5.arl") as met:
       print(met.grid, met.vertical_axis.levels)

The API page of :func:`~arlmet.converters.from_era5` lists every ERA5 variable
it converts and how. Variables it does not know are left out, so ask the Data
Store only for those you need.

How it differs from ``era52arl``
--------------------------------

The values are what HYSPLIT reads, which is not always what ``era52arl``
writes:

- geopotential becomes height through ECMWF's gravity, 9.80665 m s-2
  (``era52arl`` multiplies by 0.102);
- total cloud cover ``TCLD`` is in percent, as HYSPLIT reads it
  (``era52arl`` writes the 0 to 1 fraction);
- the heat fluxes ``SHTF`` and ``LHTF`` are positive upward, as HYSPLIT takes
  them (ERA5's, and ``era52arl``'s, are positive downward);
- the latent heat flux is named ``LHTF``, which HYSPLIT reads (``era52arl``
  names it ``LTHF``).

As ``era52arl`` does, it writes a ``DIFW`` record after each ``WWND`` and a
``DIFR`` record after ``TPP1``, which keep what 8-bit packing leaves out.

What it does not do
-------------------

- ERA5 on model levels (HYSPLIT's hybrid levels, vertical flag 4), the ensemble
  (a ``number`` dimension), and forecast-step layouts (a ``step`` dimension):
  each raises ``ValueError``. Select one ensemble member, or the valid times,
  first.
- Regridding: the converter writes the grid the input is on, which must be a
  regular latitude/longitude grid.

.. ipython:: python
   :suppress:

   os.chdir(docs_cwd)
