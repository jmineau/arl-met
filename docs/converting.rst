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
- :func:`arlmet.converters.from_gfs`: NCEP's GFS forecasts and GDAS analyses
  on pressure levels (``pgrb2`` files), as HYSPLIT's ``api2arl`` converts them.
- :func:`arlmet.converters.from_wrf`: WRF-ARW output (``wrfout`` files) on its
  own grid and model levels, as HYSPLIT's ``arw2arl`` converts it.

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~

- ERA5 on model levels (HYSPLIT's hybrid levels, vertical flag 4), the ensemble
  (a ``number`` dimension), and forecast-step layouts (a ``step`` dimension):
  each raises ``ValueError``. Select one ensemble member, or the valid times,
  first.
- Regridding: the converter writes the grid the input is on, which must be a
  regular latitude/longitude grid.

GFS and GDAS
------------

NOAA publishes GFS and GDAS as ARL files (:doc:`downloading`), but only some
of them: the 0.25 degree GFS on its own hybrid levels, the 1 degree GDAS on
pressure levels. :func:`~arlmet.converters.from_gfs` converts NCEP's GRIB2
``pgrb2`` files themselves, so any cycle, forecast hour or resolution (0.25,
0.5 or 1 degree) can be written on pressure levels. NOAA keeps them in a public
bucket, ``https://noaa-gfs-bdp-pds.s3.amazonaws.com``, one file per cycle and
forecast hour:

.. code-block:: text

   gfs.20250901/00/atmos/gfs.t00z.pgrb2.0p25.f003     GFS, 0.25 degree, 3-hour forecast
   gdas.20250901/00/atmos/gdas.t00z.pgrb2.1p00.f000   GDAS, 1 degree, analysis

Open each file with cfgrib (``pip install cfgrib eccodes``), which splits a
GRIB2 file into one Dataset per kind of level, and pass all the Datasets of all
the files together:

.. code-block:: python

   import cfgrib

   files = ["gdas.t18z.pgrb2.1p00.f006", "gdas.t00z.pgrb2.1p00.f000"]
   datasets = [ds for f in files for ds in cfgrib.open_datasets(f)]

Each valid time becomes one time step. An analysis has no precipitation or
fluxes, which a forecast accumulates or averages over its hours, so where two
files hold a field at the same valid time the shorter forecast is used: above,
the 00 UTC analysis gives the instantaneous fields and the 18 UTC cycle's
6-hour forecast the precipitation and fluxes, as NOAA's GDAS ARL files are
made. GFS empties its precipitation bucket every 6 hours (a 3-hour forecast
holds hours 0 to 3, a 6-hour forecast 0 to 6), which is what HYSPLIT reads in
``TPP6``. Here made-up Datasets laid out as cfgrib returns them stand in for
those two files:

.. ipython:: python

   datasets = docs_examples.gfs_sample()
   [list(d.data_vars) for d in datasets]
   ds = arlmet.converters.from_gfs(datasets)
   ds
   ds["forecast_hour"].values  # the analysis's

.. ipython:: python

   arlmet.write_dataset(ds, "gdas.arl")

The API page of :func:`~arlmet.converters.from_gfs` lists the GRIB2 variables
it converts. All the pressure levels of a file are written; to write fewer,
select them first, in the Dataset that holds ``gh``
(``d.sel(isobaricInhPa=[1000, 925, 850, 700, 500, 300, 200])``). NCEP's
planetary boundary layer height is a local GRIB2 parameter that ecCodes reads
as ``unknown``, so ``PBLH`` is not written, and the GDAS files' instantaneous
total cloud cover is missing at every point, so they give no ``TCLD``. Like
``api2arl``, it writes no ``DIF*`` records.

WRF
---

:func:`~arlmet.converters.from_wrf` converts WRF-ARW's ``wrfout`` netCDF files
on the model's own grid (Lambert conformal, polar stereographic or Mercator)
and levels, as ``arw2arl`` does with its default variables. Open the files of
one run together, joined along ``Time``, so that each hour's precipitation can
be taken from the run's accumulated totals:

.. code-block:: python

   import xarray as xr

   wrf = xr.open_mfdataset(
       "wrfout_d01_2013-06-30_*", combine="nested", concat_dim="Time"
   )

Here a small made-up run, laid out as xarray opens ``wrfout`` files, stands in
for them:

.. ipython:: python

   wrf = docs_examples.wrf_sample()
   ds = arlmet.converters.from_wrf(wrf)
   ds
   ds.arl.grid
   ds.arl.vertical_axis  # sigma levels, with the model top as offset

.. ipython:: python

   arlmet.write_dataset(ds, "wrf.arl")

The file has source ``AWRF``, which HYSPLIT reads as WRF. It keeps WRF's
staggered winds where WRF puts them, ``UWND`` on the west and ``VWND`` on the
south face of each cell, and HYSPLIT (and :func:`arlmet.sample_points`) read
them there. ``WWND`` is WRF's vertical velocity as a pressure tendency
(hPa s-1), and ``TPP1`` the precipitation since the time step before, per
hour, each with a ``DIF*`` record. The first time step has no precipitation,
as ``arw2arl`` writes it when it has no earlier file.

Converted alongside ``arw2arl`` from the same four hours of a 12 km WRF run,
every field matched its file but the two it computes, ``TEMP`` and ``WWND``,
which differ in the last bits of single precision, as Fortran's and NumPy's
arithmetic do: ``TEMP`` mostly by 1e-4 K, and by one packing step (0.125 K)
at 0.006 % of its values; ``WWND`` by at most 6e-5 hPa s-1. ``arw2arl`` keeps the
lowest 50 levels unless told otherwise; all are kept here, and
``ds.isel(level=slice(0, 50))`` keeps fewer. Its other two variable lists, the
mass-coupled time-averaged winds and TKE, are not converted.

.. ipython:: python
   :suppress:

   os.chdir(docs_cwd)
