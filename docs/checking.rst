Checking Files
==============

Use :meth:`arlmet.File.check` to find damage that does not stop a file from
opening. Opening a file already raises :class:`arlmet.ARLFormatError` when it is
cut short. A file can also hold a time step written partway, skip a time step,
or have records overwritten with other bytes, and still open. HYSPLIT then stops
partway through a run, or reads bad values. Checking a file before a run, or
after downloading or cropping it, finds these first.

.. code-block:: python

   import arlmet

   with arlmet.File("20210226_18-23_hrrr") as met:
       for problem in met.check():
           print(problem)

.. code-block:: text

   2021-02-26 19:00 has 43 of the 298 data records the first time step has.

``check()`` returns one sentence per problem, and an empty list for a file
with none.

What is checked
---------------

The checks follow how HYSPLIT reads a file (``metset.f``):

- **Every time step has as many data records as the first.** HYSPLIT steps from
  one index record to the next by the first time step's record count, so a
  shorter time step makes it lose its place, and the file seems to end early.
- **The time steps are as far apart as the first two.** HYSPLIT stops with
  ``meteorological data time interval varies`` when the spacing changes, as it
  does where a time step is missing.
- **Every data record's header can be read**, and names the variable, level, and
  hour its index record lists. A record overwritten with null bytes, or copied
  from another time step, fails this.

``check()`` reads each record's 50-byte header. For a 1 GB HRRR block that takes
a few seconds the first time the file is read from disk, and under a second
once it is cached.

What is not checked
-------------------

- **The values.** Index records hold a checksum for each record, but NOAA's GDAS
  and NAM12 files store 0 for the precipitation and fluxes taken from a 6-hour
  forecast, so ``check()`` does not compare them.
- **Hours the file should hold but does not.** A file cut after a whole time step
  looks complete. Compare :attr:`arlmet.File.times` with the hours you expect,
  for example from the file's name.

Checking a folder
-----------------

To check a folder of files, catch the files that do not open as well:

.. code-block:: python

   from pathlib import Path

   import arlmet

   for path in sorted(Path("hrrr").glob("*_hrrr")):
       try:
           with arlmet.File(path) as met:
               problems = met.check()
       except arlmet.ARLFormatError as exc:
           problems = [str(exc)]
       if problems:
           print(path.name, problems[0])
