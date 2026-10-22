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

``check()`` returns one message per problem, and an empty list for a file
with none.

What is checked
---------------

The checks follow how HYSPLIT reads a file (``metset.f``):

- **Every time step has as many data records as the first.** HYSPLIT steps from
  one index record to the next by the first time step's record count, so a
  shorter time step makes it lose its place, and the file seems to end early.
- **Each time step in the file is as far after the one before it as the second
  is after the first.** HYSPLIT walks the time steps in file order and stops
  with ``meteorological data time interval varies`` when the spacing changes, as
  it does where a time step is missing, repeated, or out of order. It also finds
  a time step by its position in the file (``advpnt.f``), so where it does not
  stop it reads the wrong hours. :class:`arlmet.File` itself orders time steps
  by valid time and skips a byte-for-byte repeat (some NOAA HRRR archive files
  have one), so such a file reads well in arlmet and still needs fixing before
  HYSPLIT reads it: :func:`arlmet.concat` with the file alone, or
  :func:`arlmet.extract_subset`, writes each time step once and in order.
- **Every data record's header can be read**, and names the variable, level, and
  hour its index record lists. A record overwritten with null bytes, or copied
  from another time step, fails this.
- **No record is flagged missing**, by a negative forecast hour or the name
  ``NULL`` in its header (see :doc:`format`). HYSPLIT skips such a record, and
  stops when a time step has more than 20; arl-met reads it as NaN.

``check()`` reads each record's 50-byte header. For a 1 GB HRRR block that takes
a few seconds the first time the file is read from disk, and under a second
once it is cached.

What is not checked
-------------------

- **The values.** ``check()`` reads only the headers. To compare each record's
  packed bytes with the checksum its index record holds, call
  :meth:`arlmet.DataRecord.verify_checksum`, which reads the whole record (see
  below).
- **Hours the file should hold but does not.** A file cut after a whole time step
  looks complete. Compare :attr:`arlmet.File.times` with the hours you expect,
  for example from the file's name.

Checking the values
-------------------

Each index record holds a checksum of each data record's packed bytes.
:meth:`arlmet.DataRecord.verify_checksum` returns ``False`` when a record's
bytes do not match it:

.. code-block:: python

   with arlmet.File("gdas1.sep25.w1") as met:
       bad = [record for record in met.records if not record.verify_checksum()]

This reads every byte of the file. A stored checksum of 0 means none was
recorded, and ``verify_checksum()`` returns ``True`` for it: NOAA's GDAS and
NAM12 files store 0 for the precipitation and fluxes taken from a 6-hour
forecast, at 00, 06, 12 and 18 UTC, and those values are good.

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
