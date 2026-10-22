Checking Files
==============

Use :meth:`arlmet.File.check` to find damage that does not stop a file from
opening. Opening a file already raises :class:`arlmet.ARLFormatError` when it is
cut short. A file can also hold a time step written partway, skip a time step,
or have records overwritten with other bytes, and still open. HYSPLIT then stops
partway through a run, or reads bad values. Checking a file before a run, or
after downloading or cropping it, finds these first.

The examples run when the docs are built, on ``met.arl``, a small file of made-up
values that the build writes with :func:`arlmet.write_dataset`
(:doc:`quickstart` describes it), and on a folder ``hrrr/`` of more like it, in
which one file has a record overwritten with null bytes and one was cut short.

.. ipython:: python
   :suppress:

   import os
   from pathlib import Path

   import pandas as pd

   import docs_examples

   docs_cwd = docs_examples.temp_workdir()
   docs_examples.sample_file("met.arl")

   # hrrr/: three files of six hours each. The second has its sixth record (50 +
   # nx * ny bytes) overwritten with null bytes, and the third is cut short.
   os.mkdir("hrrr")
   for first in pd.date_range("2024-01-01", periods=3, freq="6h"):
       hours = pd.date_range(first, periods=6, freq="h")
       docs_examples.sample_file(f"hrrr/{first:%Y%m%d_%H}_hrrr", hours, upper=False)
   record = 50 + 61 * 51
   data = bytearray(Path("hrrr/20240101_06_hrrr").read_bytes())
   data[6 * record : 7 * record] = bytes(record)
   Path("hrrr/20240101_06_hrrr").write_bytes(data)
   data = Path("hrrr/20240101_12_hrrr").read_bytes()
   Path("hrrr/20240101_12_hrrr").write_bytes(data[:-500])

.. ipython:: python

   import arlmet

   with arlmet.File("hrrr/20240101_06_hrrr") as met:
       for problem in met.check():
           print(problem)

``check()`` returns one message per problem, and an empty list for a file
with none:

.. ipython:: python

   with arlmet.File("met.arl") as met:
       print(met.check())

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

.. ipython:: python

   with arlmet.File("met.arl") as met:
       bad = [record for record in met.records if not record.verify_checksum()]
   bad

This reads every byte of the file. A stored checksum of 0 means none was
recorded, and ``verify_checksum()`` returns ``True`` for it: NOAA's GDAS and
NAM12 files store 0 for the precipitation and fluxes taken from a 6-hour
forecast, at 00, 06, 12 and 18 UTC, and those values are good.

Checking a folder
-----------------

To check a folder of files, catch the files that do not open as well:

.. ipython:: python

   from pathlib import Path

   for path in sorted(Path("hrrr").glob("*_hrrr")):
       try:
           with arlmet.File(path) as met:
               problems = met.check()
       except arlmet.ARLFormatError as exc:
           problems = [str(exc)]
       if problems:
           print(path.name, problems[0])

.. ipython:: python
   :suppress:

   os.chdir(docs_cwd)
