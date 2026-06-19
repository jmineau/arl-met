Concatenating ARL Files
=======================

Use :func:`arlmet.concat` to join several ARL files into one, and
:func:`arlmet.concat_by_time` to do it in batch across a directory. The common
reason is HYSPLIT's input limit: a simulation accepts at most 12 meteorological
files when a single grid is used (`Compilation Limits
<https://www.ready.noaa.gov/hysplitusersguide/S441.htm>`_). Combining many short
files (e.g. 6-hourly) into fewer long ones (e.g. daily) keeps a long run under
that cap.

ARL files are flat streams of fixed-size records, so joining them is a
byte-level append — the same result as ``cat a.arl b.arl > out.arl`` — with no
repacking. Every record, including ``DIF*`` records and checksums, is preserved
exactly.

Joining a list of files
-----------------------

Pass the input paths and an output path. By default the inputs are ordered by
their earliest valid time, so the output is chronological regardless of the
order you list them in.

.. code-block:: python

   import arlmet

   arlmet.concat(
       ["20240101_00_hrrr", "20240101_06_hrrr", "20240101_12_hrrr"],
       "20240101_hrrr",
   )

``concat()`` returns the new file opened in read mode, so you can chain straight
into analysis. Use it as a context manager (or call ``.close()``) when you keep
the return value; ignore it if you only need the file on disk.

.. code-block:: python

   with arlmet.concat(paths, "20240101_hrrr") as combined:
       print(combined.times)

Pass ``sort=False`` to join the files in exactly the order given, like ``cat``.

What is validated
-----------------

Before writing, the inputs are scanned and joined only if they form one coherent
record stream. ``concat()`` raises ``ValueError`` when:

- the inputs disagree on grid or vertical axis (different grids produce different
  record lengths, which would corrupt the stream)
- the same valid time appears in more than one input (arl-met cannot read a file
  with duplicate times, and HYSPLIT behaviour on repeats is undefined)
- a source file is empty, or the output path is also one of the inputs

Batch concatenation by time
---------------------------

:func:`arlmet.concat_by_time` groups every ARL file in a directory into
time-binned chunks and concatenates each group. Each file is assigned to a bin by
its **first valid time, read from the file's index record** — not parsed from the
filename — so it is robust to any naming scheme.

.. code-block:: python

   import arlmet

   arlmet.concat_by_time(
       "hrrr/",                       # directory to scan
       "daily/",                      # output directory (created if missing)
       freq="1D",                     # one output file per day
       pattern="*_hrrr",              # which files to read
       template="{time:%Y%m%d}_hrrr",  # how to name each output
   )

``freq`` is a fixed-frequency pandas offset alias giving the size of each output
chunk: ``"1D"`` is one file per day, ``"6h"`` one per six hours, and so on. Each
file is binned by its first valid time floored to this frequency, so ``freq``
should be at least as long as any single input file's time span.

``template`` is a ``str.format`` string for the output filenames, given the bin
start time as ``time`` (a :class:`pandas.Timestamp`). ``concat_by_time()``
returns the list of written paths, in time order.

Limit the range with ``time_range`` to skip files whose first valid time falls
outside an inclusive ``(start, end)`` window:

.. code-block:: python

   arlmet.concat_by_time(
       "hrrr/",
       "daily/",
       freq="1D",
       pattern="*_hrrr",
       template="{time:%Y%m%d}_hrrr",
       time_range=("2024-01-01", "2024-01-31 23:00"),
   )

Limitations
-----------

- Concatenated files must share one grid and one vertical axis.
- Valid times must not repeat across the inputs of a single output file.
- ``concat_by_time`` bins each file by its first valid time, so ``freq`` should
  be no shorter than a single input file's span (e.g. use ``freq="1D"`` for
  6-hourly inputs, not ``freq="1h"``).
- ``pattern`` should match only ARL files; a matched file that cannot be read as
  ARL raises ``ValueError``.
