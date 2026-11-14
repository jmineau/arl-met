Probing a File
===============

Use :func:`arlmet.probe` for a cheap summary of one ARL file: its first valid
time, the time between its first two time steps, how many time steps it
holds, its grid, source, and forecast hour, and whether its size divides into
a whole number of time steps or leaves a remainder (a truncated file). It
reads only the first two index records and :func:`os.stat`'s the file, never
a full index record scan or a data record, so it costs about what opening the
file waits on, not what reading every time step does.

The example below runs on ``met.arl``, a small file of made-up values that the
build writes with :func:`arlmet.write_dataset` (:doc:`quickstart` describes
it), and on two copies of it: one cut short, one with its first time step
repeated byte for byte, as some NOAA HRRR archive files do.

.. ipython:: python
   :suppress:

   import os
   from pathlib import Path

   import docs_examples

   docs_cwd = docs_examples.temp_workdir()
   docs_examples.sample_file("met.arl")

   whole = Path("met.arl").read_bytes()
   step = len(whole) // 4  # met.arl has four time steps, the same length
   Path("truncated.arl").write_bytes(whole[: 2 * step - 10])
   Path("repeated.arl").write_bytes(whole[:step] + whole)

.. ipython:: python

   import arlmet

   arlmet.probe("met.arl")

A probe's result is a frozen :class:`arlmet.Probe`: ``first`` and ``step`` are
the valid time and spacing of the first two time steps, ``n_steps`` is how
many the file holds, ``grid``, ``source`` and ``forecast`` come from the
first index record, and ``whole`` says whether the file's size matches an
exact number of time steps.

Whole or truncated
-------------------

``n_steps`` and ``whole`` come from one division: the first time step's
length (its index record plus one data record per variable per level) into
the file's size. A remainder means the file is cut short partway through a
time step:

.. ipython:: python

   arlmet.probe("truncated.arl")

``truncated.arl`` is ``met.arl`` with its second time step cut 10 bytes
short: ``n_steps`` counts only the one whole time step that fits, ``whole``
is ``False``, and ``step`` is ``None`` because there is no second whole time
step to compare the first with.

What it does not check
-----------------------

A probe reads two time steps and assumes every other one is the same length
as the first. Two things it does not catch, that :meth:`arlmet.File.check`
does by reading every index record in the file:

- **A later time step with a different set of variables.** If the tenth time
  step holds one fewer variable than the first, every count a probe computes
  past that point is off, but the probe itself has no way to know: it never
  reads a tenth index record.
- **A byte-identical repeated time step**, which some NOAA HRRR archive files
  have and :class:`arlmet.File` quietly skips when it opens one. A probe
  reads the repeated copy as if it were a genuine second time step:

  .. ipython:: python

     arlmet.probe("repeated.arl")

  ``repeated.arl`` is ``met.arl`` with its first time step copied right after
  itself. The probe's ``step`` is zero, since the "second" time step it read
  is really the first one again; a real archive file's genuine second time
  step would be read where the repeated copy's position happens to put it,
  which is ``n_steps`` counting one too many and every later time step
  (beyond the two a probe reads) mislocated too.

For either of these, or to compare each record's checksum with its packed
bytes, use :meth:`arlmet.File.check` (:doc:`checking`) or
:meth:`arlmet.DataRecord.verify_checksum`. Treat a probe's result as
unverified until one of those has run on the file.

When to use it
---------------

:func:`arlmet.inventory` opens every file in a directory and reads every one of
its time steps, to build a table you can query by time window
(:doc:`inventorying`); that is the right tool for finding the files a run needs.
A probe is for when you already have one file's path and want to know what
it holds, or whether it looks whole, without paying for a directory's worth
of reads: for example, a quick look at a single downloaded file, or a
lightweight first pass before deciding whether a file is worth the fuller
read ``inventory`` or ``check`` give it.

.. ipython:: python
   :suppress:

   os.chdir(docs_cwd)
