Writing ARL Files
=================

arl-met can write an ``xarray.Dataset`` back to ARL format with
:func:`arlmet.write_dataset`.

The easiest workflow is a round-trip: open an existing ARL file, make a small
change, and write the result back out.

Round-trip from an existing ARL file
------------------------------------

.. code-block:: python

   import arlmet

   ds = arlmet.open_dataset("input.arl", levels=[0, 1, 2]).load()
   ds["TEMP"] = ds["TEMP"] - 273.15
   arlmet.write_dataset(ds, "edited.arl")

This works because datasets produced by :func:`arlmet.open_dataset` already
carry the ``arl_*`` metadata needed by the writer.

Writer requirements
-------------------

The dataset must satisfy a few ARL-specific constraints:

- it must be an ``xarray.Dataset``
- it must include ``time`` and ``level`` coordinates
- it must include a ``forecast`` coordinate aligned with ``time``
- horizontal dimensions must match the ARL grid metadata
- variable names must be 4 characters or fewer
- ``DIF*`` variables are not currently writable

The writer accepts datasets produced by :func:`arlmet.open_dataset` directly,
or datasets with equivalent metadata in ``attrs``.

Required metadata
-----------------

The writer needs:

- a source identifier via ``attrs["source"]`` or ``attrs["arl_source"]``
- grid metadata via ``attrs["grid"]`` or serialized ``arl_*`` projection attrs
- vertical metadata via ``attrs["vertical_axis"]`` or serialized ``arl_vertical_*`` attrs

If you start from ``open_dataset()``, these attributes are already present.

Omitting a slice with all-NaN values
------------------------------------

The writer treats an all-NaN horizontal slice as an omitted field for that time
and level. Partially missing slices are rejected.

.. code-block:: python

   import numpy as np
   import arlmet

   ds = arlmet.open_dataset("input.arl").load()
   ds["SPHU"][0, 0] = np.nan  # not valid if only part of the slice is NaN

If you want to omit a variable at one time and level, set the whole 2D slice to
NaN instead of only part of it.

Writing a manually constructed dataset
--------------------------------------

It is possible to write a dataset that was not created by ``open_dataset()``,
but you must provide the ARL metadata yourself.

In practice, it is usually easier and safer to start from an existing ARL file,
modify the data variables you need, and preserve the original metadata.