API Reference
=============

Use this reference when you want the exact signature, parameters, return
values, and class interfaces for arl-met. Most workflows start with
``open_dataset()``, ``write_dataset()``,
``extract_subset()``, or ``sample_points()``.

.. currentmodule:: arlmet

High-Level I/O
--------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   open_dataset
   write_dataset

File Operations
---------------

Transform ARL files: crop a subset, sample at points, or join files together.
See the :doc:`cropping`, :doc:`sampling`, and :doc:`concatenating` guides.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   extract_subset
   sample_points
   concat
   concat_by_time

Vertical Coordinates
--------------------

Derive pressure and height coordinates from an open Dataset. See the
:doc:`vertical` guide for coordinate-system details and limitations.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   pressure
   z_agl
   z_msl

Low-Level File Model
--------------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   File
   RecordSet
   DataRecord

Grid And Vertical Metadata
--------------------------

These are immutable, hashable value objects. Derive modified copies with
:func:`dataclasses.replace` (``Projection``, ``Grid``, ``GridWindow``) or by
constructing a new axis.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   Projection
   Grid
   GridWindow
   VerticalAxis
   SigmaAxis
   PressureAxis
   TerrainAxis
   HybridAxis

Binary Metadata And Packing
---------------------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   Header
   IndexRecord
   calculate_checksum
   pack
   unpack

Errors And Warnings
-------------------

Raised or emitted when file content is not valid ARL. ``ARLFormatError``
subclasses ``ValueError``, so ``except ValueError`` also catches it.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   ARLFormatError
   ARLFormatWarning

Remote Archive Sources
----------------------

.. currentmodule:: arlmet.archives

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   get_archive
   Archive
   HRRRArchive
   HRRRv1Archive
   NAMArchive
   NAMSArchive
   GDASArchive
   GDAS0p5Archive
   GFSArchive
   NARRArchive
   ReanalysisArchive

.. data:: ARCHIVES
   :no-index:

   Every registered source class by name (read-only mapping). See
   :func:`get_archive`.
