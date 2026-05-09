API Reference
=============

Use this reference when you want the exact signature, parameters, return
values, and class interfaces for arl-met. Most workflows start with
``open_dataset()``, ``write_dataset()``, ``extract_subset()``, or
``sample_points()``.

.. currentmodule:: arlmet

High-Level I/O
--------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   open_dataset
   write_dataset

Cropping And Sampling
---------------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   extract_subset
   sample_points

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

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   Projection
   Grid
   Grid3D
   VerticalAxis

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

Remote Archive Sources
----------------------

.. currentmodule:: arlmet.sources

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   MeteorologySource
   HRRRSource
   HRRRv1Source
   NAMSource
   NAMSSource
   GDASSource
   GDAS0p5Source
   GFSSource
   NARRSource
   ReanalysisSource
