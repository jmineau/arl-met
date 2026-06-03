arl-met
=======

.. image:: https://github.com/jmineau/arl-met/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/jmineau/arl-met/actions/workflows/tests.yml
   :alt: Tests

.. image:: https://github.com/jmineau/arl-met/actions/workflows/docs.yml/badge.svg
   :target: https://github.com/jmineau/arl-met/actions/workflows/docs.yml
   :alt: Documentation

.. image:: https://github.com/jmineau/arl-met/actions/workflows/quality.yml/badge.svg
   :target: https://github.com/jmineau/arl-met/actions/workflows/quality.yml
   :alt: Code Quality

.. image:: https://codecov.io/gh/jmineau/arl-met/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/jmineau/arl-met
   :alt: Code Coverage

.. image:: https://badge.fury.io/py/arlmet.svg
   :target: https://badge.fury.io/py/arlmet
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/arlmet.svg
   :target: https://pypi.org/project/arlmet/
   :alt: Python Version

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License

.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
   :target: https://github.com/astral-sh/ruff
   :alt: Ruff

.. image:: https://img.shields.io/badge/pyrefly-checked-brightgreen.svg
   :target: https://pyrefly.org
   :alt: Pyrefly

Read and analyze ARL meteorological files.

arl-met reads, writes, subsets, and samples NOAA ARL meteorology files used by
HYSPLIT and STILT. The documentation is organized by task: start with the quick
start, move to the user guides for downloading and cropping data, and then use
the API reference for the full object model.

.. raw:: html

    <div class="arlmet-card-grid">
       <a class="arlmet-card" href="getting-started.html">
          <div class="arlmet-card__eyebrow">Getting Started</div>
          <h2 class="arlmet-card__title">Install arl-met and open your first ARL file</h2>
          <p class="arlmet-card__body">Start with installation, the quickstart workflow, and the shortest path from file on disk to xarray dataset.</p>
       </a>
       <a class="arlmet-card" href="user-guides.html">
          <div class="arlmet-card__eyebrow">User Guides</div>
          <h2 class="arlmet-card__title">Download, crop, sample, and write ARL data</h2>
          <p class="arlmet-card__body">Use task-focused guides for NOAA archive downloads, spatial cropping, level selection, point sampling, vertical coordinates, and writing datasets back to ARL files.</p>
       </a>
       <a class="arlmet-card" href="api.html">
          <div class="arlmet-card__eyebrow">API Reference</div>
          <h2 class="arlmet-card__title">Explore the full function, class, and source API</h2>
          <p class="arlmet-card__body">Jump into the grouped API reference for top-level I/O, the low-level file model, metadata classes, and archive source objects.</p>
       </a>
    </div>

.. toctree::
   :hidden:
   :maxdepth: 2

   getting-started
   user-guides
   api

Development
-----------

See :doc:`contributing` for contribution guidance.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
