arl-met
=======

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.22900754.svg
   :target: https://doi.org/10.5281/zenodo.22900754
   :alt: DOI

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
          <p class="arlmet-card__body">Use task-focused guides for NOAA archive downloads, spatial cropping, level selection, point sampling, vertical coordinates, concatenating files, and writing datasets back to ARL files.</p>
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

See :doc:`contributing` for contribution guidance and :doc:`development` for
notes on arl-met's internals.

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
