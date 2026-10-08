Installation
============

Runtime Installation
--------------------

Install the package from PyPI when you only need the core reader and writer:

.. code-block:: bash

   pip install arlmet

Install the optional remote-download dependencies when you want to fetch data
from the NOAA ARL archives:

.. code-block:: bash

   pip install "arlmet[archives]"

Development Installation
------------------------

This project uses ``uv`` for local development, dependency management, and CI.

.. code-block:: bash

   git clone https://github.com/jmineau/arl-met.git
   cd arl-met
   uv sync
   uv run pre-commit install

Common development commands
---------------------------

.. code-block:: bash

   just test             # the offline tests (just test-network runs the rest)
   just quality-check    # lint, type check, docstrings, tests
   just build-docs       # this documentation, into docs/_build/html
   just docs-serve       # preview it at http://127.0.0.1:8000, rebuilt on every save

``just`` with no arguments lists the rest.

Requirements
------------

- Python 3.11 or newer
- ``uv`` for development workflows

Installing From Source With pip
-------------------------------

If you prefer a plain editable install:

.. code-block:: bash

   git clone https://github.com/jmineau/arl-met.git
   cd arl-met
   pip install -e .
