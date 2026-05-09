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

   pip install "arlmet[sources]"

Development Installation
------------------------

This project uses ``uv`` for local development, dependency management, and CI.

.. code-block:: bash

   git clone https://github.com/jmineau/arl-met.git
   cd arl-met
   uv sync --dev

Common development commands
---------------------------

.. code-block:: bash

   uv run pytest -q
   uv run ruff check .
   uv run pyright src/arlmet
   uv run sphinx-build -M html docs docs/_build
   uv run docstr-coverage src/arlmet --skip-magic --skip-init --skip-property --fail-under 95

Requirements
------------

- Python 3.10 or newer
- ``uv`` for development workflows

Installing From Source With pip
-------------------------------

If you prefer a plain editable install:

.. code-block:: bash

   git clone https://github.com/jmineau/arl-met.git
   cd arl-met
   pip install -e .
