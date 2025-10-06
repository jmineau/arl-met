Contributing
============

Thank you for your interest in contributing to arl-met!

Development Setup
-----------------

For development, clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/jmineau/arl-met.git
   cd arl-met
   pip install -e ".[dev]"

Code Formatting
---------------

This project uses `Black <https://black.readthedocs.io/>`_ for code formatting.

Format code automatically
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   make format
   # or directly with black
   black arlmet/

Check code formatting
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   make check
   # or directly with black
   black --check arlmet/

Pre-commit Hooks
----------------

Install pre-commit hooks to automatically format code before commits:

.. code-block:: bash

   pre-commit install

The pre-commit hook will automatically run Black and other checks on your code before each commit.

Building Documentation
----------------------

To build the documentation locally:

.. code-block:: bash

   # Install documentation dependencies
   pip install -e ".[docs]"

   # Build the documentation
   make docs

   # Or use sphinx-build directly
   cd docs
   sphinx-build -b html source build/html

The built documentation will be available in ``docs/build/html/index.html``.
