Installation
============

Requirements
------------

arl-met requires Python 3.10 or later.

Install from GitHub
-------------------

You can install the package directly from GitHub:

.. code-block:: bash

   pip install git+https://github.com/jmineau/arl-met.git

Development Installation
------------------------

For development, clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/jmineau/arl-met.git
   cd arl-met
   pip install -e ".[dev]"

To also install documentation dependencies:

.. code-block:: bash

   pip install -e ".[dev,docs]"

Dependencies
------------

The package requires the following dependencies:

- numpy
- pandas
- xarray
- pyproj
