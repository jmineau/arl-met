# Documentation

This directory contains the Sphinx documentation for the arl-met project.

## Building the Documentation

To build the documentation:

1. Install the documentation dependencies:
   ```bash
   pip install -e ".[docs]"
   ```

2. Build the HTML documentation:
   ```bash
   make docs
   ```
   
   Or from within the docs directory:
   ```bash
   cd docs
   make html
   ```

3. View the documentation:
   Open `docs/build/html/index.html` in your web browser.

## Documentation Structure

- `source/` - Documentation source files (reStructuredText)
  - `conf.py` - Sphinx configuration
  - `index.rst` - Main documentation page
  - `api.rst` - API reference
  - `installation.rst` - Installation guide
  - `quickstart.rst` - Quick start guide
  - `contributing.rst` - Contributing guide
  - `format.rst` - ARL file format documentation
  - `_static/` - Static files (images, CSS, etc.)
  - `_templates/` - Custom templates

- `build/` - Generated documentation (not committed to git)

## Theme

The documentation uses the [PyData Sphinx Theme](https://pydata-sphinx-theme.readthedocs.io/), which provides a clean, modern look and is commonly used by scientific Python projects.

## Editing the Documentation

The documentation is written in reStructuredText (`.rst` files). To edit:

1. Modify the appropriate `.rst` file in `docs/source/`
2. Rebuild the documentation with `make docs`
3. Refresh your browser to see the changes

For more information on reStructuredText syntax, see the [Sphinx documentation](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html).
