"""
Sphinx configuration for the arl-met docs.

The full list of settings: https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import sys
import warnings
from importlib.metadata import version as package_version
from pathlib import Path

from sphinx.deprecation import RemovedInSphinx10Warning

sys.path.insert(0, str(Path(__file__).parent / "_ext"))

warnings.filterwarnings(
    "ignore",
    category=RemovedInSphinx10Warning,
    module=r"sphinx_autodoc_typehints\..*",
)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "arl-met"
copyright = "2025, James Mineau"
author = "James Mineau"
release = package_version("arlmet")  # from git tags, via setuptools-scm
version = release
# Builds from main (and local builds) are "dev"; release builds are their version.
version_match = "dev" if (".dev" in release or "+" in release) else release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "matplotlib.sphinxext.plot_directive",  # `.. plot::` runs code, shows the figure
    "IPython.sphinxext.ipython_directive",  # `.. ipython::` runs code in an .rst page
    "IPython.sphinxext.ipython_console_highlighting",  # colors its In/Out prompts
    "sphinx_autodoc_typehints",
    "api_pages",  # _ext/api_pages.py: class pages with member tables
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_title = f"{project} {version_match}"  # not the full dev version
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "github_url": "https://github.com/jmineau/arl-met",
    "show_toc_level": 2,
    "navbar_align": "left",
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    # The version dropdown. The Documentation workflow publishes dev/ (main),
    # one folder per release and stable/, and writes switcher.json listing them.
    "switcher": {
        "json_url": "https://jmineau.github.io/arl-met/switcher.json",
        "version_match": version_match,
    },
    "check_switcher": False,  # switcher.json exists only on the deployed site
    "show_version_warning_banner": True,  # point old versions at the latest
}

# -- Extension configuration -------------------------------------------------

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_ivar = True  # what a class page's tables leave in "Attributes"

# A page per module, class, function, and class member, as in pandas'
# reference. The page templates are in _templates/autosummary/.
autodoc_default_options = {
    "member-order": "bysource",
}

# Examples run when the docs build, so they show real output and fail the build
# when they break. A `.. plot::` directive (in a docstring's Examples section or
# on any page) shows its code and the figure it draws.
plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = [("png", 100)]

# In an .rst page, `.. ipython:: python` runs its code and shows each line with
# its output; a block that raises or warns fails the build. Its figures (a line
# `@savefig name.png` above the plotting call) are saved under docs/_build.
ipython_savefig_dir = "_build/savefig"

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = True

# Intersphinx settings
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "xarray": ("https://docs.xarray.dev/en/stable", None),
}
