API Stability
=============

arlmet follows `Semantic Versioning <https://semver.org/>`_ with
`PEP 440 <https://peps.python.org/pep-0440/>`_ version numbers. This page says
what counts as the public API, and what may change between releases.

The public API
--------------

The public API is:

- every name in ``arlmet.__all__`` (``import arlmet; arlmet.<name>``),
  including the documented methods and attributes of the classes it exports;
- the :mod:`arlmet.sources` module (``pip install "arlmet[sources]"``);
- the ``ds.arl`` Dataset accessor and the ``engine="arl"`` xarray backend;
- the layout of Datasets returned by :func:`arlmet.open_dataset` (dimension,
  coordinate, and attribute names) and the names of files cached by
  :meth:`arlmet.sources.MeteorologySource.fetch`.

One exception: how Datasets represent forecast hours (today the
``forecast_hour(time)`` variable, which holds only each time step's
index-record forecast) is **provisional** and may change during beta. ARL
records carry their own forecast hours, which differ within a time step for
accumulated fields such as precipitation, and the Dataset API does not yet
represent them (`#40 <https://github.com/jmineau/arl-met/issues/40>`_).

Anything else is internal and may change in any release: submodules and names
not listed in their module's ``__all__``, names starting with an underscore,
and the ``arlmet.ops`` and ``arlmet.xarray`` subpackage paths (import from
``arlmet`` instead).

What changes when
-----------------

**Beta (0.1.0b*)**: the public API is frozen. Beta releases fix bugs, add
features, and may still make a breaking change if a real problem is found,
but only when there is no other way, and always with a ``**Breaking:**``
entry in the `changelog <https://github.com/jmineau/arl-met/blob/main/CHANGELOG.md>`_.

**0.x releases (from 0.1.0)**: a breaking change bumps the minor version
(``0.1`` → ``0.2``); bug fixes and backward-compatible features bump the
patch version (``0.1.0`` → ``0.1.1``). Removals are announced with a
``DeprecationWarning`` for at least one minor release first.

**1.0 and later**: breaking changes only in a new major version.

Behavior that is a bug fix, not a break
---------------------------------------

Making arlmet match the ARL format or HYSPLIT where it did not is a bug fix,
even if it changes results. For example, rejecting a malformed file that used
to be read silently, or correcting a coordinate that was computed wrongly.
Such changes are listed under "Fixed" in the changelog.

Supported versions
------------------

arlmet supports the CPython versions that are not end-of-life and have NumPy
wheels (currently 3.11 through 3.14), on Linux, macOS, and Windows. Support
for a Python version is dropped in a minor release after it reaches
end-of-life.
