"""
Pytest setup for the docstring examples under ``src/`` (``--doctest-modules``).

An example reads ``"met.arl"`` or ``"met_00.arl"`` without saying where it comes
from, as the docs pages do. Each example runs in a folder of its own that holds
those files, made of made-up values by ``docs/_ext/docs_examples.py``, the same
helper the docs build uses. Real ARL files are gigabytes, so an example that
needs one is marked ``# doctest: +SKIP``.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "docs" / "_ext"))


@pytest.fixture(autouse=True)
def _doctest_files(request, tmp_path, monkeypatch):
    """Give each docstring example a folder with the small sample files in it."""
    if request.node.__class__.__name__ != "DoctestItem":
        return
    import docs_examples

    monkeypatch.chdir(tmp_path)
    docs_examples.sample_file("met.arl")
    docs_examples.sample_file("met_00.arl", docs_examples.TIMES[:1])
    docs_examples.sample_file("met_06.arl", docs_examples.TIMES[1:2])
