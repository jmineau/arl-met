"""Test basic functionality of arlmet."""

import arlmet


def test_version():
    """Test that version is defined."""
    assert hasattr(arlmet, "__version__")
    assert isinstance(arlmet.__version__, str)


def test_author():
    """Test that author is defined."""
    assert hasattr(arlmet, "__author__")
    assert isinstance(arlmet.__author__, str)


def test_email():
    """Test that email is defined."""
    assert hasattr(arlmet, "__email__")
    assert isinstance(arlmet.__email__, str)
