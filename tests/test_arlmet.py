"""Tests for arlmet main module."""

import pytest

from arlmet import ARLMet, open_dataset


class TestARLMetImports:
    """Tests for importing from arlmet package."""

    def test_arlmet_class_import(self):
        """Test that ARLMet class can be imported."""
        assert ARLMet is not None
        assert callable(ARLMet)

    def test_open_dataset_import(self):
        """Test that open_dataset function can be imported."""
        assert open_dataset is not None
        assert callable(open_dataset)


class TestARLMetInit:
    """Tests for ARLMet initialization."""

    def test_arlmet_empty_records_raises_error(self):
        """Test that initializing with empty records raises ValueError."""
        with pytest.raises(ValueError, match="No valid records provided"):
            ARLMet(records=[])
