"""Tests for arlmet.records module."""

import numpy as np
import pandas as pd
import pytest

from arlmet.records import (
    Header,
    IndexRecord,
    DataRecord,
    letter_to_thousands,
    restore_year,
    unpack,
)


class TestLetterToThousands:
    """Tests for letter_to_thousands function."""

    def test_letter_a(self):
        """Test conversion of letter 'A'."""
        assert letter_to_thousands("A") == 1000

    def test_letter_b(self):
        """Test conversion of letter 'B'."""
        assert letter_to_thousands("B") == 2000

    def test_letter_z(self):
        """Test conversion of letter 'Z'."""
        assert letter_to_thousands("Z") == 26000

    def test_lowercase_returns_zero(self):
        """Test that lowercase letters return 0."""
        assert letter_to_thousands("a") == 0

    def test_digit_returns_zero(self):
        """Test that digits return 0."""
        assert letter_to_thousands("1") == 0

    def test_space_returns_zero(self):
        """Test that space returns 0."""
        assert letter_to_thousands(" ") == 0


class TestRestoreYear:
    """Tests for restore_year function."""

    def test_year_below_40(self):
        """Test year below 40 maps to 2000s."""
        assert restore_year(25) == 2025
        assert restore_year(39) == 2039
        assert restore_year(0) == 2000

    def test_year_40_and_above(self):
        """Test year 40 and above maps to 1900s."""
        assert restore_year(40) == 1940
        assert restore_year(99) == 1999
        assert restore_year(85) == 1985

    def test_four_digit_year(self):
        """Test that 4-digit years are returned unchanged."""
        assert restore_year(2025) == 2025
        assert restore_year(1985) == 1985
        assert restore_year(2000) == 2000

    def test_string_input(self):
        """Test that string inputs are converted properly."""
        assert restore_year("25") == 2025
        assert restore_year("85") == 1985
        assert restore_year("2025") == 2025


class TestUnpack:
    """Tests for unpack function."""

    def test_unpack_simple_grid(self):
        """Test unpacking a simple grid with constant values."""
        # Create a simple 3x3 grid with all 127s (differential encoding for 0 delta)
        nx, ny = 3, 3
        data = bytearray([127] * (nx * ny))  # 127 represents zero differential
        precision = 0.01
        exponent = 7
        initial_value = 100.0

        result = unpack(data, nx, ny, precision, exponent, initial_value)

        assert result.shape == (ny, nx)
        assert result.dtype == np.float32
        # All values should be the initial value since deltas are 0
        np.testing.assert_array_almost_equal(result, initial_value, decimal=2)

    def test_unpack_with_small_values(self):
        """Test that values below precision are set to zero."""
        nx, ny = 2, 2
        # 127 represents zero differential
        data = bytearray([127, 127, 127, 127])
        precision = 200.0  # Very high precision threshold
        exponent = 7
        initial_value = 0.0

        result = unpack(data, nx, ny, precision, exponent, initial_value)

        # Values below precision should be zeroed
        assert result.shape == (ny, nx)
        # All values should be close to zero due to precision threshold
        np.testing.assert_array_equal(result, 0.0)

    def test_unpack_checksum_valid(self):
        """Test unpacking with valid checksum."""
        nx, ny = 2, 2
        data = bytearray([127, 127, 127, 127])
        precision = 0.01
        exponent = 7
        initial_value = 50.0

        # Calculate the rotating checksum
        # This mimics the FORTRAN logic
        calculated_checksum = 0
        for k in range(nx * ny):
            calculated_checksum += data[k]
            if calculated_checksum >= 256:
                calculated_checksum -= 255

        # Should not raise an error
        result = unpack(
            data, nx, ny, precision, exponent, initial_value, calculated_checksum
        )
        assert result.shape == (ny, nx)

    def test_unpack_checksum_invalid(self):
        """Test unpacking with invalid checksum raises error."""
        nx, ny = 2, 2
        data = bytearray([127, 127, 127, 127])
        precision = 0.01
        exponent = 7
        initial_value = 50.0
        checksum = 999  # Invalid checksum

        with pytest.raises(ValueError, match="Checksum mismatch"):
            unpack(data, nx, ny, precision, exponent, initial_value, checksum)

    def test_unpack_different_exponents(self):
        """Test unpacking with different exponent values."""
        nx, ny = 2, 2
        data = bytearray([137, 137, 137, 137])  # 10 above neutral (127)
        precision = 0.01
        initial_value = 0.0

        # Different exponents should produce different scaling
        result_exp5 = unpack(data, nx, ny, precision, 5, initial_value)
        result_exp7 = unpack(data, nx, ny, precision, 7, initial_value)

        # Results should be different due to different scaling
        assert not np.allclose(result_exp5, result_exp7)

    def test_unpack_grid_shape(self):
        """Test that unpacked grid has correct shape."""
        nx, ny = 10, 15
        data = bytearray([127] * (nx * ny))
        precision = 0.01
        exponent = 7
        initial_value = 0.0

        result = unpack(data, nx, ny, precision, exponent, initial_value)

        assert result.shape == (ny, nx)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32


class TestIndexRecordAttrs:
    """Tests for IndexRecord.to_attrs method."""

    def test_to_attrs_returns_dict(self):
        """Test that to_attrs returns a dictionary."""
        # Create a minimal Header
        header = Header(
            year=2025,
            month=1,
            day=1,
            hour=0,
            forecast=0,
            level=0,
            grid=(0, 0),
            variable="INDX",
            exponent=7,
            precision=0.01,
            initial_value=0.0,
        )

        # Create a minimal IndexRecord
        index_record = IndexRecord(
            header=header,
            source="TEST",
            forecast_hour=0,
            minutes=0,
            pole_lat=90.0,
            pole_lon=0.0,
            tangent_lat=45.0,
            tangent_lon=0.0,
            grid_size=50.0,
            orientation=0.0,
            cone_angle=45.0,
            sync_x=1.0,
            sync_y=1.0,
            sync_lat=40.0,
            sync_lon=-100.0,
            reserved=0.0,
            nx=10,
            ny=10,
            nz=1,
            vertical_flag=2,
            index_length=200,
            levels=[],
        )

        attrs = index_record.to_attrs()

        assert isinstance(attrs, dict)
        assert "source" in attrs
        assert attrs["source"] == "TEST"
        assert "vertical_coordinate_system" in attrs
        assert attrs["vertical_coordinate_system"] == "pressure"

    def test_to_attrs_contains_expected_keys(self):
        """Test that to_attrs includes all expected keys."""
        header = Header(
            year=2025,
            month=1,
            day=1,
            hour=0,
            forecast=0,
            level=0,
            grid=(0, 0),
            variable="INDX",
            exponent=7,
            precision=0.01,
            initial_value=0.0,
        )

        index_record = IndexRecord(
            header=header,
            source="TEST",
            forecast_hour=6,
            minutes=30,
            pole_lat=90.0,
            pole_lon=0.0,
            tangent_lat=45.0,
            tangent_lon=-95.0,
            grid_size=12.0,
            orientation=0.0,
            cone_angle=45.0,
            sync_x=1.0,
            sync_y=1.0,
            sync_lat=40.0,
            sync_lon=-100.0,
            reserved=0.0,
            nx=100,
            ny=100,
            nz=5,
            vertical_flag=1,
            index_length=500,
            levels=[],
        )

        attrs = index_record.to_attrs()

        expected_keys = [
            "source",
            "forecast_hour",
            "pole_lat",
            "pole_lon",
            "tangent_lat",
            "tangent_lon",
            "grid_size",
            "orientation",
            "cone_angle",
            "sync_x",
            "sync_y",
            "sync_lat",
            "sync_lon",
            "vertical_coordinate_system",
        ]

        for key in expected_keys:
            assert key in attrs

        assert attrs["forecast_hour"] == 6
        assert attrs["grid_size"] == 12.0
        assert attrs["vertical_coordinate_system"] == "sigma"


class TestDataRecordAttrs:
    """Tests for DataRecord.to_attrs method."""

    def test_to_attrs_known_variable(self):
        """Test that to_attrs returns attributes for known variables."""
        header = Header(
            year=2025,
            month=1,
            day=1,
            hour=0,
            forecast=0,
            level=0,
            grid=(0, 0),
            variable="TEMP",
            exponent=7,
            precision=0.01,
            initial_value=273.15,
        )

        data = bytearray([127] * 100)
        record = DataRecord(header=header, data=data)

        attrs = record.to_attrs()

        assert isinstance(attrs, dict)
        assert "long_name" in attrs
        assert "units" in attrs
        assert attrs["long_name"] == "Temperature"
        assert attrs["units"] == "K"

    def test_to_attrs_unknown_variable(self):
        """Test that to_attrs returns empty dict for unknown variables."""
        header = Header(
            year=2025,
            month=1,
            day=1,
            hour=0,
            forecast=0,
            level=0,
            grid=(0, 0),
            variable="UNKN",
            exponent=7,
            precision=0.01,
            initial_value=0.0,
        )

        data = bytearray([127] * 100)
        record = DataRecord(header=header, data=data)

        attrs = record.to_attrs()

        assert isinstance(attrs, dict)
        assert len(attrs) == 0

    def test_to_attrs_surface_variable(self):
        """Test that to_attrs works for surface variables."""
        header = Header(
            year=2025,
            month=1,
            day=1,
            hour=0,
            forecast=0,
            level=0,
            grid=(0, 0),
            variable="T02M",
            exponent=7,
            precision=0.01,
            initial_value=273.15,
        )

        data = bytearray([127] * 100)
        record = DataRecord(header=header, data=data)

        attrs = record.to_attrs()

        assert "long_name" in attrs
        assert "units" in attrs
        assert attrs["long_name"] == "Temperature at 2 m"
        assert attrs["units"] == "K"
