"""Tests for arlmet.metadata module."""

from collections import OrderedDict

from arlmet.metadata import (
    Header,
    IndexRecord,
    LvlInfo,
    VarInfo,
    letter_to_thousands,
    restore_year,
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


class TestIndexRecord:
    """Tests for IndexRecord helpers."""

    def test_vertical_axis_uses_index_metadata(self):
        """Test vertical axis reconstruction from index metadata."""
        index = IndexRecord(
            header=Header(
                year=2025,
                month=1,
                day=1,
                hour=0,
                forecast=0,
                level=0,
                grid=(0, 0),
                variable="INDX",
                exponent=0,
                precision=0.0,
                initial_value=0.0,
            ),
            source="TEST",
            forecast=0,
            minutes=0,
            pole_lat=90.0,
            pole_lon=0.0,
            tangent_lat=1.0,
            tangent_lon=1.0,
            grid_size=0.0,
            orientation=0.0,
            cone_angle=0.0,
            sync_x=1.0,
            sync_y=1.0,
            sync_lat=0.0,
            sync_lon=0.0,
            reserved=25.0,
            nx=10,
            ny=10,
            nz=2,
            vertical_flag=4,
            index_length=124,
            levels=[
                LvlInfo(level=0, height=1.0, variables=OrderedDict({"PRSS": VarInfo(0, "")})),
                LvlInfo(level=1, height=0.5, variables=OrderedDict({"TEMP": VarInfo(0, "")})),
            ],
        )

        axis = index.vertical_axis

        assert axis.flag == 4
        assert axis.offset == 25.0
        assert axis.coord_system == "hybrid"
        assert axis.levels.tolist() == [1.0, 0.5]
