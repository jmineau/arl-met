"""Tests for arlmet.metadata module."""

from arlmet.metadata import letter_to_thousands, restore_year


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
