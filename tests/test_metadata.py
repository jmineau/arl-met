"""Tests for arlmet.metadata module."""

import io
from collections import OrderedDict
from dataclasses import FrozenInstanceError

import pytest

from arlmet.header import Header, letter_to_thousands, restore_year
from arlmet.index import IndexRecord, LvlInfo, VarInfo


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
            levels=[
                LvlInfo(
                    level=0, height=1.0, variables=OrderedDict({"PRSS": VarInfo(0, "")})
                ),
                LvlInfo(
                    level=1, height=0.5, variables=OrderedDict({"TEMP": VarInfo(0, "")})
                ),
            ],
        )

        axis = index.vertical_axis

        assert axis.flag == 4
        assert axis.offset == 25.0
        assert axis.coord_system == "hybrid"
        assert axis.levels.tolist() == [1.0, 0.5]


def _index_record(nx: int, ny: int, *, grid: tuple[int, int] | None = None):
    """Build a one-level IndexRecord for an ``nx`` x ``ny`` lat/lon grid."""
    if grid is None:
        grid = ((nx // 1000) * 1000, (ny // 1000) * 1000)
    header = Header(
        year=2025,
        month=1,
        day=1,
        hour=0,
        forecast=0,
        level=0,
        grid=grid,
        variable="INDX",
        exponent=0,
        precision=0.0,
        initial_value=0.0,
    )
    return IndexRecord(
        header=header,
        source="TEST",
        forecast=0,
        minutes=0,
        pole_lat=90.0,
        pole_lon=0.0,
        tangent_lat=0.1,
        tangent_lon=0.1,
        grid_size=0.0,
        orientation=0.0,
        cone_angle=0.0,
        sync_x=1.0,
        sync_y=1.0,
        sync_lat=0.0,
        sync_lon=0.0,
        reserved=0.0,
        nx=nx,
        ny=ny,
        nz=1,
        vertical_flag=2,
        levels=[
            LvlInfo(
                level=0, height=1000.0, variables=OrderedDict({"PRSS": VarInfo(7, "")})
            )
        ],
    )


class TestIndexRecordGridSize:
    def test_nx_ny_are_full_sizes_through_a_byte_roundtrip(self):
        index = _index_record(1799, 1059)
        raw = index.tobytes()

        # The fixed portion stores only the remainder below 1000 ...
        fixed = raw[Header.N_BYTES : Header.N_BYTES + IndexRecord.N_BYTES_FIXED]
        assert fixed[93:99] == b"799 59"
        # ... and the header's grid letters carry the thousands.
        assert raw[12:14] == b"AA"

        parsed = IndexRecord.from_position(io.BytesIO(raw), 0)
        assert (parsed.nx, parsed.ny) == (1799, 1059)
        assert (parsed.grid.nx, parsed.grid.ny) == (1799, 1059)
        assert parsed == index

    def test_mismatched_header_grid_raises(self):
        with pytest.raises(ValueError, match="thousands"):
            _index_record(1799, 1059, grid=(0, 0))

    def test_tobytes_computes_index_length_without_mutation(self):
        index = _index_record(20, 20)
        expected = IndexRecord.N_BYTES_FIXED + 8 + 8  # one level, one variable
        assert index.index_length == expected
        raw = index.tobytes()
        assert index.index_length == expected
        assert len(raw) == Header.N_BYTES + expected
        fixed = raw[Header.N_BYTES : Header.N_BYTES + IndexRecord.N_BYTES_FIXED]
        assert int(fixed[104:108]) == expected

    def test_index_record_and_header_are_frozen(self):
        index = _index_record(20, 20)
        with pytest.raises(FrozenInstanceError):
            index.nx = 30  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            index.header.forecast = 3  # type: ignore[misc]
