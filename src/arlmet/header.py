"""Binary codec for the fixed 50-byte ARL record header."""

import string
from collections.abc import Callable
from dataclasses import dataclass
from math import floor, log10
from typing import Any, ClassVar

import pandas as pd

from arlmet._time import ensure_timestamp
from arlmet.grid import Grid

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def restore_year(yr: str | int):
    """
    Convert 2-digit year to 4-digit year.

    Years < 40 are mapped to 2000+yr, otherwise 1900+yr.
    Already 4-digit years (>= 1900) are returned unchanged.
    """
    yr = int(yr)
    if yr >= 1900:
        return yr
    return 2000 + yr if (yr < 40) else 1900 + yr


def letter_to_thousands(char: str) -> int:
    """Convert letter to thousands digit for large grids. A=1000, B=2000, …"""
    if char in string.ascii_uppercase:
        return (string.ascii_uppercase.index(char) + 1) * 1000
    return 0


def thousands_to_letter(value: int) -> str:
    """
    Convert thousands value back to ARL grid header character.

    Zero is encoded as ``9`` in the files seen so far. Positive thousands
    are encoded as letters with ``A=1000``.
    """
    if value == 0:
        return "9"
    if value % 1000 != 0 or value < 0 or value > 26000:
        raise ValueError(f"Unsupported grid thousands value: {value}")
    return string.ascii_uppercase[(value // 1000) - 1]


def format_fortran_float(value: float) -> str:
    """Format a float using the ARL/Fortran-style scientific notation."""
    if value == 0.0:
        return " 0.0000000E+00"
    exponent = floor(log10(abs(value))) + 1
    mantissa = value / (10**exponent)
    return f"{mantissa:10.7f}E{exponent:+03d}"


def format_fixed_width_float(value: float, width: int) -> str:
    """Format a float into a fixed-width decimal field for index records."""
    if width < 2:
        raise ValueError("width must be at least 2")

    if value == 0.0:
        return "." + ("0" * (width - 1))

    for decimals in range(width, -1, -1):
        text = f"{value:.{decimals}f}"
        if text.startswith("0.") and len(text) - 1 <= width:
            text = text[1:]
        elif text.startswith("-0.") and len(text) - 1 <= width:
            text = "-" + text[2:]

        if len(text) <= width:
            return text.rjust(width)

    raise ValueError(f"Value {value} cannot be represented in width {width}")


def split_grid_component(total: int) -> tuple[int, int]:
    """Split a total grid dimension into thousands and remainder components."""
    if total < 0:
        raise ValueError("Grid dimensions must be non-negative.")
    return (total // 1000) * 1000, total % 1000


def record_length_from_grid(grid: Grid) -> int:
    """
    Calculate the ARL record length for a given grid.

    Record length is the fixed header length plus the number of grid points.
    """
    return Header.N_BYTES + grid.nx * grid.ny


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


@dataclass
class Header:
    """
    Fixed-width 50-byte header present at the start of every ARL record.

    Parameters
    ----------
    year, month, day, hour : int
        Valid time components stored in the record header.
    forecast : int
        Forecast hour associated with the record.
    level : int
        ARL vertical level index.
    grid : tuple[int, int]
        Thousands-encoded x and y grid header components.
    variable : str
        Four-character ARL variable name.
    exponent : int
        Differential packing exponent.
    precision : float
        Packed-data precision used during unpacking.
    initial_value : float
        Initial grid value at the start of the differential packing stream.

    Attributes
    ----------
    N_BYTES : int
        Fixed serialized size of the header.
    time : pandas.Timestamp
        Timestamp reconstructed from the header date fields.

    Methods
    -------
    from_bytes(data)
        Parse a Header from raw bytes.
    tobytes()
        Serialize the header to its fixed-width ASCII representation.
    """

    year: int
    month: int
    day: int
    hour: int
    forecast: int
    level: int
    grid: tuple[int, int]
    variable: str
    exponent: int
    precision: float
    initial_value: float

    N_BYTES: ClassVar[int] = 50

    FIELDS: ClassVar[dict[str, tuple[int, int, Callable[[str], Any]]]] = {
        "year": (0, 2, restore_year),
        "month": (2, 4, int),
        "day": (4, 6, int),
        "hour": (6, 8, int),
        "forecast": (8, 10, int),
        "level": (10, 12, int),
        "grid": (12, 14, str),
        "variable": (14, 18, str),
        "exponent": (18, 22, int),
        "precision": (22, 36, float),
        "initial_value": (36, 50, float),
    }

    @classmethod
    def from_bytes(cls, data: bytes) -> "Header":
        """Parse header from raw bytes."""
        if len(data) != cls.N_BYTES:
            raise ValueError(
                f"{cls.__name__} must be exactly {cls.N_BYTES} bytes, got {len(data)}"
            )

        header = data.decode("ascii", errors="ignore")

        parsed = {}
        for name, (start, end, type_converter) in cls.FIELDS.items():
            field_str = header[start:end]
            parsed[name] = type_converter(field_str)

        parsed["grid"] = (
            letter_to_thousands(parsed["grid"][0]),
            letter_to_thousands(parsed["grid"][1]),
        )

        return cls(**parsed)

    def __getitem__(self, key: Any) -> Any:
        return getattr(self, key)

    @property
    def time(self) -> pd.Timestamp:
        """Timestamp reconstructed from the header date fields."""
        return ensure_timestamp(
            pd.Timestamp(year=self.year, month=self.month, day=self.day, hour=self.hour)
        )

    def tobytes(self) -> bytes:
        """Serialize the header to its fixed-width ASCII representation."""
        yy = self.year % 100
        grid = "".join(thousands_to_letter(value) for value in self.grid)
        header = (
            f"{yy:02d}"
            f"{self.month:2d}"
            f"{self.day:2d}"
            f"{self.hour:2d}"
            f"{self.forecast:2d}"
            f"{self.level:2d}"
            f"{grid:>2}"
            f"{self.variable:<4}"
            f"{self.exponent:4d}"
            f"{format_fortran_float(self.precision)}"
            f"{format_fortran_float(self.initial_value)}"
        )
        if len(header) != self.N_BYTES:
            raise ValueError(
                f"Header serialization produced {len(header)} bytes, expected {self.N_BYTES}."
            )
        return header.encode("ascii")
