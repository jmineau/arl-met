"""Binary codec for the fixed 50-byte ARL record header."""

import string
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

import pandas as pd

from arlmet._time import ensure_timestamp
from arlmet.exceptions import ARLFormatError
from arlmet.grid import Grid

__all__ = ["Header"]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def restore_year(yr: str | int) -> int:
    """
    Convert 2-digit year to 4-digit year.

    Years < 40 are mapped to 2000+yr, otherwise 1900+yr.
    Already 4-digit years (>= 1900) are returned unchanged.
    """
    yr = int(yr)
    if yr >= 1900:
        return yr
    return 2000 + yr if (yr < 40) else 1900 + yr


FIRST_YEAR = 1940
LAST_YEAR = 2039


def check_year(year: int) -> int:
    """Return *year*, or raise if the two-digit year of an ARL label cannot hold it."""
    if not FIRST_YEAR <= year <= LAST_YEAR:
        raise ValueError(
            f"ARL files hold a two-digit year, so times must be from {FIRST_YEAR} "
            f"to {LAST_YEAR}; got the year {year}."
        )
    return year


def letter_to_thousands(char: str) -> int:
    """Convert a header grid character to thousands: A=1000, B=2000, …; else 0."""
    if char in string.ascii_uppercase:
        return (string.ascii_uppercase.index(char) + 1) * 1000
    return 0


def thousands_to_letter(value: int) -> str:
    """
    Convert a thousands value to its ARL grid header character.

    As HYSPLIT's ``pakset`` writes it, ``CHAR(64 + thousands)``: ``@`` for
    none, ``A`` for 1000, ``B`` for 2000, and so on. HYSPLIT's ``metset``
    reads both characters so once either is ``@`` or above, so a grid with
    one axis of 1000 or more needs ``@`` on the other (``9`` there reads as
    -7000). A grid under 1000 on both axes has the grid number ``99`` instead
    (:meth:`Header.tobytes`).
    """
    if value % 1000 != 0 or value < 0 or value > 26000:
        raise ValueError(f"Unsupported grid thousands value: {value}")
    return chr(64 + value // 1000)


def format_fortran_float(value: float) -> str:
    """
    Format a float as Fortran's ``E14.7`` writes it, as HYSPLIT's ``pakrec`` does.

    Seven significant digits, rounded from the exact binary value with ties
    to even (``f"{value:.6e}"`` rounds that way), as ``0.dddddddE+xx``.
    """
    if value == 0.0:
        return " 0.0000000E+00"
    mantissa, exponent = f"{value:.6e}".split("e")
    sign = "-" if value < 0 else " "
    digits = mantissa.lstrip("-").replace(".", "")
    return f"{sign}0.{digits}E{int(exponent) + 1:+03d}"


def format_fixed_width_float(value: float, width: int) -> str:
    """
    Format a float into a fixed-width decimal field for index records.

    The field always holds a decimal point (``11000.``, as HYSPLIT's own
    ``F6.0`` writes it): HYSPLIT reads level heights with ``F6.2``, which takes
    a field without one as having two implied decimals (``" 11000"`` is
    110.00). A value too large to fit with a point raises ``ValueError``.
    """
    if width < 2:
        raise ValueError("width must be at least 2")

    if value == 0.0:
        return "." + ("0" * (width - 1))

    for decimals in range(width, -1, -1):
        text = f"{value:.{decimals}f}" + ("." if decimals == 0 else "")
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


@dataclass(frozen=True)
class Header:
    """
    Fixed-width 50-byte header present at the start of every ARL record.

    Headers are immutable value objects; use :func:`dataclasses.replace` to
    derive a modified copy.

    Parameters
    ----------
    year, month, day, hour : int
        Valid time components stored in the record header.
    forecast : int
        Forecast hour associated with the record. The label holds two digits,
        so :meth:`tobytes` writes ``min(forecast, 99)``, as HYSPLIT's ``pakrec``
        does (an index record keeps the full hour, up to 999, in its own field).
    level : int
        ARL vertical level index.
    grid : tuple[int, int]
        Thousands part of the x and y grid sizes, as stored in the header's
        two grid characters (``(1000, 0)`` for a 1799x799 grid, written
        ``A@``; ``(0, 0)`` below 1000 points, written ``99``). The remainders live in the index record, whose
        ``nx``/``ny`` are the full sizes.
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
        """
        Parse header from raw bytes.

        Raises
        ------
        ARLFormatError
            If ``data`` is not exactly 50 bytes or a field cannot be parsed.
        """
        if len(data) != cls.N_BYTES:
            raise ARLFormatError(
                f"{cls.__name__} must be exactly {cls.N_BYTES} bytes, got {len(data)}"
            )

        header = data.decode("ascii", errors="ignore")

        parsed = {}
        for name, (start, end, type_converter) in cls.FIELDS.items():
            field_str = header[start:end]
            try:
                parsed[name] = type_converter(field_str)
            except ValueError as exc:
                raise ARLFormatError(
                    f"Invalid ARL record header: cannot parse {name} "
                    f"from {field_str!r} (bytes {start}-{end})"
                ) from exc

        parsed["grid"] = (
            letter_to_thousands(parsed["grid"][0]),
            letter_to_thousands(parsed["grid"][1]),
        )

        return cls(**parsed)

    @property
    def time(self) -> pd.Timestamp:
        """Timestamp reconstructed from the header date fields."""
        return ensure_timestamp(
            pd.Timestamp(year=self.year, month=self.month, day=self.day, hour=self.hour)
        )

    def tobytes(self) -> bytes:
        """
        Serialize the header to its fixed-width ASCII representation.

        A forecast hour above 99 is written as 99, the most the two-character
        label holds (HYSPLIT's ``pakrec`` writes ``MIN(hour, 99)``). The grid
        characters are ``99`` below 1000 points on both axes, else one per
        axis, ``@`` for no thousands (HYSPLIT's ``pakset``).
        """
        yy = check_year(self.year) % 100
        # The grid number 99 below 1000 points, else a character per axis.
        grid = (
            "".join(thousands_to_letter(value) for value in self.grid)
            if any(self.grid)
            else "99"
        )
        header = (
            f"{yy:02d}"
            f"{self.month:2d}"
            f"{self.day:2d}"
            f"{self.hour:2d}"
            f"{min(self.forecast, 99):2d}"
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
