import string
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from math import floor, log10
from typing import Any, ClassVar

import pandas as pd

from arlmet.grid import Grid, Projection
from arlmet.vertical import VerticalAxis


def letter_to_thousands(char: str) -> int:
    """
    Convert letter to thousands digit for large grids.
    A=1000, B=2000, C=3000, etc.
    """
    if char in string.ascii_uppercase:
        return (string.ascii_uppercase.index(char) + 1) * 1000
    return 0


def restore_year(yr: str | int):
    """
    Convert 2-digit year to 4-digit year.

    Parameters
    ----------
    yr : str or int
        Year value (2-digit or 4-digit).

    Returns
    -------
    int
        4-digit year. Years < 40 are mapped to 2000+yr, otherwise 1900+yr.
        Already 4-digit years (>= 1900) are returned unchanged.
    """
    yr = int(yr)
    if yr >= 1900:
        return yr
    # This was in hysplit python code
    return 2000 + yr if (yr < 40) else 1900 + yr


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
    """
    Format a float using the ARL/Fortran-style scientific notation.
    """
    if value == 0.0:
        return " 0.0000000E+00"

    exponent = floor(log10(abs(value))) + 1
    mantissa = value / (10**exponent)
    return f"{mantissa:10.7f}E{exponent:+03d}"


def format_fixed_width_float(value: float, width: int) -> str:
    """
    Format a float into a fixed-width decimal field for index records.
    """
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
    """
    Split a total grid dimension into thousands and remainder components.
    """
    if total < 0:
        raise ValueError("Grid dimensions must be non-negative.")
    return (total // 1000) * 1000, total % 1000


@dataclass
class Header:
    "First 50 bytes of each record"

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

    FIELDS: ClassVar[dict[str, tuple[int, int, callable]]] = {
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
        """
        if len(data) != cls.N_BYTES:
            raise ValueError(
                f"{cls.__name__} must be exactly {cls.N_BYTES} bytes, got {len(data)}"
            )

        header = data.decode("ascii", errors="ignore")

        parsed = {}
        for name, (start, end, type_converter) in cls.FIELDS.items():
            # Slice the record, then apply the type conversion
            field_str = header[start:end]
            parsed[name] = type_converter(field_str)

        # Parse grid as tuple of strings
        # If the character is a letter, it indicates thousands
        # If the character is a digit, it indicates the Grid Number (00-99)
        # However I don't think the grid number is relevant for unpacking
        # In these case, no additional grid points will be addded to nx or ny
        parsed["grid"] = (
            letter_to_thousands(parsed["grid"][0]),
            letter_to_thousands(parsed["grid"][1]),
        )

        return cls(**parsed)

    def __getitem__(self, key: Any) -> Any:
        return getattr(self, key)

    @property
    def time(self) -> pd.Timestamp:
        """
        Get the timestamp for this record.

        Returns
        -------
        pd.Timestamp
            Timestamp constructed from year, month, day, and hour fields.
        """
        return pd.Timestamp(
            year=self.year, month=self.month, day=self.day, hour=self.hour
        )

    def tobytes(self) -> bytes:
        """
        Convert header to bytes.

        Returns
        -------
        bytes
            Binary representation of the header.
        """
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


@dataclass
class VarInfo:
    checksum: int
    reserved: str


@dataclass
class LvlInfo:
    """
    Information about a single vertical level.

    Parameters
    ----------
    level : int
        Level index.
    height : float
        Height in units of the vertical coordinate system.
    variables : dict[str, VarInfo]
        Dictionary mapping variable names to VarInfo (checksum and reserved).
    """

    level: int
    height: float
    variables: OrderedDict[str, VarInfo]


@dataclass
class IndexRecord:
    """
    Represents a complete ARL index record that precedes data records for each time period.
    Combines the fixed portion with the variable levels portion.

    Parameters
    ----------
    header : Header
        The header information for the index record.
    source : str
        Source identifier (4 characters).
    forecast : int
        Forecast hour.
    minutes : int
        Minutes after the hour.
    pole_lat : float
        Pole latitude position of the grid projection.
        For lat-lon grids: max latitude of the grid.
    pole_lon : float
        Pole longitude position of the grid projection.
        For lat-lon grids: max longitude of the grid.
    tangent_lat : float
        Reference latitude at which the grid spacing is defined.
        For conical and mercator projections, this is the latitude
        at which the grid touches the surface.
        For lat-lon grids: grid spacing in degrees latitude.
    tangent_lon : float
        Reference longitude at which the grid spacing is defined.
        For conical and mercator projections, this is the longitude
        at which the grid touches the surface.
        For lat-lon grids: grid spacing in degrees longitude.
    grid_size : float
        Grid spacing in km at the reference position.
        For lat-lon grids: value of zero signals that the grid is a lat-lon grid.
    orientation : float
        Angle at the reference point made by the y-axis and the local direction of north.
        For lat-lon grids: 0
    cone_angle : float
        Angle between the axis and the surface of the cone.
        For regular projections it equals the latitude at which the grid is tangent to the earth's surface.
        Stereographic: ±90, Mercator: 0, Lambert Conformal: 0 ~ 90
        For lat-lon grids: 0
    sync_x : float
        Grid x-coordinate used to equate a position on the grid with a position on earth.
        This is a unitless grid index (FORTRAN 1-based).
    sync_y : float
        Grid y-coordinate used to equate a position on the grid with a position on earth.
        This is a unitless grid index (FORTRAN 1-based).
    sync_lat : float
        Earth latitude corresponding to the grid position (sync_x, sync_y).
        For lat-lon grids: latitude of the (0,0) grid point position.
    sync_lon : float
        Earth longitude corresponding to the grid position (sync_x, sync_y).
        For lat-lon grids: longitude of the (0,0) grid point position.
    nx : int
        Number of grid points in the x-direction (columns).
    ny : int
        Number of grid points in the y-direction (rows).
    nz : int
        Number of vertical levels.
    vertical_flag : int
        Vertical coordinate system type (1=sigma, 2=pressure, 3=terrain, 4=hybrid).
    index_length : int
        Total length of the index record in bytes, including fixed and variable portions.
    levels : list[LvlInfo]
        List of levels, each containing:
            - level: Level index (0 to nz-1).
            - height: Height of the level in units of the vertical coordinate.
            - vars: Dictionary mapping variable names to VarInfo (checksum and reserved).

    Attributes
    ----------
    N_BYTES_FIXED : int
        Number of bytes in the fixed portion of the index record (108 bytes).
    time : pd.Timestamp
        The valid time of the record, calculated from the header time and minutes.
    """

    header: Header = field(repr=False)
    source: str
    forecast: int
    minutes: int
    pole_lat: float
    pole_lon: float
    tangent_lat: float
    tangent_lon: float
    grid_size: float
    orientation: float
    cone_angle: float
    sync_x: float
    sync_y: float
    sync_lat: float
    sync_lon: float
    reserved: float
    nx: int
    ny: int
    nz: int
    vertical_flag: int
    index_length: int
    levels: Sequence[LvlInfo]

    N_BYTES_FIXED: ClassVar[int] = 108

    @classmethod
    def from_position(cls, file, position) -> "IndexRecord":
        """
        Create an IndexRecord by reading from a file at a specific position.

        Parameters
        ----------
        file : file-like object
            The file to read from.
        position : int
            The byte position in the file where the index record starts.

        Returns
        -------
        IndexRecord
            The parsed IndexRecord object.
        """
        # Seek to the specified position
        file.seek(position)

        # Parse the header
        header_bytes = file.read(Header.N_BYTES)
        if not header_bytes:
            raise EOFError(
                f"Reached end of file while reading header at position {position}"
            )
        header = Header.from_bytes(header_bytes)

        if header.variable != "INDX":
            raise ValueError(
                f"Expected 'INDX' record at position {position}, found '{header.variable}'"
            )

        # Parse the index record
        fixed = IndexRecord.parse_fixed(data=file.read(IndexRecord.N_BYTES_FIXED))
        extended = file.read(fixed["index_length"] - IndexRecord.N_BYTES_FIXED)
        levels = IndexRecord.parse_extended(data=extended, nz=fixed["nz"])
        index = IndexRecord(header=header, **fixed, levels=levels)

        return index

    def serialize_fixed(self, index_length: int | None = None) -> bytes:
        """
        Serialize the fixed 108-byte portion of the index record.
        """
        if index_length is None:
            index_length = self.index_length

        values = [
            self.pole_lat,
            self.pole_lon,
            self.tangent_lat,
            self.tangent_lon,
            self.grid_size,
            self.orientation,
            self.cone_angle,
            self.sync_x,
            self.sync_y,
            self.sync_lat,
            self.sync_lon,
            self.reserved,
        ]
        proj = "".join(format_fixed_width_float(value, 7) for value in values)
        fixed = (
            f"{self.source:<4}"
            f"{self.forecast:3d}"
            f"{self.minutes:2d}"
            f"{proj}"
            f"{self.nx:3d}"
            f"{self.ny:3d}"
            f"{self.nz:3d}"
            f"{self.vertical_flag:2d}"
            f"{index_length:4d}"
        )
        if len(fixed) != self.N_BYTES_FIXED:
            raise ValueError(
                f"Fixed index serialization produced {len(fixed)} bytes, expected {self.N_BYTES_FIXED}."
            )
        return fixed.encode("ascii")

    def serialize_extended(self) -> bytes:
        """
        Serialize the variable-length level/variable portion of the index record.
        """
        chunks: list[str] = []
        for level in self.levels:
            chunks.append(format_fixed_width_float(level.height, 6))
            chunks.append(f"{len(level.variables):2d}")
            for name, info in level.variables.items():
                reserved = (info.reserved or " ")[:1]
                chunks.append(f"{name:<4}{info.checksum:3d}{reserved}")
        return "".join(chunks).encode("ascii")

    def tobytes(self) -> bytes:
        """
        Serialize the exact used bytes of the index record, including its header.
        """
        extended = self.serialize_extended()
        index_length = self.N_BYTES_FIXED + len(extended)
        self.index_length = index_length
        fixed = self.serialize_fixed(index_length=index_length)
        return self.header.tobytes() + fixed + extended

    def to_record_bytes(self, record_size: int) -> bytes:
        """
        Serialize the index record padded to one full ARL record.
        """
        raw = self.tobytes()
        if len(raw) > record_size:
            raise ValueError(
                f"Index record uses {len(raw)} bytes, which exceeds record size {record_size}."
            )
        return raw.ljust(record_size, b" ")

    @staticmethod
    def parse_fixed(data: bytes) -> dict[str, Any]:
        """
        Parse the fixed 108-byte portion of an index record from raw bytes.

        Parameters
        ----------
        data : bytes
            Raw bytes containing the fixed portion of the index record.
            108 bytes expected.

        Returns
        -------
        dict
            Parsed fields as a dictionary.
        """
        if len(data) < IndexRecord.N_BYTES_FIXED:
            raise ValueError(
                f"IndexRecord fixed portion must be at least {IndexRecord.N_BYTES_FIXED} bytes, got {len(data)}"
            )

        fields = {}

        # Parse the fixed portion of the index record
        # Format: (A4)(I3)(I2)(12F7)(3I3)(I2)(I4)
        fixed = data[: IndexRecord.N_BYTES_FIXED].decode("ascii", errors="ignore")

        fields["source"] = fixed[:4].strip()
        fields["forecast"] = int(fixed[4:7].strip())
        fields["minutes"] = int(fixed[7:9].strip())

        # Parse 12 floating point values (each 7 characters)
        proj_section = fixed[9 : 9 + 12 * 7]  # 12 * 7 = 84 characters
        proj_names = [
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
            "reserved",
        ]
        for i in range(12):
            start = i * 7
            end = start + 7
            val = float(proj_section[start:end].strip())
            if val > 180:
                # Adjust longitudes greater than 180 degrees
                val = -(360 - val)
            fields[proj_names[i]] = val

        # Parse grid dimensions (3 integers, 3 characters each)
        grid_section = fixed[93:102]
        fields["nx"] = int(grid_section[0:3].strip())
        fields["ny"] = int(grid_section[3:6].strip())
        fields["nz"] = int(grid_section[6:9].strip())

        # Parse vertical level information
        fields["vertical_flag"] = int(fixed[102:104].strip())
        fields["index_length"] = int(fixed[104:108].strip())

        return fields

    @staticmethod
    def parse_extended(data: bytes, nz: int) -> list[LvlInfo]:
        """
        Parse the variable-length portion of an index record
        containing level information.

        Parameters
        ----------
        data : bytes
            Raw bytes containing the extended portion of the index record.
        nz : int
            Number of vertical levels (from the fixed portion).

        Returns
        -------
        list[LevelInfo]
            List of LevelInfo objects for each vertical level.
        """
        extended = data.decode("ascii", errors="ignore")

        lvls = []

        # Loop through levels to extract variable info
        cursor = 0
        for i in range(nz):
            height = float(
                extended[cursor : cursor + 6].strip()
            )  # in units of vertical flag

            vars = OrderedDict()

            num_vars = int(extended[cursor + 6 : cursor + 8].strip())
            # Loop through variables for this level
            for j in range(num_vars):
                start = cursor + 8 + j * 8
                end = start + 8

                name = extended[start : start + 4].strip()
                vars[name] = VarInfo(
                    checksum=int(extended[start + 4 : start + 7].strip()),
                    reserved=extended[start + 7 : end].strip(),  # usually blank
                )

            lvls.append(LvlInfo(level=i, height=height, variables=vars))

            cursor += 8 + num_vars * 8  # Move cursor to next level

        return lvls

    @property
    def time(self) -> pd.Timestamp:
        """
        Get the valid time for this index record.

        Returns
        -------
        pd.Timestamp
            Valid time calculated from header time plus minutes offset.
        """
        return self.header.time + pd.Timedelta(minutes=self.minutes)

    @property
    def total_nx(self) -> int:
        """
        Total number of grid points in the x-direction,
        including any additional thousands from the grid letters.

        Returns
        -------
        int
            Total number of grid points in x-direction.
        """
        return self.nx + self.header.grid[0]

    @property
    def total_ny(self) -> int:
        """
        Total number of grid points in the y-direction,
        including any additional thousands from the grid letters.

        Returns
        -------
        int
            Total number of grid points in y-direction.
        """
        return self.ny + self.header.grid[1]

    @property
    def grid(self) -> Grid:
        """
        Create a Grid object based on the index record's grid parameters.

        Returns
        -------
        Grid
            The constructed Grid object.
        """
        # Build projection
        proj = Projection(
            pole_lat=self.pole_lat,
            pole_lon=self.pole_lon,
            tangent_lat=self.tangent_lat,
            tangent_lon=self.tangent_lon,
            grid_size=self.grid_size,
            orientation=self.orientation,
            cone_angle=self.cone_angle,
            sync_x=self.sync_x,
            sync_y=self.sync_y,
            sync_lat=self.sync_lat,
            sync_lon=self.sync_lon,
        )
        return Grid(
            projection=proj,
            nx=self.total_nx,
            ny=self.total_ny,
        )

    @property
    def vertical_axis(self) -> VerticalAxis:
        """
        Create the file-level vertical axis definition from the index record.
        """
        return VerticalAxis(
            flag=self.vertical_flag,
            levels=[lvl.height for lvl in self.levels],
            offset=self.reserved,
        )
