"""
ARL record parsing and data unpacking.

This module provides classes and functions for parsing ARL file records,
including index records, data records, headers, and the differential unpacking
algorithm. It handles the binary format used in ARL meteorological files.
"""

import string
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import pandas as pd

# ARL meteorological variable definitions
ARL_SURFACE_VARIABLES = {
    "U10M": ("U-component of wind at 10 m", "m/s"),
    "V10M": ("V-component of wind at 10 m", "m/s"),
    "T02M": ("Temperature at 2 m", "K"),
    "PBLH": ("Boundary Layer Height", "m"),
    "PRSS": ("Pressure at surface", "hPa"),
    "MSLP": ("Pressure at mean sea level", "hPa"),
    "TMPS": ("Temperature at surface", "K"),
    "USTR": ("Friction Velocity", "m/s"),
    "TSTR": ("Friction Temperature", "K"),
    "RGHS": ("Surface Roughness", "m"),
    "UMOF": ("U-Momentum flux", "N/m2"),
    "VMOF": ("V-Momentum flux", "N/m2"),
    "SHTF": ("Sfc sensible heat flux", "W/m2"),
    "LTHF": ("Latent heat flux", "W/m2"),
    "DSWF": ("Downward short wave flux", "W/m2"),
    "RH2M": ("Relative humidity at 2 m", "%"),
    "SPH2": ("Specific humidity at 2 m", "kg/kg"),
    "CAPE": ("Convective Available Potential Energy", "J/kg"),
    "TCLD": ("Total cloud cover", "%"),
    "TPPA": ("Total precipitation for whole dataset", "m"),
    "TPPD": ("Total precipitation (24-h)", "m"),
    "TPPT": ("Total precipitation (12-h)", "m"),
    "TPP6": ("Total precipitation (6-h)", "m"),
    "TPP3": ("Total precipitation (3-h)", "m"),
    "TPP1": ("Total precipitation (1-h)", "m"),
    "PRT6": ("Precipitation Rate (6-h)", "m/minute"),
    "PRT3": ("Precipitation Rate (3-h)", "m/minute"),
}

ARL_UPPER_VARIABLES = {
    "UWND": ("U wind component (respect to grid)", "m/s"),
    "VWND": ("V wind component (respect to grid)", "m/s"),
    "HGTS": ("Geopotential height", "gpm"),
    "TEMP": ("Temperature", "K"),
    "WWND": ("Pressure vertical velocity", "hPa/s"),
    "RELH": ("Relative Humidity", "%"),
    "SPHU": ("Specific Humidity", "kg/kg"),
    "DZDT": ("vertical velocity", "m/s"),
    "TKEN": ("turbulent kinetic energy", "m2/s2"),
}


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


def unpack(
    data: bytes | bytearray,
    nx: int,
    ny: int,
    precision: float,
    exponent: int,
    initial_value: float,
    checksum: int | None = None,
) -> np.ndarray:
    """
    Unpacks a differentially packed byte array into a 2D numpy array.

    This function is a vectorized Python translation of the HYSPLIT
    FORTRAN PAKINP subroutine. It uses a differential unpacking scheme
    where each value is derived from the previous one. The implementation
    is optimized with numpy for high performance.

    Parameters
    ----------
    data : bytes or bytearray
        Packed input data as a 1D array of bytes.
    nx : int
        The number of columns in the full data grid.
    ny : int
        The number of rows in the full data grid.
    precision : float
        Precision of the packed data. Values with an absolute value smaller
        than this will be set to zero.
    exponent : int
        The packing scaling exponent.
    initial_value : float
        The initial real value at the grid position (0,0).
    checksum : int, optional
        If provided, a checksum is calculated over `data` and compared
        against this value. A `ValueError` is raised on mismatch.
        If None (default), the check is skipped.

    Returns
    -------
    np.ndarray
        The unpacked 2D numpy array of shape (ny, nx) and dtype float32.

    Raises
    ------
    ValueError
        If a `checksum` is provided and the calculated checksum does not match.
    """
    # --- Vectorized Unpacking ---

    # Calculate the scaling exponent
    scexp = 1.0 / (2.0 ** (7 - exponent))

    # Convert byte array to a 2D numpy grid and calculate the differential values
    grid = np.frombuffer(data, dtype=np.uint8).reshape((ny, nx)).astype(np.float32)
    diffs = (grid - 127) * scexp

    # The first column is a cumulative sum of its own diffs, starting with initial_value.
    # We create an array with initial_value followed by the first column's diffs.
    first_col_vals = np.concatenate(([initial_value], diffs[:, 0]))
    # The cumulative sum gives the unpacked values for the entire first column.
    unpacked_col0 = np.cumsum(first_col_vals)[1:]

    # Replace the first column of diffs with these now-unpacked starting values.
    diffs[:, 0] = unpacked_col0

    # The rest of the grid can now be unpacked by a cumulative sum across the rows (axis=1).
    # Each row starts with its correct, fully unpacked value in the first column.
    unpacked_grid = np.cumsum(diffs, axis=1)

    # Apply the precision check to the final grid.
    unpacked_grid[np.abs(unpacked_grid) < precision] = 0.0

    # --- Optional: Calculate checksum and verify if checksum is provided ---
    if checksum is not None:
        # Calculate the rotating checksum over the entire input array
        calculated_checksum = 0
        for k in range(nx * ny):
            calculated_checksum += data[k]
            # This logic mimics the FORTRAN: "sum carries over the eighth bit add one"
            # It's a form of 1's complement checksum addition.
            if calculated_checksum >= 256:
                calculated_checksum -= 255

        if calculated_checksum != checksum:
            raise ValueError(
                f"Checksum mismatch: calculated {calculated_checksum}, expected {checksum}"
            )

    return unpacked_grid


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

        fields = {
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

        parsed = {}
        for name, (start, end, type_converter) in fields.items():
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

    def to_dict(self) -> dict[str, Any]:
        """
        Convert header to dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary representation of the header fields.
        """
        return {
            "year": self.year,
            "month": self.month,
            "day": self.day,
            "hour": self.hour,
            "forecast": self.forecast,
            "level": self.level,
            "grid": self.grid,
            "variable": self.variable,
            "exponent": self.exponent,
            "precision": self.precision,
            "initial_value": self.initial_value,
        }

    def to_bytes(self) -> bytes:
        """
        Convert header to bytes (not yet implemented).

        Returns
        -------
        bytes
            Binary representation of the header.

        Raises
        ------
        NotImplementedError
            This functionality is not yet implemented.
        """
        raise NotImplementedError


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
    index : int
        Level index.
    height : float
        Height in units of the vertical coordinate system.
    variables : dict[str, VarInfo]
        Dictionary mapping variable names to VarInfo (checksum and reserved).
    """

    index: int
    height: float
    variables: dict[str, VarInfo]


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
    forecast_hour : int
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
    forecast_hour: int
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
        fields["forecast_hour"] = int(fixed[4:7].strip())
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

            vars = {}

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

            lvls.append(LvlInfo(index=i, height=height, variables=vars))

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


@dataclass
class DataRecord:
    """
    Represents a data record containing packed meteorological data.

    Parameters
    ----------
    header : Header
        The header for this data record.
    data : bytes
        The packed binary data.
    """

    header: Header
    data: bytes = field(repr=False)

    @property
    def variable(self) -> str:
        """
        Get the variable name for this data record.

        Returns
        -------
        str
            Variable name from the header.
        """
        return self.header.variable

    @property
    def level(self) -> int:
        """
        Get the vertical level for this data record.

        Returns
        -------
        int
            Level index from the header.
        """
        return self.header.level

    @property
    def forecast(self) -> int:
        """
        Get the forecast hour for this data record.

        Returns
        -------
        int
            Forecast hour from the header.
        """
        return self.header.forecast

    def unpack(self, nx, ny) -> np.ndarray:
        """
        Unpack the data record into a 2D numpy array.

        Parameters
        ----------
        nx : int
            Number of grid points in the x-direction.
        ny : int
            Number of grid points in the y-direction.

        Returns
        -------
        np.ndarray
            Unpacked 2D numpy array of shape (ny, nx).
        """
        return unpack(
            data=self.data,
            nx=nx,
            ny=ny,
            precision=self.header.precision,
            exponent=self.header.exponent,
            initial_value=self.header.initial_value,
        )
