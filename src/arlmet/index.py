"""Binary codecs for ARL index records: VarInfo, LvlInfo, and IndexRecord."""

from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, BinaryIO, ClassVar

import pandas as pd

from arlmet._time import ensure_timestamp
from arlmet.grid import Grid, Projection
from arlmet.header import Header, format_fixed_width_float
from arlmet.vertical import VerticalAxis


def _derive_index_forecast(
    record_forecasts: Iterable[int], explicit_forecast: int | None = None
) -> int:
    """
    Derive an index-record forecast from per-record forecasts.

    If an explicit forecast is provided, use it directly. Otherwise, use the
    lowest non-negative forecast from the records, or -1 if all are missing.
    """
    if explicit_forecast is not None:
        return explicit_forecast

    nonnegative_forecasts = sorted(
        int(value) for value in record_forecasts if int(value) >= 0
    )
    return nonnegative_forecasts[0] if nonnegative_forecasts else -1


@dataclass
class VarInfo:
    """Checksum and reserved-byte metadata for one variable within an index record."""

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
    Index record describing one ARL time step and its contained variables.

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
    levels : sequence of LvlInfo
        Variable manifests for each stored vertical level.

    Attributes
    ----------
    N_BYTES_FIXED : int
        Number of bytes in the fixed portion of the index record (108 bytes).
    time : pd.Timestamp
        The valid time of the record, calculated from the header time and minutes.

    Methods
    -------
    from_position(file, position)
        Read and parse an index record from a file handle.
    tobytes()
        Serialize the exact used bytes of the index record.
    to_record_bytes(record_size)
        Serialize the index record padded to one ARL record.

    Examples
    --------
    >>> from arlmet.header import Header
    >>> from arlmet.index import IndexRecord
    >>> header = Header(
    ...     year=2024,
    ...     month=7,
    ...     day=18,
    ...     hour=0,
    ...     forecast=0,
    ...     level=0,
    ...     grid=(0, 0),
    ...     variable="INDX",
    ...     exponent=0,
    ...     precision=0.0,
    ...     initial_value=0.0,
    ... )
    >>> isinstance(header.time, pd.Timestamp)
    True
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
    def from_position(cls, file: BinaryIO, position: int) -> "IndexRecord":
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
        file.seek(position)

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

        fixed = IndexRecord.parse_fixed(data=file.read(IndexRecord.N_BYTES_FIXED))
        extended = file.read(fixed["index_length"] - IndexRecord.N_BYTES_FIXED)
        levels = IndexRecord.parse_extended(data=extended, nz=fixed["nz"])
        return IndexRecord(header=header, **fixed, levels=levels)

    def serialize_fixed(self, index_length: int | None = None) -> bytes:
        """Serialize the fixed 108-byte portion of the index record."""
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
        """Serialize the variable-length level/variable portion of the index record."""
        chunks: list[str] = []
        for level in self.levels:
            chunks.append(format_fixed_width_float(level.height, 6))
            chunks.append(f"{len(level.variables):2d}")
            for name, info in level.variables.items():
                reserved = (info.reserved or " ")[:1]
                chunks.append(f"{name:<4}{info.checksum:3d}{reserved}")
        return "".join(chunks).encode("ascii")

    def tobytes(self) -> bytes:
        """Serialize the exact used bytes of the index record, including its header."""
        extended = self.serialize_extended()
        index_length = self.N_BYTES_FIXED + len(extended)
        self.index_length = index_length
        fixed = self.serialize_fixed(index_length=index_length)
        return self.header.tobytes() + fixed + extended

    def to_record_bytes(self, record_length: int) -> bytes:
        """Serialize the index record padded to one full ARL record."""
        raw = self.tobytes()
        if len(raw) > record_length:
            raise ValueError(
                f"Index record uses {len(raw)} bytes, which exceeds record length {record_length}."
            )
        return raw.ljust(record_length, b" ")

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
                f"IndexRecord fixed portion must be at least {IndexRecord.N_BYTES_FIXED} bytes, "
                f"got {len(data)}"
            )

        fixed = data[: IndexRecord.N_BYTES_FIXED].decode("ascii", errors="ignore")

        fields: dict[str, Any] = {}
        fields["source"] = fixed[:4].strip()
        fields["forecast"] = int(fixed[4:7].strip())
        fields["minutes"] = int(fixed[7:9].strip())

        proj_section = fixed[9 : 9 + 12 * 7]
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
                val = -(360 - val)
            fields[proj_names[i]] = val

        grid_section = fixed[93:102]
        fields["nx"] = int(grid_section[0:3].strip())
        fields["ny"] = int(grid_section[3:6].strip())
        fields["nz"] = int(grid_section[6:9].strip())

        fields["vertical_flag"] = int(fixed[102:104].strip())
        fields["index_length"] = int(fixed[104:108].strip())

        return fields

    @staticmethod
    def parse_extended(data: bytes, nz: int) -> list[LvlInfo]:
        """
        Parse the variable-length portion of an index record.

        Parameters
        ----------
        data : bytes
            Raw bytes containing the extended portion of the index record.
        nz : int
            Number of vertical levels (from the fixed portion).

        Returns
        -------
        list[LvlInfo]
            List of LvlInfo objects for each vertical level.
        """
        extended = data.decode("ascii", errors="ignore")

        lvls = []
        cursor = 0
        for i in range(nz):
            height = float(extended[cursor : cursor + 6].strip())

            vars: OrderedDict[str, VarInfo] = OrderedDict()
            num_vars = int(extended[cursor + 6 : cursor + 8].strip())
            for j in range(num_vars):
                start = cursor + 8 + j * 8
                end = start + 8
                name = extended[start : start + 4].strip()
                vars[name] = VarInfo(
                    checksum=int(extended[start + 4 : start + 7].strip()),
                    reserved=extended[start + 7 : end].strip(),
                )

            lvls.append(LvlInfo(level=i, height=height, variables=vars))
            cursor += 8 + num_vars * 8

        return lvls

    @property
    def time(self) -> pd.Timestamp:
        """Valid time calculated from header time plus minutes offset."""
        return ensure_timestamp(self.header.time + pd.Timedelta(minutes=self.minutes))

    @property
    def total_nx(self) -> int:
        """Total x grid points including thousands from header grid letters."""
        return self.nx + self.header.grid[0]

    @property
    def total_ny(self) -> int:
        """Total y grid points including thousands from header grid letters."""
        return self.ny + self.header.grid[1]

    @property
    def grid(self) -> Grid:
        """Construct a Grid from the index record's projection parameters."""
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
        return Grid(projection=proj, nx=self.total_nx, ny=self.total_ny)

    @property
    def vertical_axis(self) -> VerticalAxis:
        """Construct the vertical axis definition from the index record."""
        return VerticalAxis.from_flag(
            self.vertical_flag,
            levels=[lvl.height for lvl in self.levels],
            offset=self.reserved,
        )
