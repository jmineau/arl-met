from dataclasses import dataclass, field
from pathlib import Path
import string
from typing import Dict, Any, List, Tuple, ClassVar

import pandas as pd
import pyproj
import numpy as np
import xarray as xr


# TODO
# - vertical axis
# - CF compliance
# - VariableCatalog


# ARL meteorological variable definitions
ARL_SURFACE_VARIABLES = {
    'U10M': ('U-component of wind at 10 m', 'm/s'),
    'V10M': ('V-component of wind at 10 m', 'm/s'),
    'T02M': ('Temperature at 2 m', 'K'),
    'PBLH': ('Boundary Layer Height', 'm'),
    'PRSS': ('Pressure at surface', 'hPa'),
    'MSLP': ('Pressure at mean sea level', 'hPa'),
    'TMPS': ('Temperature at surface', 'K'),
    'USTR': ('Friction Velocity', 'm/s'),
    'TSTR': ('Friction Temperature', 'K'),
    'RGHS': ('Surface Roughness', 'm'),
    'UMOF': ('U-Momentum flux', 'N/m2'),
    'VMOF': ('V-Momentum flux', 'N/m2'),
    'SHTF': ('Sfc sensible heat flux', 'W/m2'),
    'LTHF': ('Latent heat flux', 'W/m2'),
    'DSWF': ('Downward short wave flux', 'W/m2'),
    'RH2M': ('Relative humidity at 2 m', '%'),
    'SPH2': ('Specific humidity at 2 m', 'kg/kg'),
    'CAPE': ('Convective Available Potential Energy', 'J/kg'),
    'TCLD': ('Total cloud cover', '%'),
    'TPPA': ('Total precipitation for whole dataset', 'm'),
    'TPPD': ('Total precipitation (24-h)', 'm'),
    'TPPT': ('Total precipitation (12-h)', 'm'),
    'TPP6': ('Total precipitation (6-h)', 'm'),
    'TPP3': ('Total precipitation (3-h)', 'm'),
    'TPP1': ('Total precipitation (1-h)', 'm'),
    'PRT6': ('Precipitation Rate (6-h)', 'm/minute'),
    'PRT3': ('Precipitation Rate (3-h)', 'm/minute'),
}

ARL_UPPER_VARIABLES = {
    'UWND': ('U wind component (respect to grid)', 'm/s'),
    'VWND': ('V wind component (respect to grid)', 'm/s'),
    'HGTS': ('Geopotential height', 'gpm'),
    'TEMP': ('Temperature', 'K'),
    'WWND': ('Pressure vertical velocity', 'hPa/s'),
    'RELH': ('Relative Humidity', '%'),
    'SPHU': ('Specific Humidity', 'kg/kg'),
    'DZDT': ('vertical velocity', 'm/s'),
    'TKEN': ('turbulent kinetic energy', 'm2/s2'),
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
    yr = int(yr)
    if yr >= 1900:
        return yr
    # This was in hysplit python code
    return 2000 + yr if (yr < 40) else 1900 + yr


def wrap_lons(lons: np.ndarray) -> np.ndarray:
    """
    Wrap longitude values to -180 to 180 degree range.
    
    Parameters
    ----------
    lons : np.ndarray
        Longitude values in degrees
        
    Returns
    -------
    np.ndarray
        Longitude values wrapped to [-180, 180] range
    """
    return ((lons + 180) % 360) - 180


def unpack(data: bytes | bytearray, nx: int, ny: int,
           precision: float, exponent: int, initial_value: float,
           checksum: int | None = None) -> np.ndarray:
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
    scexp = 1.0 / (2.0**(7 - exponent))

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
            raise ValueError(f"Checksum mismatch: calculated {calculated_checksum}, expected {checksum}")
            
    return unpacked_grid


def open_dataset(filename: Path | str, **kwargs) -> xr.Dataset:
    """
    Open an ARL meteorological data file as an xarray Dataset.

    Parameters
    ----------
    filename : Path or str
        Path to the ARL data file.
    **kwargs
        sel-like keyword arguments passed to `ARLMet.load()`.

    Returns
    -------
    xr.Dataset
        The ARL data as an xarray Dataset.
    """
    met = ARLMet(path=filename)
    ds = met.load(**kwargs)
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset()
    return ds


@dataclass
class Projection:
    """
    ARL Grid Projection

    Parameters
    ----------
    pole_lat : float
        Pole latitude position of the grid projection. Most projections will be defined 
        at +90 or -90 depending upon the hemisphere. For lat-lon grids: latitude of the 
        grid point with the maximum grid point value.
    pole_lon : float  
        Pole longitude position of the grid projection. The longitude 180 degrees from 
        which the projection is cut. For lat-lon grids: longitude of the grid point with 
        the maximum grid point value.
    tangent_lat : float
        Reference latitude at which the grid spacing is defined. For lat-lon grids: 
        grid spacing in degrees latitude.
    tangent_lon : float
        Reference longitude at which the grid spacing is defined. For lat-lon grids:
        grid spacing in degrees longitude.
    grid_size : float
        Grid spacing in km at the reference position. For lat-lon grids: value of zero 
        signals that the grid is a lat-lon grid.
    orientation : float
        Grid orientation or the angle at the reference point made by the y-axis and the 
        local direction of north. For lat-lon grids: value always = 0.
    cone_angle : float
        Angle between the axis and the surface of the cone. For regular projections it 
        equals the latitude at which the grid is tangent to the earth's surface. Polar 
        stereographic: ±90, Mercator: 0, Lambert Conformal: between limits, Oblique 
        stereographic: 90. For lat-lon grids: value always = 0.
    sync_x : float
        Grid x-coordinate used to equate a position on the grid with a position on earth
        (paired with sync_y, sync_lat, sync_lon).
    sync_y : float  
        Grid y-coordinate used to equate a position on the grid with a position on earth
        (paired with sync_x, sync_lat, sync_lon).
    sync_lat : float
        Earth latitude corresponding to the grid position (sync_x, sync_y). For lat-lon 
        grids: latitude of the (0,0) grid point position.
    sync_lon : float
        Earth longitude corresponding to the grid position (sync_x, sync_y). For lat-lon 
        grids: longitude of the (0,0) grid point position.
    reserved : float
        Reserved for future use.

    Attributes
    ----------
    crs : pyproj.CRS
        The pyproj CRS object representing the base grid projection.
        The projection is defined without false easting/northing offsets.
    is_latlon : bool
        True if the grid is a lat-lon grid (grid_size == 0).
    """

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

    params: Dict[str, Any] = field(init=False, repr=False)

    PARAMS: ClassVar[Dict[str, Any]] = {
        'ellps': 'WGS84',
        'R': 6371.2 * 1e3,  # Use a fixed radius to match HYSPLIT
        'units': 'm'
    }

    def __post_init__(self):
        if self.orientation != 0.0:
            raise NotImplementedError("Rotated grids with non-zero orientation are not supported.")

        self.params = self._get_params()

    @property
    def is_latlon(self) -> bool:
        return self.grid_size == 0.0

    def _get_params(self) -> Dict[str, Any]:
        params = self.PARAMS.copy()

        if self.is_latlon:  # Lat/Lon grid
            params.pop('units')
            params.update({
                'proj': 'latlong',
            })
        elif abs(self.cone_angle) == 90.0:  # Stereographic
            if abs(self.pole_lat) == 90.0:  # Polar Stereographic
                params.update({
                    'proj': 'stere',
                    'lat_0': self.pole_lat,
                    'lon_0': self.tangent_lon,
                    'lat_ts': self.tangent_lat
                })
            else:  # Oblique Stereographic
                params.update({
                    'proj': 'sterea',
                    'lat_0': self.pole_lat,
                    'lon_0': self.tangent_lon,
                    'lat_ts': self.tangent_lat
                })
        elif self.cone_angle == 0.0:  # Mercator
            params.update({
                'proj': 'merc',
                'lat_ts': self.tangent_lat,
                'lon_0': self.tangent_lon
            })
        else:  # Lambert Conformal Conic
            params.update({
                'proj': 'lcc',
                'lat_0': self.tangent_lat,
                'lon_0': self.tangent_lon,
                'lat_1': self.cone_angle
            })

        return params


@dataclass
class Grid:
    """
    Represents the horizontal grid of the ARL data.

    Parameters
    ----------
    proj : Projection
        The grid projection information.
    nx : int
        Number of grid points in the x-direction (columns).
    ny : int
        Number of grid points in the y-direction (rows).

    Attributes
    ----------
    origin : Tuple[float, float]
        Origin (lower-left corner) in the base CRS (projected coordinates).
    crs : pyproj.CRS
        Coordinate reference system for the grid.
    coords : Dict[str, Any]
        Coordinates of the grid points in the base CRS (projected coordinates).
    """

    proj: Projection
    nx: int
    ny: int

    origin: Tuple[float, float] = field(init=False)
    crs: pyproj.CRS = field(init=False)
    coords: Dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self):
        self.origin = self._calculate_origin()
        self.crs = self._calculate_crs()
        self.coords = self._calculate_coords()

    @property
    def is_latlon(self) -> bool:
        return self.proj.is_latlon

    @property
    def dims(self) -> tuple:
        if self.is_latlon:
            return ('lat', 'lon')
        return ('y', 'x')

    def _calculate_origin(self) -> Tuple[float, float]:
        'Calculate the origin (lower-left corner) in the base CRS'
        proj = self.proj

        if self.is_latlon:
            # For lat-lon grids, the origin is simply the sync point
            return proj.sync_lon, proj.sync_lat

        # Calculate what the projected coordinates of the sync point should be
        base_crs = pyproj.CRS.from_dict(proj.params)
        transformer = pyproj.Transformer.from_proj('EPSG:4326', base_crs,
                                                   always_xy=True)
        sync_proj_x, sync_proj_y = transformer.transform(proj.sync_lon, proj.sync_lat)

        # Convert sync grid coordinates to projected coordinates
        # Grid coordinates are 1-based, so sync_x=1, sync_y=1 means bottom-left corner
        sync_grid_x_m = (proj.sync_x - 1) * proj.grid_size * 1000  # convert km to m
        sync_grid_y_m = (proj.sync_y - 1) * proj.grid_size * 1000  # convert km to m

        # Calculate the origin offset to align grid coordinates with projected coordinates
        origin_x = sync_grid_x_m - sync_proj_x
        origin_y = sync_grid_y_m - sync_proj_y
        return origin_x, origin_y

    def _calculate_crs(self) -> pyproj.CRS:
        params = self.proj.params.copy()

        if self.is_latlon:
            # Use specific ellps and 
            return pyproj.CRS.from_dict(params)

        # Create new pyproj CRS with false easting/northing
        params.update({
            'x_0': self.origin[0],
            'y_0': self.origin[1]
        })
        return pyproj.CRS.from_dict(params)

    def _calculate_coords(self):
        proj = self.proj

        if self.is_latlon:
            lon_0, lat_0 = self.origin
            dlat = proj.tangent_lat
            dlon = proj.tangent_lon
            lats = lat_0 + np.arange(self.ny) * dlat
            lons = lon_0 + np.arange(self.nx) * dlon
            lons = wrap_lons(lons)
            return {'lon': lons, 'lat': lats}

        # Calculate the coordinates in the projection space
        grid_size = proj.grid_size * 1000  # km to m
        x_coords = np.arange(self.nx) * grid_size
        y_coords = np.arange(self.ny) * grid_size

        # Create a transformer from the projection to lat/lon
        transformer = pyproj.Transformer.from_crs(self.crs, "EPSG:4326",
                                                  always_xy=True)

        # Transform the coordinates to lat/lon
        xx, yy = np.meshgrid(x_coords, y_coords)
        lons, lats = transformer.transform(xx, yy)
        lons = wrap_lons(lons)

        return {'x': x_coords, 'y': y_coords,
                'lon': (('y', 'x'), lons),
                'lat': (('y', 'x'), lats)}


class VerticalAxis:
    """
    Represents the vertical axis of the ARL data.

    Parameters
    ----------
    vertical_flag : int
        Vertical coordinate system type (1=sigma, 2=pressure, 3=terrain, 4=hybrid).
    levels : List[float]
        List of vertical levels corresponding to the vertical coordinate system.
    """

    FLAGS = {
        1: 'sigma',  # (fraction)
        2: 'pressure',  # (mb)
        3: 'terrain',  # (fraction)
        4: 'hybrid'  # (mb: offset.fraction)
    }

    def __init__(self, vertical_flag: int, levels: List[float]):
        self.flag = vertical_flag
        self.levels = levels

    @property
    def coord_type(self) -> str:
        return VerticalAxis.FLAGS.get(self.flag, 'unknown')

    def calculate_heights(self) -> np.ndarray | None:
        """
        Calculate heights in meters for each vertical level.
        """
        if self.coord_type == 'sigma':
            # fraction
            pass
        elif self.coord_type == 'pressure':
            # mb
            pass
        elif self.coord_type == 'terrain':
            # fraction
            pass
        elif self.coord_type == 'hybrid':
            # mb offset.fraction
            pass
        else:
            raise ValueError(f"Unknown vertical coordinate type")

        return None


class Grid3D(Grid):
    """
    Represents a 3D grid with horizontal and vertical dimensions.

    Parameters
    ----------
    proj: Projection
        The grid projection information.
    nx : int
        Number of grid points in the x-direction (columns).
    ny : int
        Number of grid points in the y-direction (rows).
    vertical_axis : VerticalAxis
        Vertical axis information including coordinate type and levels.
    """

    def __init__(self, proj: Projection, nx: int, ny: int,
                 vertical_axis: VerticalAxis):
        super().__init__(proj=proj, nx=nx, ny=ny)
        self.vertical_axis = vertical_axis

        # self.coords['level'] = self.vertical_axis.levels

    @property
    def dims(self) -> tuple:
        xy_dims = super().dims
        return xy_dims  # TODO: add vertical dimension


@dataclass
class Header:
    'First 50 bytes of each record'

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
    def from_bytes(cls, data: bytes) -> 'Header':
        """
        Parse header from raw bytes.
        """
        if len(data) != cls.N_BYTES:
            raise ValueError(f"{cls.__name__} must be exactly {cls.N_BYTES} bytes, got {len(data)}")

        header = data.decode('ascii', errors='ignore')

        fields = {
            'year': (0, 2, restore_year),
            'month': (2, 4, int),
            'day': (4, 6, int),
            'hour': (6, 8, int),
            'forecast': (8, 10, int),
            'level': (10, 12, int),
            'grid': (12, 14, str),
            'variable': (14, 18, str),
            'exponent': (18, 22, int),
            'precision': (22, 36, float),
            'initial_value': (36, 50, float),
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
        parsed['grid'] = (letter_to_thousands(parsed['grid'][0]),
                          letter_to_thousands(parsed['grid'][1]))

        return cls(**parsed)

    @property
    def time(self) -> pd.Timestamp:
        return pd.Timestamp(year=self.year, month=self.month,
                            day=self.day, hour=self.hour)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'year': self.year,
            'month': self.month,
            'day': self.day,
            'hour': self.hour,
            'forecast': self.forecast,
            'level': self.level,
            'grid_x': self.grid[0],
            'grid_y': self.grid[1],
            'variable': self.variable,
            'exponent': self.exponent,
            'precision': self.precision,
            'initial_value': self.initial_value
        }

    def to_bytes(self) -> bytes:
        raise NotImplementedError


@dataclass
class LevelInfo:
    """Information about a single vertical level."""
    index: int
    height: float  # in units of the vertical coordinate system
    variables: Dict[str, Tuple[int, Any]]  # variable name -> (checksum, reserved) mapping

    @property
    def variable_names(self) -> List[str]:
        """Get list of variable names at this level."""
        return list(self.variables.keys())


class VariableCatalog:
    """
    Catalog:
      levels: `nz`
      surface variables: `var1`, `var2`, `var3`
      upper air variables: `var1`, `var2`, `var3`, ... `varn`
      heights: `h1`, `h2`, `h3`, ... `hn`
    """
    def __init__(self, levels: List[Dict[str, Any]]):
        self.levels = levels


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
    levels : List[Dict[str, Any]]
        List of levels, each containing:
            - level: Level number (1-based).
            - height: Height of the level in units of the vertical coordinate.
            - vars: List of variables at this level, each as a tuple:
                (name: str, checksum: int, reserved: str)

    Attributes
    ----------
    N_BYTES_FIXED : int
        Number of bytes in the fixed portion of the index record (108 bytes).
    time : pd.Timestamp
        The valid time of the record, calculated from the header time and minutes.
    grid : Grid3D
        The 3D grid representation including horizontal and vertical dimensions.
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
    levels: List[Dict[str, Any]]

    grid: Grid3D = field(init=False, repr=False)

    N_BYTES_FIXED: ClassVar[int] = 108

    def __post_init__(self):
        self.grid = self._build_grid()

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
            raise ValueError(f"IndexRecord fixed portion must be at least {IndexRecord.N_BYTES_FIXED} bytes, got {len(data)}")

        fields = {}

        # Parse the fixed portion of the index record
        # Format: (A4)(I3)(I2)(12F7)(3I3)(I2)(I4)
        fixed = data[:IndexRecord.N_BYTES_FIXED].decode('ascii', errors='ignore')

        fields['source'] = fixed[:4].strip()
        fields['forecast_hour'] = int(fixed[4:7].strip())
        fields['minutes'] = int(fixed[7:9].strip())

        # Parse 12 floating point values (each 7 characters)
        proj_section = fixed[9:9 + 12 * 7]  # 12 * 7 = 84 characters
        proj_names = [
            'pole_lat', 'pole_lon', 'tangent_lat', 'tangent_lon',
            'grid_size', 'orientation', 'cone_angle', 'sync_x',
            'sync_y', 'sync_lat', 'sync_lon', 'reserved'
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
        fields['nx'] = int(grid_section[0:3].strip())
        fields['ny'] = int(grid_section[3:6].strip())
        fields['nz'] = int(grid_section[6:9].strip())

        # Parse vertical level information
        fields['vertical_flag'] = int(fixed[102:104].strip())
        fields['index_length'] = int(fixed[104:108].strip())

        return fields

    @staticmethod
    def parse_extended(data: bytes, nz: int) -> VariableCatalog:
        """
        Parse the variable-length levels portion from raw bytes.

        Parameters
        ----------
            data: Raw bytes containing variable information
            nz: Number of vertical levels

        Returns:
            VariableCatalog instance
        """
        variables = data.decode('ascii', errors='ignore')

        # Loop through levels to extract variable info
        lvls = []
        cursor = 0
        for i in range(1, (nz + 1)):  # 1-based indexing
            height = float(variables[cursor:cursor+6].strip())  # in units of vertical coordinate
            num_vars = int(variables[cursor+6:cursor+8].strip())
            vars = []
            # Loop through variables for this level
            for j in range(num_vars):
                start = cursor + 8 + j*8
                end = start + 8
                name = variables[start:start+4].strip()
                checksum = int(variables[start+4:start+7].strip())
                reserved = variables[start+7:end].strip()  # usually blank
                vars.append((name, checksum, reserved))

            lvls.append({
                'level': i,
                'height': height,
                'vars': vars
            })

            # Move cursor to next level
            cursor += 8 + num_vars * 8

        return VariableCatalog(levels=lvls)

    @property
    def time(self) -> pd.Timestamp:
        return self.header.time + pd.Timedelta(minutes=self.minutes)

    def _build_grid(self) -> Grid3D:
        proj = Projection(pole_lat=self.pole_lat, pole_lon=self.pole_lon,
                          tangent_lat=self.tangent_lat, tangent_lon=self.tangent_lon,
                          grid_size=self.grid_size, orientation=self.orientation,
                          cone_angle=self.cone_angle, sync_x=self.sync_x,
                          sync_y=self.sync_y, sync_lat=self.sync_lat,
                          sync_lon=self.sync_lon, reserved=self.reserved)

        nx = self.nx + self.header.grid[0]
        ny = self.ny + self.header.grid[1]

        vertical_axis = VerticalAxis(vertical_flag=self.vertical_flag,
                                     levels=[lvl['height'] for lvl in self.levels])

        return Grid3D(proj=proj, nx=nx, ny=ny,
                      vertical_axis=vertical_axis)


class DataRecord:
    """
    Represents a data record containing packed meteorological data.
    """

    def __init__(self, index_record: IndexRecord, header: Header, data: bytes):
        self.index_record = index_record
        self.header = header
        self.data = data

    def __repr__(self):
        return (f"DataRecord(index_record={repr(self.index_record)}, "
                f"header={repr(self.header)}, data_length={len(self.data)})")

    @property
    def time(self) -> pd.Timestamp:
        return self.index_record.time

    def unpack(self) -> xr.DataArray:
        """
        Unpack the data record into a 2D xarray DataArray.
        """
        grid = self.index_record.grid
        nx, ny = grid.nx, grid.ny
        dims = grid.dims
        coords = grid.coords

        unpacked = unpack(data=self.data,
                          nx=nx, ny=ny,
                          precision=self.header.precision,
                          exponent=self.header.exponent,
                          initial_value=self.header.initial_value)


        da = xr.DataArray(data=unpacked, dims=dims[:2], coords=coords,
                          name=self.header.variable)

        # Sort on dims
        da = da.sortby(list(dims[:2]))

        # Expand dimensions for time, forecast, level, grid
        da = da.expand_dims({
            'time': [self.time],
            'forecast': [self.header.forecast],   # TODO add note about -1=NAN
            'level': [self.header.level],
            # 'grid': None  # TODO how to identify grid?
        })

        # Calculate height
        # da = da.assign_coords({
        #     'height': ('level', )
        # })

        # TODO: add CF attributes

        return da


class ARLMet:
    """
    ARL (Air Resources Laboratory) packed meteorological data.

    Extracts metadata from meteorological data file headers and provides
    methods to work with ARL format files.
    """

    def __init__(self, path: Path | str):  # TODO: move to classmethod
        self.path = Path(path)

        if not self.path.exists():
            raise ValueError("Invalid file path")

        # Open the file
        with self.path.open('rb') as f:
            data = f.read()

        records = []

        index_record = None
        nxy = 0

        # Build the index
        cursor = 0
        while cursor < len(data):
            # Read the next header
            header = Header.from_bytes(data[cursor:cursor + Header.N_BYTES])
            cursor += Header.N_BYTES

            if header.variable == 'INDX':
                # Parse the index record to get grid dimensions
                fixed_end = cursor + IndexRecord.N_BYTES_FIXED
                fixed = IndexRecord.parse_fixed(data=data[cursor:fixed_end])
                index_len = fixed['index_length']
                index_end = cursor + index_len
                catalog = IndexRecord.parse_extended(data=data[fixed_end:index_end],
                                                     nz=fixed['nz'])
                index_record = IndexRecord(header=header, **fixed,
                                           levels=catalog.levels)
                cursor += index_len

                # Calculate grid size
                nxy = index_record.grid.nx * index_record.grid.ny
                cursor += (nxy - index_len)  # Skip any extra bytes in index record
            else:
                if index_record is None:
                    raise ValueError("Data record found before index record")
                record = DataRecord(index_record=index_record, header=header,
                                    data=data[cursor:cursor + nxy])

                records.append({
                    'grid': None,  # TODO how to identify grid?
                    'time': record.time,
                    'forecast': header.forecast,
                    'level': header.level,
                    'variable': header.variable,
                    'record': record,
                    })
                cursor += nxy  # Move cursor past packed data

        index = pd.DataFrame(records)
        # index_keys = ['grid', 'time', 'forecast', 'level', 'variable']  # TODO grid
        index_keys = ['time', 'forecast', 'level', 'variable']
        self._index = index.set_index(index_keys)['record']

    @property
    def records(self) -> List[DataRecord]:
        return self._index.tolist()

    def load(self, **kwargs) -> xr.Dataset | xr.DataArray:
        """
        Load data into an xarray Dataset.

        Accepts xarray sel-style indexing to select specific times, variables, levels, forecasts, or grids.

        Parameters
        ----------
        **kwargs : dict
        """
        # Select records matching criteria
        index = self._index.to_xarray()
        records = index.sel(**kwargs).values.flatten().tolist()

        # Unpack each record to DataArray
        arrays = [r.unpack() for r in records
                  if isinstance(r, DataRecord)]  # drop nans from xarray

        # variable is a dim of index, not data
        variables = kwargs.pop('variable', None)

        if len(arrays) == 1:
            # Single record, return DataArray
            return arrays[0].sel(**kwargs)

        # Multiple records, return Dataset
        ds = xr.merge(arrays)
        if variables is not None:
            ds = ds[variables]  # select only requested variables
        return ds.sel(**kwargs)

    def __add__(self, other: 'ARLMet') -> 'ARLMet':
        raise NotImplementedError("Merging ARLMet instances is not yet implemented.")