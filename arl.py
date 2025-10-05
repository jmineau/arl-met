from pathlib import Path
import os
import string
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import pyproj
import numpy as np
import xarray as xr


# TODO
# - CRS
# - height calculation
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

# Vertical coordinate system types
VERTICAL_COORDS = {
    1: 'sigma',  # (fraction)
    2: 'pressure',  # (mb)
    3: 'terrain',  # (fraction)
    4: 'hybrid'  # (mb: offset.fraction)
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


class CRS:
    """
    ARL Coordinate Reference System (CRS)

    Represents the coordinate reference system and projection parameters for ARL meteorological data.

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
    ref_lat : float
        Reference latitude at which the grid spacing is defined. For lat-lon grids: 
        grid spacing in degrees latitude.
    ref_lon : float
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
    """

    def __init__(self, pole_lat: float, pole_lon: float,
                 ref_lat: float, ref_lon: float,
                 grid_size: float, orientation: float,
                 cone_angle: float,
                 sync_x: float, sync_y: float,
                 sync_lat: float, sync_lon: float):
        self.pole_lat = pole_lat
        self.pole_lon = pole_lon
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        self.grid_size = grid_size
        self.orientation = orientation
        self.cone_angle = cone_angle
        self.sync_x = sync_x
        self.sync_y = sync_y
        self.sync_lat = sync_lat
        self.sync_lon = sync_lon

        self._crs = self._build_crs()

    @property
    def is_latlon(self) -> bool:
        return self.grid_size == 0.0

    def _build_crs(self):
        if self.is_latlon:
            # Lat/Lon grid
            return pyproj.CRS.from_epsg(4326)  # WGS84

        # Determine projection type based on cone_angle and pole_lat
        if abs(self.cone_angle) == 90.0:  # Stereographic
            if abs(self.pole_lat) == 90.0:  # Polar Stereographic
                proj_str = (f"+proj=stere +lat_0={self.pole_lat} +lon_0={self.pole_lon} "
                            f"+lat_ts={self.ref_lat} +x_0=0 +y_0=0 +ellps=WGS84")
            else:  # Oblique Stereographic
                proj_str = (f"+proj=sterea +lat_0={self.pole_lat} +lon_0={self.pole_lon} "
                            f"+lat_ts={self.ref_lat} +x_0=0 +y_0=0 +ellps=WGS84")
        elif self.cone_angle == 0.0:  # Mercator
            proj_str = (f"+proj=merc +lat_ts={self.ref_lat} +lon_0={self.ref_lon} "
                        f"+x_0=0 +y_0=0 +ellps=WGS84")
        else:  # Lambert Conformal Conic
            proj_str = (f"+proj=lcc +lat_0={self.ref_lat} +lon_0={self.ref_lon} "
                        f"+lat_1={self.cone_angle} +x_0=0 +y_0=0 +ellps=WGS84")

        return pyproj.CRS.from_proj4(proj_str)

    def to_pyproj(self) -> pyproj.CRS:
        return self._crs

    def to_wkt(self) -> str:
        return self._crs.to_wkt()

    def calculate_coordinates(self, nx: int, ny: int
                              ):
        if self.is_latlon:
            lat_0 = self.sync_lat
            lon_0 = self.sync_lon
            lat_1 = self.pole_lat
            lon_1 = self.pole_lon
            dlat = self.ref_lat
            dlon = self.ref_lon
            lats = lat_0 + np.arange(ny) * dlat
            lons = lon_0 + np.arange(nx) * dlon
            return {'lon': lons, 'lat': lats}

        # Create a transformer from the projection to lat/lon
        transformer = pyproj.Transformer.from_crs(self._crs, "EPSG:4326")

        # Calculate the coordinates in the original projection
        x_coords = self.sync_x + np.arange(nx) * self.grid_size * 1000  # convert km to m
        y_coords = self.sync_y + np.arange(ny) * self.grid_size * 1000  # convert km to m

        # Transform the coordinates to lat/lon
        yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
        lats, lons = transformer.transform(xx, yy)

        return {'x': x_coords, 'y': y_coords,
                'lon': (('y', 'x'), lons),
                'lat': (('y', 'x'), lats)}


class Header:
    'First 50 bytes of each record'

    N_BYTES = 50

    def __init__(self, year: int, month: int, day: int, hour: int,
                 forecast: int | None, level: int, grid: tuple[int, int],
                 variable: str, exponent: int,
                 precision: float, initial_value: float):
        self.year = year
        self.month = month
        self.day = day
        self.hour = hour
        self.forecast = forecast
        self.level = level
        self.grid = grid
        self.variable = variable
        self.exponent = exponent
        self.precision = precision
        self.initial_value = initial_value

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

        if parsed['forecast'] == -1:
            # Forecast hour is -1 for missing data
            parsed['forecast'] = None

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

    def __repr__(self):
        return (f"Header(year={self.year}, month={self.month}, day={self.day}, "
                f"hour={self.hour}, forecast={self.forecast}, level={self.level}, "
                f"grid={self.grid}, variable='{self.variable}', exponent={self.exponent}, "
                f"precision={self.precision}, initial_value={self.initial_value})")

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


class IndexRecordFixed:
    """
    Represents the fixed 108-byte portion of an ARL index record.
    Contains projection and grid information.
    """

    N_BYTES = 108

    def __init__(self, source: str, forecast_hour: int, minutes: int,
                 pole_lat: float, pole_lon: float, ref_lat: float,
                 ref_lon: float, grid_size: float, orientation: float,
                 cone_angle: float, sync_x: float, sync_y: float,
                 sync_lat: float, sync_lon: float,
                 nx: int, ny: int, nz: int, vertical_coord: int,
                 index_length: int):
        self.source = source
        self.forecast_hour = forecast_hour
        self.minutes = minutes
        self.pole_lat = pole_lat
        self.pole_lon = pole_lon
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        self.grid_size = grid_size
        self.orientation = orientation
        self.cone_angle = cone_angle
        self.sync_x = sync_x
        self.sync_y = sync_y
        self.sync_lat = sync_lat
        self.sync_lon = sync_lon
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.vertical_coord = vertical_coord
        self.index_length = index_length

    @classmethod
    def from_bytes(cls, data: bytes) -> 'IndexRecordFixed':
        """
        Parse the fixed 108-byte portion of an index record from raw bytes.
        
        Args:
            data: Raw bytes starting from position 50 in ARL file
            
        Returns:
            IndexRecordFixed instance
        """
        if len(data) < cls.N_BYTES:
            raise ValueError(f"IndexRecordFixed requires at least {cls.N_BYTES} bytes, got {len(data)}")

        # Parse the fixed portion of the index record
        # Format: (A4)(I3)(I2)(12F7)(3I3)(I2)(I4)
        record = data[:cls.N_BYTES].decode('ascii', errors='ignore')

        source = record[:4].strip()
        forecast_hour = int(record[4:7].strip())
        minutes = int(record[7:9].strip())

        # Parse 12 floating point values (each 7 characters)
        proj_section = record[9:9 + 12 * 7]  # 12 * 7 = 84 characters
        proj_names = [
            'pole_lat', 'pole_lon', 'ref_lat', 'ref_lon',
            'ref_grid', 'orientation', 'cone_angle', 'sync_x',
            'sync_y', 'sync_lat', 'sync_lon', 'reserved'
        ]
        proj_values = {}
        for i in range(12):
            start = i * 7
            end = start + 7
            val = float(proj_section[start:end].strip())
            if val > 180:
                # Adjust longitudes greater than 180 degrees
                val = -(360 - val)
            proj_values[proj_names[i]] = val
        proj_values.pop('reserved')  # Remove reserved field

        # Parse grid dimensions (3 integers, 3 characters each)
        grid_section = record[93:102]
        nx = int(grid_section[0:3].strip())
        ny = int(grid_section[3:6].strip())
        nz = int(grid_section[6:9].strip())

        # Parse vertical level information
        vertical_coord = int(record[102:104].strip())
        index_length = int(record[104:108].strip())

        return cls(
            source, forecast_hour, minutes,
            proj_values['pole_lat'], proj_values['pole_lon'],
            proj_values['ref_lat'], proj_values['ref_lon'],
            proj_values['ref_grid'], proj_values['orientation'],
            proj_values['cone_angle'], proj_values['sync_x'],
            proj_values['sync_y'], proj_values['sync_lat'],
            proj_values['sync_lon'], nx, ny, nz,
            vertical_coord, index_length
        )
    
    def __repr__(self):
        return (f"IndexRecordFixed(source='{self.source}', forecast_hour={self.forecast_hour}, "
                f"minutes={self.minutes}, pole_lat={self.pole_lat}, pole_lon={self.pole_lon}, "
                f"ref_lat={self.ref_lat}, ref_lon={self.ref_lon}, "
                f"grid_size={self.grid_size}, orientation={self.orientation}, "
                f"cone_angle={self.cone_angle}, sync_x={self.sync_x}, sync_y={self.sync_y}, "
                f"sync_lat={self.sync_lat}, sync_lon={self.sync_lon}, nx={self.nx}, "
                f"ny={self.ny}, nz={self.nz}, vertical_coord={self.vertical_coord}, "
                f"index_length={self.index_length})")


class VariableCatalog:
    pass


class IndexRecordExtended:
    """
    Represents the variable-length portion of an ARL index record.
    Contains level information with variables and checksums.
    """
    
    def __init__(self, levels: List[Dict[str, Any]]):
        self.levels = levels

    @classmethod
    def from_bytes(cls, data: bytes, nz: int) -> 'IndexRecordExtended':
        """
        Parse the variable-length levels portion from raw bytes.
        
        Parameters
        ----------
            data: Raw bytes containing variable information
            nz: Number of vertical levels
            
        Returns:
            IndexRecordVariable instance
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
                checksum = int(variables[start+4:end].strip())
                vars.append((name, checksum))

            lvls.append({
                'level': i,
                'height': height,
                'vars': vars
            })

            # Move cursor to next level
            cursor += 8 + num_vars * 8

        return cls(lvls)

    def __repr__(self):
        return f"IndexRecordExtended(levels={repr(self.levels)})"


class IndexRecord:
    """
    Represents a complete ARL index record that precedes data records for each time period.
    Combines the fixed portion with the variable levels portion.
    """

    def __init__(self,
                 source: str, forecast_hour: int, minutes: int,
                 pole_lat: float, pole_lon: float, ref_lat: float,
                 ref_lon: float, grid_size: float, orientation: float,
                 cone_angle: float, sync_x: float, sync_y: float,
                 sync_lat: float, sync_lon: float,
                 nx: int, ny: int, nz: int, vertical_coord: int,
                 index_length: int, levels: List[Dict[str, Any]]):
        self.source = source
        self.forecast_hour = forecast_hour
        self.minutes = minutes
        self.pole_lat = pole_lat
        self.pole_lon = pole_lon
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        self.grid_size = grid_size
        self.orientation = orientation
        self.cone_angle = cone_angle
        self.sync_x = sync_x
        self.sync_y = sync_y
        self.sync_lat = sync_lat
        self.sync_lon = sync_lon
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.vertical_coord = vertical_coord
        self.index_length = index_length
        self.levels = levels

    @classmethod
    def from_parts(cls, fixed: IndexRecordFixed, extended: IndexRecordExtended) -> 'IndexRecord':
        return cls(
            source=fixed.source,
            forecast_hour=fixed.forecast_hour,
            minutes=fixed.minutes,
            pole_lat=fixed.pole_lat,
            pole_lon=fixed.pole_lon,
            ref_lat=fixed.ref_lat,
            ref_lon=fixed.ref_lon,
            grid_size=fixed.grid_size,
            orientation=fixed.orientation,
            cone_angle=fixed.cone_angle,
            sync_x=fixed.sync_x,
            sync_y=fixed.sync_y,
            sync_lat=fixed.sync_lat,
            sync_lon=fixed.sync_lon,
            nx=fixed.nx,
            ny=fixed.ny,
            nz=fixed.nz,
            vertical_coord=fixed.vertical_coord,
            index_length=fixed.index_length,
            levels=extended.levels
        )

    def __repr__(self):
        return (f"IndexRecord(source='{self.source}', forecast_hour={self.forecast_hour}, "
                f"minutes={self.minutes}, pole_lat={self.pole_lat}, pole_lon={self.pole_lon}, "
                f"ref_lat={self.ref_lat}, ref_lon={self.ref_lon}, "
                f"grid_size={self.grid_size}, orientation={self.orientation}, "
                f"cone_angle={self.cone_angle}, sync_x={self.sync_x}, sync_y={self.sync_y}, "
                f"sync_lat={self.sync_lat}, sync_lon={self.sync_lon}, nx={self.nx}, "
                f"ny={self.ny}, nz={self.nz}, vertical_coord={self.vertical_coord}, "
                f"index_length={self.index_length}, levels={repr(self.levels)})")

    @property
    def crs(self) -> CRS:
        return CRS(
            pole_lat=self.pole_lat,
            pole_lon=self.pole_lon,
            ref_lat=self.ref_lat,
            ref_lon=self.ref_lon,
            grid_size=self.grid_size,
            orientation=self.orientation,
            cone_angle=self.cone_angle,
            sync_x=self.sync_x,
            sync_y=self.sync_y,
            sync_lat=self.sync_lat,
            sync_lon=self.sync_lon
        )


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
        return self.header.time + pd.Timedelta(minutes=self.index_record.minutes)

    def unpack(self) -> xr.DataArray:
        """
        Unpack the data record into a 2D xarray DataArray.
        """
        nx = self.index_record.nx + self.header.grid[0]
        ny = self.index_record.ny + self.header.grid[1]

        unpacked = unpack(data=self.data, nx=nx, ny=ny,
                          precision=self.header.precision,
                          exponent=self.header.exponent,
                          initial_value=self.header.initial_value)

        crs = self.index_record.crs
        if crs.is_latlon:
            dims = ('lat', 'lon')
        else:
            dims = ('y', 'x')
        coords = crs.calculate_coordinates(nx=nx, ny=ny)

        da = xr.DataArray(data=unpacked, dims=dims, coords=coords,
                          name=self.header.variable)

        # Expand dimensions for time, forecast, level, grid
        da = da.expand_dims({
            'time': [self.time],
            'forecast': [self.header.forecast],
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

    def __init__(self, path: str):
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
                end_fixed = cursor + IndexRecordFixed.N_BYTES
                fixed = IndexRecordFixed.from_bytes(data[cursor:end_fixed])
                end_index = cursor + fixed.index_length
                extended = IndexRecordExtended.from_bytes(data[end_fixed:end_index],
                                                          nz=fixed.nz)
                index_record = IndexRecord.from_parts(fixed=fixed, extended=extended)
                cursor += fixed.index_length

                # Calculate grid size
                nx = index_record.nx + header.grid[0]
                ny = index_record.ny + header.grid[1]
                nxy = nx * ny
                cursor += (nxy - fixed.index_length)  # Skip any extra bytes in index record
            else:
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
        index = self._index.to_xarray()
        records = index.sel(**kwargs).values.flatten().tolist()
        arrays = [r.unpack() for r in records
                  if isinstance(r, DataRecord)]

        if len(arrays) == 1:
            return arrays[0]

        ds = xr.merge(arrays)
        return ds.sel(**kwargs)
