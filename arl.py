from pathlib import Path
import os
import string
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


# TODO
# - Build index of headers to be able to seek to specific variable/level/time/forecast
# - Add methods to read specific variables/levels/times


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


class Header:
    'First 50 bytes of each record in ARL file'
    def __init__(self, year: int, month: int, day: int, hour: int,
                 forecast: int | None, level: int, grid: tuple[int, int],
                 variable: str, exponent: int,
                 precision: float, value_1_1: float):
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
        self.value_1_1 = value_1_1

    @classmethod
    def from_bytes(cls, data: bytes) -> 'Header':
        """
        Parse header from raw bytes.
        """
        if len(data) != 50:
            raise ValueError(f"Header must be exactly 50 bytes, got {len(data)}")

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
            'value_1_1': (36, 50, float),
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
        parsed['grid'] = (letter_to_thousands(parsed['grid'][0]),
                          letter_to_thousands(parsed['grid'][1]))

        return cls(**parsed)

    def __repr__(self):
        return (f"Header(year={self.year}, month={self.month}, day={self.day}, "
                f"hour={self.hour}, forecast={self.forecast}, level={self.level}, "
                f"grid={self.grid}, variable='{self.variable}', exponent={self.exponent}, "
                f"precision={self.precision}, value_1_1={self.value_1_1})")


class IndexRecordFixed:
    """
    Represents the fixed 108-byte portion of an ARL index record.
    Contains projection and grid information.
    """

    N_BYTES = 108

    def __init__(self, source: str, forecast_hour: int, minutes: int,
                 pole_lat: float, pole_lon: float, tangent_lat: float,
                 tangent_lon: float, grid_size: float, orientation: float,
                 cone_angle: float, sync_x: float, sync_y: float,
                 sync_lat: float, sync_lon: float,
                 nx: int, ny: int, nz: int, vertical_coord: int,
                 index_length: int):
        self.source = source
        self.forecast_hour = forecast_hour
        self.minutes = minutes
        self.pole_lat = pole_lat
        self.pole_lon = pole_lon
        self.tangent_lat = tangent_lat
        self.tangent_lon = tangent_lon
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
                f"tangent_lat={self.tangent_lat}, tangent_lon={self.tangent_lon}, "
                f"grid_size={self.grid_size}, orientation={self.orientation}, "
                f"cone_angle={self.cone_angle}, sync_x={self.sync_x}, sync_y={self.sync_y}, "
                f"sync_lat={self.sync_lat}, sync_lon={self.sync_lon}, nx={self.nx}, "
                f"ny={self.ny}, nz={self.nz}, vertical_coord={self.vertical_coord}, "
                f"index_length={self.index_length})")


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
    Combines the fixed header portion with the variable levels portion.
    """

    def __init__(self, fixed: IndexRecordFixed, extended: IndexRecordExtended):
        self._fixed = fixed
        self._extended = extended

        self.source = fixed.source
        self.forecast_hour = fixed.forecast_hour
        self.minutes = fixed.minutes
        self.pole_lat = fixed.pole_lat
        self.pole_lon = fixed.pole_lon
        self.tangent_lat = fixed.tangent_lat
        self.tangent_lon = fixed.tangent_lon
        self.grid_size = fixed.grid_size
        self.orientation = fixed.orientation
        self.cone_angle = fixed.cone_angle
        self.sync_x = fixed.sync_x
        self.sync_y = fixed.sync_y
        self.sync_lat = fixed.sync_lat
        self.sync_lon = fixed.sync_lon
        self.nx = fixed.nx
        self.ny = fixed.ny
        self.nz = fixed.nz
        self.vertical_coord = fixed.vertical_coord
        self.index_length = fixed.index_length
        self.levels = extended.levels

    def __repr__(self):
        return (f"IndexRecord(source='{self.source}', forecast_hour={self.forecast_hour}, "
                f"minutes={self.minutes}, pole_lat={self.pole_lat}, pole_lon={self.pole_lon}, "
                f"tangent_lat={self.tangent_lat}, tangent_lon={self.tangent_lon}, "
                f"grid_size={self.grid_size}, orientation={self.orientation}, "
                f"cone_angle={self.cone_angle}, sync_x={self.sync_x}, sync_y={self.sync_y}, "
                f"sync_lat={self.sync_lat}, sync_lon={self.sync_lon}, nx={self.nx}, "
                f"ny={self.ny}, nz={self.nz}, vertical_coord={self.vertical_coord}, "
                f"index_length={self.index_length}, levels={repr(self.levels)})")
    
    @property
    def num_records


class Record:
    def __init__(self, header, data):
        self.header = header
        self.data = data


class PackedData:
    """
    ARL (Air Resources Laboratory) packed meteorological data file reader.
    
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

        # Build the index header and record
        cursor = 0
        self.header = Header.from_bytes(data[cursor:cursor + 50])
        cursor += 50
        end_fixed = cursor + IndexRecordFixed.N_BYTES
        fixed = IndexRecordFixed.from_bytes(data[cursor:end_fixed])
        extended = IndexRecordExtended.from_bytes(data[end_fixed:cursor + fixed.index_length],
                                                  nz=fixed.nz)
        cursor += fixed.index_length
        self.index_record = IndexRecord(fixed, extended)

        # Calculate grid size
        self.nx = self.index_record.nx + self.header.grid[0]
        self.ny = self.index_record.ny + self.header.grid[1]
        nxy = self.nx * self.ny

        # Calculate record size
        self.record_length = 50 + nxy  # 50-byte header + nxy data points
        

        # Build index
        self.index = self._build_index()
    
    def _build_index(self):
        """
        Build a lookup index of variables by level and name.
        
        Returns
        ------- 
        """
        # Calculate the size per time step
        self.time_step_length =  self.record_length * self.index_record.nz * self.index_record.num_records

        index = {}
        position = 0
        for lvl in self.index_record.levels:
            level_num = lvl['level']
            index[level_num] = {}
            for var_name, checksum in lvl['vars']:
                index[level_num][var_name] = position * self.record_length
        return index


def unpack_data(cpack, nx, ny, nexp, var1):
    """
    Unpacks the ARL packed data format into a 2D numpy array.

    Args:
        cpack (bytes): The byte array of packed data (nx * ny).
        nx (int): Grid dimension in the x-direction.
        ny (int): Grid dimension in the y-direction.
        nexp (int): Scaling exponent.
        var1 (float): The starting value for the first grid point.

    Returns:
        numpy.ndarray: A 2D array with the unpacked data.
    """
    rdata = np.zeros((ny, nx), dtype=np.float64)
    
    # The scale factor is used to convert the packed integer back to a float.
    # It's derived from the scaling exponent (NEXP) in the record header.
    scale = 2.0**(7 - nexp)
    vold = var1
    indx = 0
    
    # print("\n--- Unpacked Data Samples ---")
    
    # The data is stored as a difference from the previous value.
    # We iterate through each grid point to reconstruct the absolute value.
    for j in range(ny):
        for i in range(nx):
            # The unpacking formula:
            # 1. Get the byte value (0-255).
            # 2. Subtract 127 to center the deviation around 0.
            # 3. Divide by the scale factor.
            # 4. Add to the previous value (vold) to get the true value.
            if indx < len(cpack):
                rdata[j, i] = (cpack[indx] - 127.0) / scale + vold
            else:
                # Handle cases where packed data is shorter than expected
                print(f"Warning: Packed data is short. Stopping at index {indx}", file=sys.stderr)
                return rdata

            vold = rdata[j, i]
            indx += 1
            
            # Print samples from the corners of the grid for verification,
            # matching the diagnostic output of the original Fortran code.
            # Fortran is 1-based, Python is 0-based, so we add 1 for display.
            # if i < 2 and j < 2:
                # print(f"Row: {j+1:5d}, Col: {i+1:5d}, Byte: {cpack[indx-1]:5d}, Value: {rdata[j,i]:12.4E}")
            # if i >= (nx - 2) and j >= (ny - 2):
                # print(f"Row: {j+1:5d}, Col: {i+1:5d}, Byte: {cpack[indx-1]:5d}, Value: {rdata[j,i]:12.4E}")

        # At the end of each row, the "previous value" is reset to the value
        # from the first cell of that same row.
        vold = rdata[j, 0]
        
    # print("---------------------------\n")
    return rdata


if __name__ == "__main__":
    # THIS WORKS, but is just a demo of reading the file
    with open(file_path, 'rb') as f:
        # --- Step 1: Read the Index Record to get grid dimensions ---
        # The first record has a fixed length of 158 bytes
        index_record_bytes = f.read(158)
        if len(index_record_bytes) < 158:
            print("Error: File is too small to be a valid ARL file.")

        label_bytes = index_record_bytes[0:50]
        header_bytes = index_record_bytes[50:158] # Only reads first 108 of header

        # Decode and parse the standard label
        label_str = label_bytes.decode('ascii', errors='ignore')
        iyr = int(label_str[0:2])
        imo = int(label_str[2:4])
        ida = int(label_str[4:6])
        ihr = int(label_str[6:8])
        kvar = label_str[14:18]
        
        print(f"Opened file date    : {iyr:5d}{imo:5d}{ida:5d}{ihr:5d}")

        if kvar != 'INDX':
            print("Error: This does not appear to be a valid ARL file.")
            print("The first record is not an 'INDX' record.")

        # Decode and parse the extended header to get grid info
        header_str = header_bytes.decode('ascii', errors='ignore')
        nx = int(header_str[93:96]) + 1000
        ny = int(header_str[96:99]) + 1000
        lenh = int(header_str[104:108])

        nxy = nx * ny
        # Each subsequent data record has a 50-byte label + packed data
        record_len = nxy + 50
        
        print(f"Grid size and lrec  : {nx} {ny} {nxy} {record_len}")
        print(f"Header record size  : {lenh:5d}")
        
        # --- Step 2: Loop through all data records in the file ---
        krec = 1
        while True:
            # Seek to the beginning of the next record
            offset = (krec - 1) * record_len
            f.seek(offset)
            
            record_bytes = f.read(record_len)
            if len(record_bytes) < record_len:
                print("\nEnd of file reached.")
                break

            data_label_bytes = record_bytes[0:50]
            cpack = record_bytes[50:]
            
            # Decode and parse the data record's label
            data_label_str = data_label_bytes.decode('ascii', errors='ignore').strip()
            kvar = data_label_str[14:18]
            nexp = int(data_label_str[18:22].strip())
            # Precision and Var1 are stored in scientific notation
            prec = float(data_label_str[22:36].strip())
            var1 = float(data_label_str[36:50].strip())

            print(f"Record {krec}: {Header.from_bytes(data_label_bytes)}")

            if kvar != 'INDX':
                unpack_data(cpack, nx, ny, nexp, var1)
                
            krec += 1