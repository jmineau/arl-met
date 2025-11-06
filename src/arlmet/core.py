import io
from abc import abstractmethod
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from functools import wraps
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from xarray.backends import CachingFileManager

from arlmet.grid import Grid3D, create_grid
from arlmet.metadata import Header, IndexRecord
from arlmet.packing import calculate_checksum, pack, unpack

try:
    import dask

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False


def open_dataset(filename_or_obj, drop_variables=None, squeeze=True):
    """
    Open an ARLMet file and convert it to an xarray Dataset.

    Parameters
    ----------
    filename_or_obj : str, Path, or file-like object
        The path to the ARL meteorology file or a file-like object.
    drop_variables : list of str, optional
        List of variable names to drop from the resulting Dataset.
    squeeze : bool, default True
        Whether to squeeze dimensions of size 1 in the resulting Dataset.

    Returns
    -------
    xarray.Dataset
        The ARL meteorology data as an xarray Dataset.
    """
    met = File(filename_or_obj)

    ds = met.to_xarray(drop_variables=drop_variables, squeeze=squeeze)
    return ds if isinstance(ds, xr.Dataset) else ds.to_dataset()


def require_mode(*allowed_modes):
    """
    Decorator to ensure a method is called only when the file is in an allowed mode.

    Args:
        *allowed_modes: A sequence of strings representing the allowed modes (e.g., 'r', 'w').

    Raises:
        io.UnsupportedOperation: If the instance's mode is not in the allowed modes.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, "mode") or self.mode not in allowed_modes:
                raise io.UnsupportedOperation(
                    f"'{func.__name__}' is only available in mode(s) {allowed_modes}, "
                    f"not '{getattr(self, 'mode', 'unknown')}'."
                )
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


class DataRecord:
    """
    A class that represents a 2D array from a differentially packed
    data source, with support for lazy loading, caching, and a
    NumPy-compatible interface.
    """

    ndim: int = 2

    def __init__(
        self,
        recordset: "RecordSet",
        position: int,
        level: int,
        variable: str,
        forecast: int | None = None,
        checksum: int | None = None,
        reserved: str | None = None,
    ):
        """
        Initializes the array representation.
        """
        self.recordset = recordset
        self.position = position
        self.level = level
        self.variable = variable

        # Cached attributes
        self._header: Header | dict[str, Any] | None = None  # Record header
        self._bytes: bytes | None = None  # The packed record bytes
        self._unpacked: npt.ArrayLike | None = None  # The unpacked data
        self._diff: DataRecord | None = None  # The difference DataRecord if applicable
        self._checksum = checksum  # The stored checksum if applicable
        self._reserved = reserved  # Reserved field if applicable

        if self.mode == "w" and forecast is None:
            raise ValueError(
                "Forecast hour must be specified when creating a new DataRecord."
            )
        elif forecast is not None:
            if self.mode == "r":
                raise ValueError(
                    "Forecast hour should not be specified when reading an existing DataRecord."
                )
            else:
                self._header = {"forecast": forecast}

    def _create_diff(
        self, position, variable, forecast=None, checksum=None, reserved=None
    ) -> "DataRecord":
        """
        Create a DataRecord representing the difference between this record
        and another DataRecord (self - other).

        Returns
        -------
        DataRecord
            A new DataRecord representing the difference.
        """
        if self._diff is not None:
            raise ValueError("Difference record already exists.")
        diff = DataRecord(
            recordset=self.recordset,
            position=position,
            variable=variable,
            level=self.level,
            forecast=forecast,
            checksum=checksum,
            reserved=reserved,
        )
        self._diff = diff
        return diff

    @property
    def mode(self) -> Literal["r", "w"]:
        return "w" if self.position == -1 else "r"

    @property
    def source(self) -> str:
        """
        Get the source associated with this data record.
        """
        return self.recordset.source

    @property
    def grid(self) -> Grid3D:
        """
        Get the grid associated with this data record.
        """
        return self.recordset.grid

    @property
    def time(self) -> pd.Timestamp:
        """
        Get the time associated with this data record.
        """
        return self.recordset.time

    @property
    def n_bytes(self) -> int:
        """
        Get the number of bytes in the packed data record.
        """
        grid = self.grid
        return Header.N_BYTES + grid.nx * grid.ny

    @property
    def header(self) -> Header:
        if not isinstance(self._header, Header):
            if self.mode == "r":
                if self._bytes is None:
                    self._bytes = self._read_bytes(
                        manager=self.recordset.file._manager,
                        offset=self.position,
                        n_bytes=self.n_bytes,
                    )

                # Parse header from bytes
                header = Header.from_bytes(self._bytes[: Header.N_BYTES])

                if header.variable != self.variable or header.level != self.level:
                    raise ValueError(
                        f"DataRecord header mismatch at position {self.position}: "
                        f"expected variable '{self.variable}' level {self.level}, "
                        f"got variable '{header.variable}' level '{header.level}'"
                    )

                self._header = header
            else:
                if self._bytes is None:
                    raise ValueError(
                        "Data must be packed before accessing header in write mode."
                    )
                else:  # Construct header from packed data
                    raise NotImplementedError(
                        "Header construction from packed data in write mode is not implemented."
                    )

                    # Get header fields
                    time = self.recordset.time
                    forecast = self._header.get("forecast")
                    exponent = self._header.get("exponent")
                    precision = self._header.get("precision")
                    initial_value = self._header.get("initial_value")
                    if forecast is None:
                        raise ValueError(
                            "Forecast hour must be set to construct header in write mode."
                        )
                    if exponent is None or precision is None or initial_value is None:
                        raise ValueError(
                            "Precision, exponent, and initial value must be set to construct header in write mode."
                        )

                    # Construct header
                    header = Header(
                        year=time.year,
                        month=time.month,
                        day=time.day,
                        hour=time.hour,
                        forecast=forecast,
                        level=self.level,
                        grid=self.grid,  # TODO this is wrong should be a tuple?
                        variable=self.variable,
                        exponent=exponent,
                        precision=precision,
                        initial_value=initial_value,
                    )
                    self._header = header
        return self._header

    @property
    def forecast(self) -> int:
        """
        Get the forecast hour for this data record.
        """
        return self.header["forecast"]

    @property
    def checksum(self) -> int:
        """
        Get the stored checksum for this data record.
        """
        if self._checksum is None:
            if self.mode == "r":
                raise ValueError("No stored checksum for this data record.")
            else:
                raise ValueError(
                    "Data must be packed before accessing checksum in write mode."
                )
        return self._checksum

    def verify_checksum(self) -> bool:
        """
        Verify the checksum of the packed data against the stored checksum.

        Returns
        -------
        bool
            True if the checksum matches, False otherwise.
        """
        raise NotImplementedError("Checksum verification is not yet implemented.")
        if self._packed is None:
            if self._packed_dask is not None:  # FIXME
                # Compute dask array to get packed data
                self._packed = self._packed_dask.compute()
            else:
                raise ValueError("No packed data to verify checksum.")

        # Calculate checksum
        calculated_checksum = calculate_checksum(packed=self._packed.tobytes())
        return calculated_checksum == self.checksum

    @property
    def dtype(self) -> npt.DTypeLike:
        if self._unpacked is None:
            # Data not loaded yet; assume float32
            return np.float32
        return self._unpacked.dtype

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape of the data grid."""
        grid = self.grid
        return (grid.ny, grid.nx)

    @property
    def data(self) -> npt.ArrayLike:
        """
        Get the data for this record.

        If the data has already been loaded, returns the cached data.
        If dask is available, returns a dask array that lazily loads the data from disk.

        Returns
        -------
        array-like
            The data array for this record.
        """
        if self._unpacked is None:
            if self.mode == "r":
                if DASK_AVAILABLE:
                    import dask
                    import dask.array as da

                    nxy = self.grid.nx * self.grid.ny
                    if nxy > 10**9:
                        # For large data grids (> 1GB), add dask layers for each computation step
                        unpacked = self._load_from_disk(driver=da)
                    else:  # For smaller data grids, delay the loading the entire grid at once
                        unpacked = dask.delayed(self._load_from_disk)(driver=np)
                        unpacked = da.from_delayed(
                            unpacked, shape=self.shape, dtype=self.dtype
                        )
                else:
                    unpacked = self._load_from_disk()
                self._unpacked = unpacked
            else:
                raise ValueError("No data to read.")
        return self._unpacked

    def __array__(self, dtype=None, copy=None) -> npt.NDArray:
        array = self[...]
        if dtype and np.dtype(dtype) != array.dtype:
            return array.astype(dtype)
        if copy:
            return array.copy()
        return array

    def __getitem__(self, key) -> npt.NDArray:
        return self.data[key]

    @require_mode("w")
    def __setitem__(self, key, value) -> None:
        if self._unpacked is None:
            if key is Ellipsis:
                # Setting the entire array
                self._unpacked = np.asarray(value)
            else:
                raise ValueError("Data must be initialized before setting slices.")

        self._unpacked[key] = value

    @require_mode("r")
    def _load_from_disk(self, driver=None) -> npt.ArrayLike:
        """
        Loads data from disk, returning a numpy array.
        """
        # Get header (dont delay)
        header = self.header  # this will load the bytes from disk

        # Read packed bytes if not already loaded
        if self._bytes is None:
            self._bytes = self._read_bytes(
                manager=self.recordset.file._manager,
                offset=self.position,
                n_bytes=self.n_bytes,
            )

        # Unpack (will be delayed if dask is available)
        ny, nx = self.shape
        unpacked = unpack(
            packed=self._bytes[Header.N_BYTES :],
            nx=nx,
            ny=ny,
            precision=header.precision,
            exponent=header.exponent,
            initial_value=header.initial_value,
            driver=driver,
        )

        # Handle diff record if present
        if self._diff is not None:
            unpacked = unpacked + self._diff._load_from_disk(driver=driver)

        return unpacked

    @staticmethod
    def _read_bytes(manager, offset, n_bytes) -> bytes:
        fh = manager.acquire()
        fh.seek(offset)
        return fh.read(n_bytes)

    @require_mode("w")
    def _pack(self) -> npt.NDArray[np.uint8]:
        if self._packed is None:
            if not isinstance(self._unpacked, np.ndarray):
                raise ValueError("Data to pack must be a numpy array.")
            if not isinstance(self._header, dict):
                raise ValueError(
                    "Header must be a dictionary in write mode before packing."
                )
            packed, precision, exponent, initial_value = pack(self._unpacked)
            self._packed = packed
            self._header["precision"] = precision
            self._header["exponent"] = exponent
            self._header["initial_value"] = initial_value

            # Calculate checksum
            # TODO

        return self._packed

    @require_mode("w")
    def _flush(self) -> None:
        """
        Flush the packed data to disk.
        """
        raise NotImplementedError("This function is not yet implemented.")
        # Pack data
        packed = self._pack()

        # Get file handle
        fh = self.recordset.file.handle

        # Write header
        header = Header(
            variable=self.variable,
            level=self.level,
            forecast=self._forecast,
            precision=self._precision,
            exponent=self._exponent,
            initial_value=self._initial_value,
        )
        if self.position == -1:
            # New record; append to end of file
            fh.seek(0, io.SEEK_END)
            self.position = fh.tell()
        else:
            fh.seek(self.position)
        fh.write(header.to_bytes())

        # Write packed data
        fh.write(packed.tobytes())

    def to_xarray(self, squeeze=True) -> xr.DataArray:
        """
        Convert this DataRecord to an xarray DataArray.
        """
        # Get 3D grid info
        dims = self.grid.dims
        coords = self.grid.coords

        # Construct DataArray
        coords_2d = {k: v for k, v in coords.items() if k in ("x", "y", "lon", "lat")}
        da = xr.DataArray(
            data=self.data, dims=dims[-2:], coords=coords_2d, name=self.variable
        )

        # Expand to 4D
        # Otherwise we cant assign derived coords to dims that don't exist
        da = da.expand_dims(("time", "level"))

        # Get derived level coordinates
        level = self.level
        height = coords["height"][level]
        # height_agl = coords['height_agl'][level]
        # height_msl = coords['height_msl'][level]

        # Add time and level coordinates
        da = da.assign_coords(
            time=[self.time],
            forecast=("time", [self.forecast]),
            level=[level],
            height=("level", [height]),
            # height_agl=('level', [height_agl]),
            # height_msl=('level', [height_msl]),
        )

        # Sort on dims
        # da = da.sortby(da.dims)

        # Add attributes
        da.attrs.update(**self.recordset.attrs)

        # Apply CF conventions TODO
        # if apply_cf:
        #     pass

        return da.squeeze() if squeeze else da


class _RecordCollection(Mapping):
    def __init__(self):
        # Initialize datarecords as an ordered dict to preserve insertion order
        # Mapping: (time, level, variable) -> DataRecord
        self._datarecords: OrderedDict[tuple[pd.Timestamp, int, str], DataRecord] = (
            OrderedDict()
        )

        # Variables accessor
        self.variables = VariableAccessor(self)

    @property
    @abstractmethod
    def source(self) -> str:
        pass

    @property
    @abstractmethod
    def grid(self) -> Grid3D:
        pass

    @property
    def attrs(self) -> dict:
        return {
            "source": self.source,
            "grid": self.grid,  # TODO how to represent grid and create grid from attrs
        }

    @property
    def records(self) -> list[DataRecord]:
        """Return a list of all DataRecords in the file."""
        return list(self._datarecords.values())

    @property
    def record_size(self) -> int:
        """
        Get the size of each data record in bytes.
        """
        grid = self.grid
        nxy = grid.nx * grid.ny
        return Header.N_BYTES + nxy

    def to_xarray(self, drop_variables=None, squeeze=True) -> xr.Dataset | xr.DataArray:
        """
        Convert the entire container to an xarray Dataset.
        """

        drop_variables = drop_variables or []

        ds = xr.combine_by_coords(
            [
                (
                    dr.to_xarray(squeeze=False).drop_vars("forecast")
                )  # forecast can change between variables
                for dr in self.records
                if dr.variable not in drop_variables
            ]
        )

        return ds.squeeze() if squeeze else ds


class VariableView:
    """
    A lazy, multidimensional view of a single variable, providing access to
    its data as a dask array and facilitating conversion to an xarray.DataArray.

    This view can represent a 4D cube (time, level, y, x) from a File object
    or a 3D slice (level, y, x) from a RecordSet object.
    """

    def __init__(self, source: _RecordCollection, name: str):
        self.source = source
        self.name = name

        self._is_file_view = isinstance(source, File)

    @property
    def data(self) -> npt.NDArray:
        """The Dask array representing the full variable view, always as 4D."""
        return self.to_xarray().data

    @property
    def dtype(self) -> npt.DTypeLike:
        return self.data.dtype

    @property
    def ndim(self) -> int:
        """Return the number of dimensions of the data cube."""
        return self.data.ndim

    @property
    def shape(self) -> tuple:
        """Return the shape of the data cube (time, level, y, x)"""
        return self.data.shape

    def to_xarray(self, squeeze=True) -> xr.DataArray:
        """Converts the view to an xarray DataArray."""
        da = xr.combine_by_coords(
            [
                dr.to_xarray(squeeze=False)
                for dr in self.source.records
                if dr.variable == self.name
            ]
        )
        if isinstance(da, xr.Dataset):
            da = da[self.name]
        return da.squeeze() if squeeze else da

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """Compute the dask array and return a numpy array."""
        return np.asarray(self.data, dtype=dtype, copy=copy)

    def __getitem__(self, key) -> npt.ArrayLike:
        """Slice the lazy dask array."""
        return self.data[key]


class VariableAccessor(Mapping):
    """
    A lazy, dictionary-like view of variables for a given File or RecordSet.
    It provides access to VariableView objects for each variable.
    """

    def __init__(self, source: _RecordCollection):
        self.source = source

    @property
    def _names(self) -> set[str]:
        """Dynamically get available variable names from the source."""
        if not hasattr(self.source, "records"):
            raise TypeError("Source must have a 'records' attribute.")
        return set(dr.variable for dr in self.source.records)

    def __getitem__(self, name: str) -> VariableView:
        return VariableView(source=self.source, name=name)

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({list(self.keys())})"


class RecordSet(_RecordCollection):
    def __init__(self, file: "File", position: int, time: pd.Timestamp):
        self.file = file
        self.position = position
        self.time = time

        # Initialize the base class
        super().__init__()

    @property
    def mode(self) -> Literal["r", "w"]:
        return "w" if self.position == -1 else "r"

    @property
    def source(self) -> str:
        """
        Get the source associated with this RecordSet.
        """
        return self.file.source

    @property
    def grid(self) -> Grid3D:
        """
        Get the grid associated with this RecordSet.
        """
        return self.file.grid

    def _create_datarecord(
        self,
        position: int,
        variable: str,
        level: int,
        forecast: int | None = None,
        checksum: int | None = None,
        reserved: str | None = None,
    ) -> DataRecord:
        """
        Internal method to create a DataRecord for an existing record on disk.
        """
        dr = DataRecord(
            recordset=self,
            position=position,
            variable=variable,
            level=level,
            forecast=forecast,
            checksum=checksum,
            reserved=reserved,
        )
        self._datarecords[(self.time, level, variable)] = dr
        self.file._datarecords[(self.time, level, variable)] = dr
        return dr

    @require_mode("w")
    def create_datarecord(
        self,
        variable: str,
        level: int,
        forecast: int = -1,
        data: np.ndarray | None = None,
    ) -> DataRecord:
        """
        Create a new DataRecord.
        """
        if not isinstance(forecast, int):
            raise ValueError("Forecast must be an integer.")
        dr = self._create_datarecord(
            position=-1, variable=variable, level=level, forecast=forecast
        )
        if data is not None:
            dr[:] = data
        return dr

    def _flush(self):
        # Placeholder for flushing data to the file.
        # Here, we would iterate through self._datarecords,
        # pack the data if write mode, and write it.
        # we would need to build the index record including
        # the LvlInfo and VarInfo structures. This would
        # also include calculating checksums for each variable.
        # need to check for diffs when writing data records.
        # raise error if no data to write
        # call _flush on each DataRecord
        raise NotImplementedError("Flushing RecordSet to disk is not yet implemented.")

    def __getitem__(self, key) -> DataRecord:
        if not isinstance(key, tuple) or len(key) != 2:
            raise KeyError("Key must be a tuple of (level, variable).")
        return self._datarecords[(self.time, *key)]

    def __iter__(self) -> Iterator[DataRecord]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self._datarecords)


class File(_RecordCollection):
    def __init__(
        self,
        path: Path | str,
        mode: Literal["r", "w", "a"] = "r",
        source: str | None = None,
        grid: Grid3D | None = None,
    ):
        self.path = Path(path)
        self.mode = mode

        # Must be consistent throughout the file
        self._source = source
        self._grid = grid

        if self.mode not in ("r", "w"):
            raise ValueError("Mode must be 'r' (read) or 'w' (write).")

        # Open the binary file handle
        bmode = self.mode + "b"
        self._manager = CachingFileManager(open, self.path, mode=bmode)

        # Initialize recordsets as an ordered dict to preserve time order
        # Mapping: time -> RecordSet
        self._recordsets: OrderedDict[pd.Timestamp, RecordSet] = OrderedDict()

        # Initialize the base class
        super().__init__()

        # Scan the file to populate recordsets in read mode
        if self.mode != "w":
            self._scan()

    @property
    def handle(self) -> io.BufferedIOBase:
        return self._manager.acquire()

    @property
    def size(self) -> int:
        return self.path.stat().st_size

    @property
    def source(self) -> str:
        if self._source is None:
            raise ValueError("Source has not been set for this File.")
        return self._source

    @source.setter
    @require_mode("w")
    def source(self, value: str):
        self._source = value

    @property
    def grid(self) -> Grid3D:
        if self._grid is None:
            raise ValueError("Grid has not been set for this File.")
        return self._grid

    @grid.setter
    @require_mode("w")
    def grid(self, value: Grid3D | None):
        if not isinstance(value, Grid3D):
            raise TypeError("grid must be a Grid3D instance.")
        self._grid = value

    @property
    def times(self) -> list[pd.Timestamp]:
        """Return a sorted list of timestamps in the file."""
        return sorted(self._recordsets.keys())

    @require_mode("w")
    def create_grid(
        self,
        nx: int,
        ny: int,
        pole_lat: float,
        pole_lon: float,
        tangent_lat: float,
        tangent_lon: float,
        grid_size: float,
        orientation: float,
        cone_angle: float,
        sync_x: float,
        sync_y: float,
        sync_lat: float,
        sync_lon: float,
        reserved: float,
        vertical_flag: int,
        heights: Sequence[float],
    ) -> Grid3D:
        """Factory method to create and set the grid for this File."""
        if self._grid is not None:
            raise ValueError("Grid has already been set for this File.")

        grid = create_grid(
            nx=nx,
            ny=ny,
            pole_lat=pole_lat,
            pole_lon=pole_lon,
            tangent_lat=tangent_lat,
            tangent_lon=tangent_lon,
            grid_size=grid_size,
            orientation=orientation,
            cone_angle=cone_angle,
            sync_x=sync_x,
            sync_y=sync_y,
            sync_lat=sync_lat,
            sync_lon=sync_lon,
            reserved=reserved,
            vertical_flag=vertical_flag,
            heights=heights,
        )
        if not isinstance(grid, Grid3D):
            raise TypeError("Grid must be a Grid3D instance.")

        self._grid = grid
        return grid

    def _create_recordset(
        self, position, source: str | None, grid: Grid3D | None, time: pd.Timestamp
    ) -> RecordSet:
        """Internal factory method to create a new RecordSet."""
        if time in self._recordsets:
            raise ValueError(f"A RecordSet for time {time} already exists.")

        if source is not None and self._source != source:
            raise ValueError("Source mismatch when creating RecordSet.")

        if grid is not None and self._grid != grid:
            raise ValueError("Grid mismatch when creating RecordSet.")

        rs = RecordSet(file=self, position=position, time=time)
        self._recordsets[time] = rs
        return rs

    @require_mode("w", "a")
    def create_recordset(self, time) -> RecordSet:
        """Factory method to create a new, writable RecordSet."""
        if self.source is None or self.grid is None:
            raise ValueError("Source and Grid must be set to create RecordSets.")

        position = -1  # New recordsets have no on-disk position yet
        source = grid = None  # skip checks in _create_recordset
        return self._create_recordset(
            position=position, source=source, grid=grid, time=time
        )

    def _scan(self) -> None:
        # Scan the file to populate recordsets in read mode
        fh = self.handle

        while fh.tell() < self.size:
            # Get starting position of each recordset
            position = fh.tell()

            # Parse index record
            try:
                index = IndexRecord.from_position(fh, position=position)
            except EOFError:
                break  # End of file

            # Set source when reading the first index record
            if self._source is None:
                self._source = index.source

            # Set grid when reading the first index record
            if self._grid is None:
                self._grid = index.grid

            # Create a RecordSet for this index record (time)
            rs = self._create_recordset(
                position=position, source=index.source, grid=index.grid, time=index.time
            )

            # Skip to the end of the index record
            record_size = self.record_size
            fh.seek(position + self.record_size)

            # Read data records for this index record
            position = fh.tell()  # start of data records
            prev_dr = None
            for lvl in index.levels:
                for var in lvl.variables:
                    checksum = lvl.variables[var].checksum
                    reserved = lvl.variables[var].reserved
                    if var.startswith("DIF"):
                        # Assign as diff record to previous data record
                        if prev_dr is None:
                            raise ValueError(
                                f"Difference record found for variable '{var}' "
                                f"at position {position} without a preceding data record."
                            )
                        prev_dr._create_diff(
                            position=position,
                            variable=var,
                            checksum=checksum,
                            reserved=reserved,
                        )
                    else:
                        # Create data record
                        prev_dr = rs._create_datarecord(
                            position=position,
                            variable=var,
                            level=lvl.level,
                            checksum=checksum,
                            reserved=reserved,
                        )
                    position += record_size  # go to next record

            # Move file pointer to the start of the next index record
            fh.seek(position)

    def close(self) -> None:
        """Flushes any changes and closes the file."""
        if self.mode == "w":
            for rs in self._recordsets.values():
                if rs.position == -1:
                    rs._flush()

        # Close the file manager
        self._manager.close()
        # TODO do i need to close mmaps?

    def __getitem__(self, key) -> RecordSet:
        if isinstance(key, str):
            # Allow lookup by string/int time representation
            key = pd.Timestamp(key)
        elif isinstance(key, int):
            # Allow lookup by positional index
            key = list(self._recordsets.keys())[key]
        return self._recordsets[key]

    def __iter__(self) -> Iterator[pd.Timestamp]:
        return iter(self._recordsets)

    def __len__(self) -> int:
        return len(self._recordsets)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
