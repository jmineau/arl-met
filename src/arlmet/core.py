import io
from abc import abstractmethod
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from functools import wraps
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from xarray.backends import CachingFileManager

from arlmet.grid import Grid, GridWindow, Projection
from arlmet.metadata import Header, IndexRecord, LvlInfo, VarInfo, split_grid_component
from arlmet.packing import calculate_checksum, pack, unpack
from arlmet.vertical import Surface, VerticalAxis


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
        self._packed: npt.NDArray[np.uint8] | None = None  # Packed payload
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
    def bytes(self) -> bytes:
        """
        Get the packed bytes for this data record.
        """
        if self._bytes is None:
            if self.mode == "r":
                fh = self.recordset.file._manager.acquire()
                fh.seek(self.position)
                self._bytes = fh.read(self.n_bytes)
            else:
                # Pack data to get bytes
                packed = self._pack()
                self._bytes = self.header.tobytes() + packed.tobytes()
        return self._bytes

    @property
    def n_bytes(self) -> int:
        """
        Get the number of bytes in the packed data record (including the header).
        """
        grid = self.grid
        return Header.N_BYTES + grid.nx * grid.ny

    @property
    def header(self) -> Header:
        if not isinstance(self._header, Header):
            if self.mode == "r":
                fh = self.recordset.file._manager.acquire()
                fh.seek(self.position)
                header = Header.from_bytes(fh.read(Header.N_BYTES))

                if header.variable != self.variable or header.level != self.level:
                    raise ValueError(
                        f"DataRecord header mismatch at position {self.position}: "
                        f"expected variable '{self.variable}' level {self.level}, "
                        f"got variable '{header.variable}' level '{header.level}'"
                    )

                self._header = header
            else:
                if not isinstance(self._header, dict):
                    raise ValueError(
                        "Header state must be a dictionary before constructing a writable record header."
                    )
                if self._diff is not None:
                    raise NotImplementedError(
                        "Writing difference records is not implemented."
                    )
                header_state = self._header
                if not {"precision", "exponent", "initial_value"} <= header_state.keys():
                    self._pack()
                    header_state = self._header
                    if not isinstance(header_state, dict):
                        raise ValueError(
                            "Writable header state must remain mutable until the header is built."
                        )

                time = self.recordset.time
                forecast = header_state.get("forecast")
                exponent = header_state.get("exponent")
                precision = header_state.get("precision")
                initial_value = header_state.get("initial_value")
                if forecast is None:
                    raise ValueError(
                        "Forecast hour must be set to construct header in write mode."
                    )
                if exponent is None or precision is None or initial_value is None:
                    raise ValueError(
                        "Precision, exponent, and initial value must be set to construct header in write mode."
                    )

                grid = (
                    split_grid_component(self.grid.nx)[0],
                    split_grid_component(self.grid.ny)[0],
                )
                self._header = Header(
                    year=time.year,
                    month=time.month,
                    day=time.day,
                    hour=time.hour,
                    forecast=int(forecast),
                    level=self.level,
                    grid=grid,
                    variable=self.variable,
                    exponent=int(exponent),
                    precision=float(precision),
                    initial_value=float(initial_value),
                )
        return self._header

    @property
    def source(self) -> str:
        """
        Get the source associated with this data record.
        """
        return self.recordset.source

    @property
    def grid(self) -> Grid:
        """
        Get the grid associated with this data record.
        """
        return self.recordset.grid

    @property
    def vertical_axis(self) -> VerticalAxis:
        """
        Get the vertical axis associated with this data record.
        """
        return self.recordset.vertical_axis

    @property
    def time(self) -> pd.Timestamp:
        """
        Get the time associated with this data record.
        """
        return self.recordset.time

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
        if self.mode == "r":
            fh = self.recordset.file._manager.acquire()
            fh.seek(self.position + Header.N_BYTES)
            packed = fh.read(self.n_bytes - Header.N_BYTES)
        else:
            packed = self._pack().tobytes()

        calculated_checksum = calculate_checksum(packed=packed)
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
        Otherwise, reads and unpacks the record eagerly as a NumPy array.

        Returns
        -------
        array-like
            The data array for this record.
        """
        if self._unpacked is None:
            if self.mode == "r":
                self._unpacked = self.read()
            else:
                raise ValueError("No data to read.")
        return self._unpacked

    def __array__(self, dtype=None, copy=None) -> npt.NDArray:
        array = np.asarray(self.data)
        if dtype and np.dtype(dtype) != array.dtype:
            return array.astype(dtype)
        if copy:
            return array.copy()
        return array

    def __getitem__(self, key) -> npt.NDArray:
        return self.data[key]

    @require_mode("r")
    def read(self, window: GridWindow | None = None) -> npt.NDArray[np.float32]:
        """
        Read this record eagerly, optionally unpacking only a subset window.
        """
        if window is None and isinstance(self._unpacked, np.ndarray):
            return self._unpacked

        fh = self.recordset.file.handle
        fh.seek(self.position)
        raw = fh.read(self.n_bytes)

        header = self._header
        if not isinstance(header, Header):
            header = Header.from_bytes(raw[: Header.N_BYTES])
            if header.variable != self.variable or header.level != self.level:
                raise ValueError(
                    f"DataRecord header mismatch at position {self.position}: "
                    f"expected variable '{self.variable}' level {self.level}, "
                    f"got variable '{header.variable}' level '{header.level}'"
                )
            self._header = header

        unpacked = unpack(
            packed=raw[Header.N_BYTES :],
            nx=self.grid.nx,
            ny=self.grid.ny,
            precision=header.precision,
            exponent=header.exponent,
            initial_value=header.initial_value,
            window=window,
            driver=np,
        )

        if self._diff is not None:
            unpacked = unpacked + self._diff.read(window=window)

        if window is None and isinstance(unpacked, np.ndarray):
            self._unpacked = unpacked
        return np.asarray(unpacked, dtype=np.float32)

    @require_mode("w")
    def __setitem__(self, key, value) -> None:
        if self._unpacked is None:
            is_full_slice = key == slice(None)
            if key is Ellipsis or is_full_slice:
                # Setting the entire array
                self._unpacked = np.asarray(value)
            else:
                raise ValueError("Data must be initialized before setting slices.")

        self._unpacked[key] = value
        self._invalidate_write_cache()

    def _invalidate_write_cache(self) -> None:
        """
        Clear cached packed/header state after a writable record changes.
        """
        if self.mode != "w":
            return

        forecast = None
        if isinstance(self._header, dict):
            forecast = self._header.get("forecast")
        elif isinstance(self._header, Header):
            forecast = self._header.forecast

        self._header = {"forecast": forecast}
        self._packed = None
        self._bytes = None
        self._checksum = None

    @require_mode("r")
    def _load_from_disk(self, driver=None) -> npt.ArrayLike:
        """
        Loads data from disk, returning a numpy array.
        """
        # Get header (dont delay)
        header = self.header  # this will load the bytes from disk

        # Unpack (will be delayed if dask is available)
        ny, nx = self.shape
        unpacked = unpack(
            packed=self.bytes[Header.N_BYTES :],
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

            self._checksum = calculate_checksum(self._packed.tobytes())

        return self._packed

    @require_mode("w")
    def _flush(self) -> None:
        """
        Flush the packed data to disk.
        """
        raw = self.bytes
        fh = self.recordset.file.handle
        if self.position == -1:
            # New record; append to end of file
            fh.seek(0, io.SEEK_END)
            self.position = fh.tell()
        else:
            fh.seek(self.position)
        fh.write(raw)

    def to_xarray(self, squeeze=True) -> xr.DataArray:
        from arlmet.xarray import datarecord_to_xarray

        return datarecord_to_xarray(self, squeeze=squeeze)


class RecordCollection(Mapping):
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
    def grid(self) -> Grid:
        pass

    @property
    def attrs(self) -> dict:
        from arlmet.xarray import record_collection_attrs

        return record_collection_attrs(self)

    @property
    def forecast_by_time(self) -> OrderedDict[pd.Timestamp, int]:
        forecasts: OrderedDict[pd.Timestamp, set[int]] = OrderedDict()
        for dr in self.records:
            forecasts.setdefault(dr.time, set()).add(dr.forecast)

        mapping: OrderedDict[pd.Timestamp, int] = OrderedDict()
        for time, values in forecasts.items():
            if len(values) != 1:
                raise ValueError(
                    f"RecordCollection contains multiple forecast hours for time {time}."
                )
            mapping[time] = next(iter(values))
        return mapping

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
        from arlmet.xarray import record_collection_to_xarray

        return record_collection_to_xarray(
            self,
            drop_variables=drop_variables,
            squeeze=squeeze,
        )


class VariableView:
    """
    A lazy, multidimensional view of a single variable, providing access to
    its data as an xarray-backed lazy array and facilitating conversion to an
    xarray.DataArray.

    This view can represent a 4D cube (time, level, y, x) from a File object
    or a 3D slice (level, y, x) from a RecordSet object.
    """

    def __init__(self, source: RecordCollection, name: str):
        self.source = source
        self.name = name

        self._is_file_view = isinstance(source, File)

    @property
    def data(self) -> npt.NDArray:
        """The array representing the full variable view, always as 4D."""
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
        from arlmet.xarray import variable_view_to_xarray

        return variable_view_to_xarray(self, squeeze=squeeze)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """Compute the array and return a numpy array."""
        return np.asarray(self.data, dtype=dtype, copy=copy)

    def __getitem__(self, key) -> npt.ArrayLike:
        """Slice the lazy array."""
        return self.data[key]


class VariableAccessor(Mapping):
    """
    A lazy, dictionary-like view of variables for a given File or RecordSet.
    It provides access to VariableView objects for each variable.
    """

    def __init__(self, source: RecordCollection):
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


class RecordSet(RecordCollection):
    def __init__(self, file: "File", position: int, time: pd.Timestamp):
        self.file = file
        self.position = position
        self.time = time

        self._sfc_terrain: DataRecord | None = None
        self._sfc_pressure: DataRecord | None = None

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
    def grid(self) -> Grid:
        """
        Get the grid associated with this RecordSet.
        """
        return self.file.grid

    @property
    def surface(self) -> Surface:
        """
        Get the surface associated with this RecordSet.
        """
        return Surface(terrain=self._sfc_terrain, pressure=self._sfc_pressure)

    @property
    def vertical_axis(self) -> VerticalAxis:
        """
        Get the file-level vertical axis bound to this recordset's surface state.
        """
        return self.file.vertical_axis.with_surface(self.surface)

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

        # Store surface variables
        if variable == "SHGT":
            self._sfc_terrain = dr
        elif variable == "PRSS":
            self._sfc_pressure = dr

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

    def _build_index_record(self) -> IndexRecord:
        """
        Build the index record for this time step from the writable records.
        """
        if not self._datarecords:
            raise ValueError("Cannot flush an empty RecordSet.")
        if len(self.source) > 4:
            raise ValueError("ARL source identifiers must be 4 characters or fewer.")

        vaxis = self.file.vertical_axis
        heights = vaxis.heights.tolist()
        if not heights:
            raise ValueError("Vertical axis must contain at least one level.")

        forecast_hours: set[int] = set()
        level_records: dict[int, OrderedDict[str, DataRecord]] = {
            level: OrderedDict() for level in range(len(heights))
        }

        for dr in self.records:
            if dr.variable.startswith("DIF") or dr._diff is not None:
                raise NotImplementedError("Writing difference records is not implemented.")
            if len(dr.variable) > 4:
                raise ValueError(
                    f"Variable names must be 4 characters or fewer, got '{dr.variable}'."
                )
            if dr._unpacked is None:
                raise ValueError(
                    f"Writable DataRecord '{dr.variable}' at level {dr.level} has no data."
                )
            if dr.level < 0 or dr.level >= len(heights):
                raise ValueError(
                    f"DataRecord level {dr.level} is outside the configured vertical axis."
                )

            forecast_hours.add(dr.forecast)
            level_records[dr.level][dr.variable] = dr
            dr._pack()

        if len(forecast_hours) != 1:
            raise ValueError(
                "All DataRecords in a RecordSet must share the same forecast hour."
            )
        forecast_hour = forecast_hours.pop()

        grid_x, nx = split_grid_component(self.grid.nx)
        grid_y, ny = split_grid_component(self.grid.ny)
        levels = [
            LvlInfo(
                level=level,
                height=float(height),
                variables=OrderedDict(
                    (
                        name,
                        VarInfo(
                            checksum=dr.checksum,
                            reserved=(dr._reserved or "")[:1],
                        ),
                    )
                    for name, dr in level_records[level].items()
                ),
            )
            for level, height in enumerate(heights)
        ]

        projection = self.grid.projection
        header = Header(
            year=self.time.year,
            month=self.time.month,
            day=self.time.day,
            hour=self.time.hour,
            forecast=forecast_hour,
            level=0,
            grid=(grid_x, grid_y),
            variable="INDX",
            exponent=0,
            precision=0.0,
            initial_value=0.0,
        )
        return IndexRecord(
            header=header,
            source=self.source,
            forecast_hour=forecast_hour,
            minutes=self.time.minute,
            pole_lat=projection.pole_lat,
            pole_lon=projection.pole_lon,
            tangent_lat=projection.tangent_lat,
            tangent_lon=projection.tangent_lon,
            grid_size=projection.grid_size,
            orientation=projection.orientation,
            cone_angle=projection.cone_angle,
            sync_x=projection.sync_x,
            sync_y=projection.sync_y,
            sync_lat=projection.sync_lat,
            sync_lon=projection.sync_lon,
            reserved=vaxis.offset,
            nx=nx,
            ny=ny,
            nz=len(levels),
            vertical_flag=vaxis.flag,
            index_length=0,
            levels=levels,
        )

    @require_mode("w")
    def _flush(self):
        index = self._build_index_record()
        fh = self.file.handle

        if self.position == -1:
            fh.seek(0, io.SEEK_END)
            self.position = fh.tell()
        else:
            fh.seek(self.position)

        fh.write(index.to_record_bytes(self.record_size))

        for level in index.levels:
            for name in level.variables:
                self[(level.level, name)]._flush()

    def __getitem__(self, key) -> DataRecord:
        if not isinstance(key, tuple) or len(key) != 2:
            raise KeyError("Key must be a tuple of (level, variable).")
        return self._datarecords[(self.time, *key)]

    def __iter__(self) -> Iterator[DataRecord]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self._datarecords)


class File(RecordCollection):
    def __init__(
        self,
        path: Path | str,
        mode: Literal["r", "w", "a"] = "r",
        source: str | None = None,
        grid: Grid | None = None,
        vertical_axis: VerticalAxis | None = None,
    ):
        # File attrs
        self.path = Path(path)
        self.mode = mode

        if self.mode not in ("r", "w"):
            raise ValueError("Mode must be 'r' (read) or 'w' (write).")

        # Open the binary file handle
        bmode = self.mode + "b"
        self._manager = CachingFileManager(open, self.path, mode=bmode)

        # Must be consistent throughout the file
        self._source: str | None = source
        self._grid: Grid | None = grid
        self._vaxis: VerticalAxis | None = vertical_axis

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
    def grid(self) -> Grid:
        if self._grid is None:
            raise ValueError("Grid has not been set for this File.")
        return self._grid

    @grid.setter
    @require_mode("w")
    def grid(self, value: Grid):
        if not isinstance(value, Grid):
            raise TypeError("grid must be a Grid instance.")
        self._grid = value

    @property
    def vertical_axis(self) -> VerticalAxis:
        if self._vaxis is None:
            raise ValueError("Vertical axis has not been set for this File.")
        return self._vaxis

    @vertical_axis.setter
    @require_mode("w")
    def vertical_axis(self, value: VerticalAxis):
        if not isinstance(value, VerticalAxis):
            raise TypeError("vertical_axis must be a VerticalAxis instance.")
        self._vaxis = value

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
    ) -> Grid:
        """Factory method to create and set the grid for this File."""
        if self._grid is not None:
            raise ValueError("Grid has already been set for this File.")

        # Build projection
        proj = Projection(
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
        )

        # Create grid
        grid = Grid(projection=proj, nx=nx, ny=ny)

        self._grid = grid
        return grid

    def _create_recordset(
        self, position, source: str | None, grid: Grid | None, time: pd.Timestamp
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

            # Set vertical axis when reading the first index record
            if self._vaxis is None:
                self._vaxis = index.vertical_axis
            elif self._vaxis != index.vertical_axis:
                raise ValueError("Vertical axis mismatch between index records.")

            # Create a RecordSet for this index record (time)
            rs = self._create_recordset(
                position=position, source=index.source, grid=index.grid, time=index.time
            )

            # Skip to the end of the index record
            record_size = self.record_size
            fh.seek(position + record_size)

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
                        dr = rs._create_datarecord(
                            position=position,
                            variable=var,
                            level=lvl.level,
                            checksum=checksum,
                            reserved=reserved,
                        )

                        # Store surface variables
                        if var == "SHGT":
                            rs._sfc_terrain = dr
                        if var == "PRSS":
                            rs._sfc_pressure = dr

                        # Keep track of previous data record for diff assignment
                        prev_dr = dr

                    position += record_size  # go to next record

            # Move file pointer to the start of the next index record
            fh.seek(position)

    def close(self) -> None:
        """Flushes any changes and closes the file."""
        try:
            if self.mode == "w":
                for rs in self._recordsets.values():
                    if rs.position == -1:
                        rs._flush()
                self.handle.flush()
        finally:
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
