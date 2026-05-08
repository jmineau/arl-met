"""DataRecord and require_mode for ARL meteorology binary I/O."""

from __future__ import annotations

import io
from functools import wraps
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from arlmet.grid import Grid, GridWindow
from arlmet.metadata import Header, split_grid_component
from arlmet.packing import calculate_checksum, pack, unpack
from arlmet.vertical import VerticalAxis

if TYPE_CHECKING:
    from arlmet.recordset import RecordSet


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
        recordset: RecordSet,
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
    ) -> DataRecord:
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
