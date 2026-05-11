"""DataRecord and require_mode for ARL meteorology binary I/O."""

from __future__ import annotations

import io
from functools import wraps
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from arlmet.grid import Grid, GridWindow
from arlmet.header import Header, record_length_from_grid, split_grid_component
from arlmet.packing import calculate_checksum, pack, unpack
from arlmet.vertical import VerticalAxis

if TYPE_CHECKING:
    import xarray as xr

    from arlmet.recordset import RecordSet


def require_mode(*allowed_modes):
    """
    Restrict a method to specific ARL read/write modes.

    Parameters
    ----------
    *allowed_modes : str
        One or more mode strings, typically ``"r"`` or ``"w"``.

    Returns
    -------
    collections.abc.Callable
        Decorator that raises :class:`io.UnsupportedOperation` when the bound
        instance mode is not allowed.
    """

    def decorator(func):
        """Wrap *func* with an ARL mode check."""

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            """Raise UnsupportedOperation when the instance mode is disallowed."""
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
    One ARL data record representing a single 2D variable slice.

    Parameters
    ----------
    recordset : RecordSet
        Parent record set for the record.
    position : int
        Byte offset of the record on disk, or ``-1`` for a writable in-memory
        record.
    level : int
        ARL vertical level index.
    variable : str
        Four-character ARL variable name.
    forecast : int, optional
        Forecast hour required when constructing a writable record.
    checksum : int, optional
        Stored ARL checksum from the index record.
    reserved : str, optional
        Reserved one-character metadata from the index record.

    Attributes
    ----------
    ndim : int
        NumPy-style dimensionality, always ``2``.
    recordset : RecordSet
        Parent record set.
    position : int
        Byte position of the record in the file.
    level : int
        ARL level index.
    variable : str
        Variable name for the record.
    shape : tuple[int, int]
        Grid shape as ``(ny, nx)``.
    data : array-like
        Cached unpacked data, loaded lazily on first access.

    Methods
    -------
    read(window=None)
        Unpack the record from disk, optionally for a subset window.
    verify_checksum()
        Validate the packed bytes against the stored checksum.
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
        self._unpacked: npt.NDArray[Any] | None = None  # The unpacked data
        self._diff: DataRecord | None = None  # The difference DataRecord if applicable
        self._derive_diff_on_pack = False
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
        if not variable.startswith("DIF"):
            raise ValueError(
                f"Difference record names must start with 'DIF', got '{variable}'."
            )
        if self.mode == "w" and forecast is None:
            if isinstance(self._header, dict):
                forecast = self._header.get("forecast")
            elif isinstance(self._header, Header):
                forecast = self._header.forecast
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
    def diff(self) -> DataRecord | None:
        """Return the attached DIF record when present."""
        return self._diff

    @property
    def bytes(self) -> bytes:
        """
        Get the packed bytes for this data record.
        """
        if self._bytes is None:
            if self.mode == "r":
                fh = self.recordset.file.handle
                fh.seek(self.position)
                self._bytes = fh.read(self.n_bytes)
            else:
                # Pack data to get bytes
                packed = self._pack()
                self._bytes = self.header.tobytes() + packed.tobytes()
        assert self._bytes is not None
        return self._bytes

    @property
    def n_bytes(self) -> int:
        """
        Get the number of bytes in the packed data record (including the header).
        """
        return record_length_from_grid(grid=self.grid)

    @property
    def header(self) -> Header:
        if not isinstance(self._header, Header):
            if self.mode == "r":
                fh = self.recordset.file.handle
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
                header_state = self._header
                if (
                    not {"precision", "exponent", "initial_value"}
                    <= header_state.keys()
                ):
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
        Verify the packed payload against the checksum stored in metadata.

        Returns
        -------
        bool
            True if the checksum matches, False otherwise.
        """
        if self.mode == "r":
            fh = self.recordset.file.handle
            fh.seek(self.position + Header.N_BYTES)
            packed = fh.read(self.n_bytes - Header.N_BYTES)
        else:
            packed = self._pack().tobytes()

        calculated_checksum = calculate_checksum(packed=packed)
        return calculated_checksum == self.checksum

    @property
    def dtype(self) -> np.dtype[Any]:
        if self._unpacked is None:
            # Data not loaded yet; assume float32
            return np.dtype(np.float32)
        return self._unpacked.dtype

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape of the data grid."""
        grid = self.grid
        return (grid.ny, grid.nx)

    @property
    def data(self) -> npt.NDArray[Any]:
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

    def __array__(self, dtype=None, copy=None) -> npt.NDArray[Any]:
        array = np.asarray(self.data)
        if dtype and np.dtype(dtype) != array.dtype:
            return array.astype(dtype)
        if copy:
            return array.copy()
        return array

    def __getitem__(self, key) -> Any:
        return self.data[key]

    @require_mode("r")
    def read(self, window: GridWindow | None = None) -> npt.NDArray[np.float32]:
        """
        Read and unpack this record eagerly.

        Parameters
        ----------
        window : GridWindow, optional
            Spatial subset to unpack. When omitted, the full grid is read.

        Returns
        -------
        numpy.ndarray
            Unpacked ``float32`` array for the full field or the requested
            window.
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
        unpacked_array = np.asarray(unpacked, dtype=np.float32)

        if self._diff is not None:
            unpacked_array = unpacked_array + self._diff.read(window=window)

        if window is None:
            self._unpacked = unpacked_array
        return unpacked_array

    def to_xarray(self, squeeze: bool = True) -> xr.DataArray:
        """
        Convert this DataRecord to an xarray.DataArray.

        Parameters
        ----------
        squeeze : bool, optional
            Whether to squeeze singleton dimensions. Default is True.

        Returns
        -------
        xarray.DataArray
            An xarray view of this data record.
        """
        import xarray as xr

        da = xr.DataArray(
            data=self.data,
            dims=self.grid.dims,
            coords=self.grid.calculate_coords(),
            name=self.variable,
        )

        da = da.expand_dims(("time", "level"))

        z_coords = self.recordset.vertical_axis.calculate_coords()
        level_value = z_coords["level"][self.level]

        da = da.assign_coords(
            time=[self.time],
            level=[level_value],
        )
        da.attrs["source"] = self.recordset.source
        return da.squeeze() if squeeze else da

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
        if isinstance(self._diff, DataRecord):
            self._diff._header = {"forecast": forecast}
            self._diff._packed = None
            self._diff._bytes = None
            self._diff._checksum = None

    @require_mode("r")
    def _load_from_disk(self, driver=None) -> Any:
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
        """Pack cached unpacked data and update header state for writing."""
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

            if isinstance(self._diff, DataRecord) and self._derive_diff_on_pack:
                reconstructed = np.asarray(
                    unpack(
                        packed=self._packed.tobytes(),
                        nx=self.grid.nx,
                        ny=self.grid.ny,
                        precision=precision,
                        exponent=exponent,
                        initial_value=initial_value,
                        driver=np,
                    ),
                    dtype=np.float32,
                )
                self._diff._unpacked = np.asarray(
                    self._unpacked - reconstructed,
                    dtype=np.float32,
                )
                self._diff._invalidate_write_cache()
                self._diff._pack()
            elif isinstance(self._diff, DataRecord) and self._diff._packed is None:
                self._diff._pack()

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
