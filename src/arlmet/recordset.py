"""RecordCollection, VariableView, VariableAccessor, and RecordSet for ARL files."""

from __future__ import annotations

import io
from abc import abstractmethod
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from arlmet.grid import Grid
from arlmet.metadata import Header, IndexRecord, LvlInfo, VarInfo, split_grid_component
from arlmet.record import DataRecord, require_mode
from arlmet.vertical import VerticalAxis

if TYPE_CHECKING:
    from arlmet.file import File


class RecordCollection(Mapping):
    """
    Shared mapping interface for collections of ARL data records.

    Attributes
    ----------
    records : list[DataRecord]
        Materialized list of records in insertion order.
    record_size : int
        Binary size in bytes of one ARL record for the collection grid.
    variables : VariableAccessor
        Lazy variable-wise accessor over the underlying records.
    """

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


class VariableView:
    """
    Lazy multi-dimensional view of one ARL variable.

    Parameters
    ----------
    source : RecordCollection
        File-like or recordset-like collection that owns the variable.
    name : str
        Variable name to expose.

    Attributes
    ----------
    data : numpy.ndarray
        Materialized array for the selected variable.
    dtype : numpy.dtype
        NumPy dtype of the materialized data.
    ndim : int
        Number of dimensions in the view.
    shape : tuple
        Array shape in ``(time, level, y, x)`` or squeezed form.
    """

    def __init__(self, source: RecordCollection, name: str):
        self.source = source
        self.name = name

        # File has a `times` attribute; RecordSet does not — use duck typing to
        # avoid importing File here (file.py imports recordset.py).
        self._is_file_view = hasattr(source, "times")

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

    def to_xarray(self, squeeze=True):
        """
        Convert this variable view to an xarray object.

        Parameters
        ----------
        squeeze : bool, default True
            Remove length-1 dimensions from the result.

        Returns
        -------
        xarray.DataArray
            Lazy xarray view over the selected variable.
        """
        # Delayed import: xarray.py imports from recordset.py — cycle broken at runtime
        from arlmet.xarray import variable_view_to_xarray

        return variable_view_to_xarray(self, squeeze=squeeze)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """Compute the array and return a numpy array."""
        return np.asarray(self.data, dtype=dtype, copy=copy)

    def __getitem__(self, key) -> npt.ArrayLike:
        """Slice the lazy array."""
        return self.data[key]


class VariableAccessor(Mapping):
    """Dictionary-like accessor that returns VariableView objects by name."""

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
    """
    Records for one valid time within an ARL file.

    Parameters
    ----------
    file : File
        Parent ARL file handle.
    position : int
        Byte offset of the index record on disk, or ``-1`` for a new writable
        record set.
    time : pandas.Timestamp
        Valid time represented by the record set.
    forecast : int, optional
        Forecast hour stored in the index record header.

    Attributes
    ----------
    file : File
        Parent ARL file.
    position : int
        Byte offset of the index record.
    time : pandas.Timestamp
        Valid time of the record set.
    forecast : int or None
        Forecast hour associated with the index record.
    records : list[DataRecord]
        Data records stored at this time.
    variables : VariableAccessor
        Lazy variable accessor inherited from RecordCollection.

    Methods
    -------
    create_datarecord(variable, level, forecast=-1, data=None)
        Create a writable data record for this time.
    """

    def __init__(
        self,
        file: File,
        position: int,
        time: pd.Timestamp,
        *,
        forecast: int | None = None,
    ):
        self.file = file
        self.position = position
        self.time = time
        #: Forecast hour from the index record header. Populated from the
        #: parsed IndexRecord when reading; pass explicitly when writing a
        #: copy or subset so the value is preserved rather than re-derived.
        self.forecast = forecast

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
    def vertical_axis(self) -> VerticalAxis:
        return self.file.vertical_axis

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
        Create a writable DataRecord attached to this time step.

        Parameters
        ----------
        variable : str
            Four-character ARL variable name.
        level : int
            ARL level index for the record.
        forecast : int, default -1
            Forecast hour to write into the record header.
        data : numpy.ndarray, optional
            Initial ``(ny, nx)`` field values to assign.

        Returns
        -------
        DataRecord
            Writable data record for the requested variable and level.
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
        heights = vaxis.levels.tolist()
        if not heights:
            raise ValueError("Vertical axis must contain at least one level.")

        forecast_hours: set[int] = set()
        level_records: dict[int, OrderedDict[str, DataRecord]] = {
            level: OrderedDict() for level in range(len(heights))
        }

        for dr in self.records:
            if dr.variable.startswith("DIF") or dr._diff is not None:
                raise NotImplementedError(
                    "Writing difference records is not implemented."
                )
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

        if self.forecast is not None:
            # Propagated from source index record (e.g. during subset/copy).
            # Per-record forecast hours may legitimately differ (e.g. GDAS).
            forecast = self.forecast
        elif len(forecast_hours) == 1:
            forecast = next(iter(forecast_hours))
        else:
            raise ValueError(
                "All DataRecords in a RecordSet must share the same forecast hour "
                "when writing new data. Pass forecast= to create_recordset() when "
                "copying from a source file that mixes forecast hours."
            )

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
            forecast=forecast,
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
            forecast=forecast,
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
        """Write the index record and all pending data records to disk."""
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
