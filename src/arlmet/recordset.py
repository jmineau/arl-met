"""RecordCollection protocol, VariableView, VariableAccessor, and RecordSet."""

from __future__ import annotations

import io
from collections import OrderedDict
from collections.abc import Iterator
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from arlmet.collection import VariableAccessor
from arlmet.grid import Grid
from arlmet.header import Header, split_grid_component
from arlmet.index import IndexRecord, LvlInfo, VarInfo, _derive_index_forecast
from arlmet.record import DataRecord, require_mode
from arlmet.vertical import VerticalAxis

if TYPE_CHECKING:
    from arlmet.file import File


class RecordSet:
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
    __getitem__(key)
        Get a DataRecord by (level, variable) key.
    __iter__()
        Iterate over DataRecords in this record set.
    __len__()
        Get the number of records in this record set.
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
        self.forecast = forecast
        self._datarecords: OrderedDict[tuple[pd.Timestamp, int, str], DataRecord] = (
            OrderedDict()
        )
        self.variables = VariableAccessor(self)

    @property
    def mode(self) -> Literal["r", "w"]:
        """Access mode of the record set, inferred from the position."""
        return "w" if self.position == -1 else "r"

    @property
    def source(self) -> str:
        """RecordSet file source"""
        return self.file.source

    @property
    def grid(self) -> Grid:
        """RecordSet file grid"""
        return self.file.grid

    @property
    def vertical_axis(self) -> VerticalAxis:
        """RecordSet file vertical axis"""
        return self.file.vertical_axis

    @property
    def record_length(self) -> int:
        """Record length in bytes for this record set, derived from the file grid."""
        return self.file.record_length

    @property
    def records(self) -> list[DataRecord]:
        """List of DataRecords in this record set."""
        return list(self._datarecords.values())

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

        # Store the record in the record set's internal mapping
        self._datarecords[(self.time, level, variable)] = dr

        return dr

    @require_mode("w")
    def create_datarecord(
        self,
        variable: str,
        level: int,
        forecast: int,
        data: np.ndarray | None = None,
        diff: str | None = None,
    ) -> DataRecord:
        """
        Create a writable DataRecord attached to this time step.

        Parameters
        ----------
        variable : str
            Four-character ARL variable name.
        level : int
            ARL level index for the record.
        forecast : int
            Forecast hour to write into the record header.
            Missing data should use a value of -1.
        data : numpy.ndarray, optional
            Initial ``(ny, nx)`` field values to assign.
        diff : str, optional
            Name of a trailing DIF record to derive from the parent field.

        Returns
        -------
        DataRecord
            Writable data record for the requested variable and level.
        """
        if variable.startswith("DIF"):
            raise ValueError(
                "Create DIF records through the parent record using diff='DIF...'."
            )
        if diff is not None:
            self.file.register_diff_binding(diff_name=diff, parent_name=variable)
        dr = self._create_datarecord(
            position=-1, variable=variable, level=level, forecast=forecast
        )
        if diff is not None:
            dr._create_diff(position=-1, variable=diff, forecast=forecast)
            dr._derive_diff_on_pack = True
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

        def record_forecast(record: DataRecord) -> int:
            # Writable records may still hold raw header fields in a dict.
            # Read forecast directly from that state so index assembly does not
            # materialize a full Header object for every record.
            header_state = record._header
            if isinstance(header_state, dict):
                forecast = header_state.get("forecast")
                if forecast is None:
                    raise ValueError(
                        f"Writable DataRecord '{record.variable}' at level {record.level} is missing a forecast hour."
                    )
                return int(forecast)
            return record.forecast

        for dr in self.records:
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

            forecast_hours.add(record_forecast(dr))
            level_records[dr.level][dr.variable] = dr
            dr._pack()
            if dr.diff is not None:
                forecast_hours.add(record_forecast(dr.diff))
                level_records[dr.level][dr.diff.variable] = dr.diff

        # Derive the index record forecast hour from the data records, ensuring consistency
        forecast = _derive_index_forecast(
            record_forecasts=forecast_hours, explicit_forecast=self.forecast
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

        fh.write(index.to_record_bytes(self.record_length))

        for level in index.levels:
            for name in level.variables:
                self._lookup_flush_record(level.level, name)._flush()

    def _lookup_flush_record(self, level: int, variable: str) -> DataRecord:
        key = (self.time, level, variable)
        record = self._datarecords.get(key)
        if record is not None:
            return record

        for parent in self.records:
            if (
                parent.level == level
                and parent.diff is not None
                and parent.diff.variable == variable
            ):
                return parent.diff

        raise KeyError(f"No writable record found for ({level}, {variable}).")

    def __getitem__(self, key) -> DataRecord:
        if not isinstance(key, tuple) or len(key) != 2:
            raise KeyError("Key must be a tuple of (level, variable).")
        return self._datarecords[(self.time, *key)]

    def __iter__(self) -> Iterator[DataRecord]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self._datarecords)
