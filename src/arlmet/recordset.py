"""RecordSet: the records for one valid time of an ARL file."""

from __future__ import annotations

import io
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Literal

import numpy.typing as npt
import pandas as pd
from typing_extensions import override

from arlmet.collection import VariableAccessor
from arlmet.grid import Grid
from arlmet.header import Header, split_grid_component
from arlmet.indexrecord import IndexRecord, LvlInfo, VarInfo, _derive_index_forecast
from arlmet.record import DataRecord, _check_dif_parent, _require_mode
from arlmet.vertical import VerticalAxis

if TYPE_CHECKING:
    from arlmet.file import File

__all__ = ["RecordSet"]


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
        Get a DataRecord by ``(level, variable)`` key.
    __contains__(key)
        ``(level, variable) in rs`` tests for a record; ``variable in rs``
        tests whether the variable is present at any level.
    __iter__()
        Iterate over DataRecords in this record set.
    __len__()
        Get the number of records in this record set.
    create_datarecord(variable, level, *, forecast, data=None, diff=None)
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
        Create a DataRecord for an existing record on disk (internal).
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

    def create_datarecord(
        self,
        variable: str,
        level: int,
        *,
        forecast: int,
        data: npt.ArrayLike | None = None,
        diff: str | Sequence[str] | None = None,
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
            Forecast hour to write into the record header. A field that is
            NaN everywhere is written as missing data, as HYSPLIT's writer
            leaves a field it never fills (``pakini.f``): forecast hour -1 and
            no values, whatever ``forecast`` says. Readers skip it, and
            :meth:`DataRecord.read` returns NaN for it.
        data : numpy.ndarray, optional
            Initial ``(ny, nx)`` field values to assign. When omitted, assign
            the whole field later with ``record[:] = values``.
        diff : str or sequence of str, optional
            Name of a trailing DIF record to derive from the parent field: the
            field minus the packed parent, what packing lost. Give several
            names to write several DIF records after the parent, in order;
            each holds what packing left after the parent and the DIF records
            before it, so a reader that adds them all (as HYSPLIT does) gets
            back to the field. One is the usual case. Each name must be one
            that HYSPLIT's name table pairs with ``variable`` at ``level``:
            ``DIFW`` with ``WWND`` or ``DZDT``, ``DIFT`` with ``TKEN`` or
            ``TKEU``, and at the surface (level 0) any ``DIF*`` name with
            ``TPP*`` or ``PRT*``.

        Returns
        -------
        DataRecord
            Writable data record for the requested variable and level.

        Raises
        ------
        ValueError
            If ``variable`` is a ``DIF*`` name, or a ``diff`` name is not one
            the name table pairs with ``variable`` at ``level``.
        """
        _require_mode(self, "w")
        if variable.startswith("DIF"):
            raise ValueError(
                "Create DIF records through the parent record using diff='DIF...'."
            )
        diff_names = (diff,) if isinstance(diff, str) else tuple(diff or ())
        for name in diff_names:
            _check_dif_parent(name, variable, level)
        dr = self._create_datarecord(
            position=-1, variable=variable, level=level, forecast=forecast
        )
        for name in diff_names:
            dr._create_diff(position=-1, variable=name, forecast=forecast)
        dr._derive_diff_on_pack = bool(diff_names)
        if data is not None:
            dr[:] = data
        return dr

    def _records_by_level(self) -> dict[int, list[DataRecord]]:
        """Each level's writable records in stream order, each parent followed by its DIF records."""
        n_levels = len(self.file.vertical_axis.levels)
        by_level: dict[int, list[DataRecord]] = {level: [] for level in range(n_levels)}
        for dr in self.records:
            if dr.level < 0 or dr.level >= n_levels:
                raise ValueError(
                    f"DataRecord level {dr.level} is outside the configured vertical axis."
                )
            by_level[dr.level].extend((dr, *dr.diffs))
        return by_level

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

        def record_forecast(record: DataRecord) -> int:
            """Return the forecast hour for a writable or on-disk DataRecord."""
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
            dr._pack()
            for rec in (dr, *dr.diffs):
                forecast_hours.add(record_forecast(rec))

        # Derive the index record forecast hour from the data records, ensuring consistency
        forecast = _derive_index_forecast(
            record_forecasts=forecast_hours, explicit_forecast=self.forecast
        )

        level_records = self._records_by_level()
        grid_x = split_grid_component(self.grid.nx)[0]
        grid_y = split_grid_component(self.grid.ny)[0]
        levels = [
            LvlInfo(
                level=level,
                height=float(height),
                variables=[
                    (
                        dr.variable,
                        VarInfo(
                            checksum=dr.checksum,
                            reserved=(dr._reserved or "")[:1],
                        ),
                    )
                    for dr in level_records[level]
                ],
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
            nx=self.grid.nx,
            ny=self.grid.ny,
            nz=len(levels),
            vertical_flag=vaxis.flag,
            levels=levels,
        )

    def _flush(self):
        """Write the index record and all pending data records to disk."""
        _require_mode(self, "w")
        index = self._build_index_record()
        fh = self.file.handle

        if self.position == -1:
            fh.seek(0, io.SEEK_END)
            self.position = fh.tell()
        else:
            fh.seek(self.position)

        fh.write(index.to_record_bytes(self.record_length))

        # The records in the order the index record lists them.
        for records in self._records_by_level().values():
            for record in records:
                record._flush()

    def __getitem__(self, key: tuple[int, str]) -> DataRecord:
        if not isinstance(key, tuple) or len(key) != 2:
            raise KeyError("Key must be a tuple of (level, variable).")
        return self._datarecords[(self.time, *key)]

    def __iter__(self) -> Iterator[DataRecord]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self._datarecords)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str):
            return any(r.variable == key for r in self._datarecords.values())
        if isinstance(key, tuple) and len(key) == 2:
            try:
                return (self.time, *key) in self._datarecords
            except TypeError:  # unhashable key parts
                return False
        return False

    @override
    def __repr__(self) -> str:
        t = self.time.strftime("%Y-%m-%d %H:%M")
        return f"RecordSet(time={t}, forecast={self.forecast}, n={len(self)})"
