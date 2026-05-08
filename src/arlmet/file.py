"""File class for reading and writing ARL meteorology binary files."""

from __future__ import annotations

import io
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from xarray.backends import CachingFileManager

from arlmet.grid import Grid, Projection
from arlmet.metadata import IndexRecord
from arlmet.record import require_mode
from arlmet.recordset import RecordCollection, RecordSet
from arlmet.sampling import sample_points_from_file, terrain_from_file
from arlmet.vertical import VerticalAxis


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
        self,
        position,
        source: str | None,
        grid: Grid | None,
        time: pd.Timestamp,
        *,
        forecast: int | None = None,
    ) -> RecordSet:
        """Internal factory method to create a new RecordSet."""
        if time in self._recordsets:
            raise ValueError(f"A RecordSet for time {time} already exists.")

        if source is not None and self._source != source:
            raise ValueError("Source mismatch when creating RecordSet.")

        if grid is not None and self._grid != grid:
            raise ValueError("Grid mismatch when creating RecordSet.")

        rs = RecordSet(file=self, position=position, time=time, forecast=forecast)
        self._recordsets[time] = rs
        return rs

    @require_mode("w", "a")
    def create_recordset(
        self, time, *, forecast: int | None = None
    ) -> RecordSet:
        """Factory method to create a new, writable RecordSet.

        Parameters
        ----------
        forecast :
            Forecast hour for the index record header. When copying or
            subsetting an existing file, pass ``src_recordset.forecast`` so
            the value is preserved exactly. When writing new data, omit and
            the value will be derived from the DataRecords (which must all
            share the same forecast hour).
        """
        if self.source is None or self.grid is None:
            raise ValueError("Source and Grid must be set to create RecordSets.")

        position = -1  # New recordsets have no on-disk position yet
        source = grid = None  # skip checks in _create_recordset
        return self._create_recordset(
            position=position, source=source, grid=grid, time=time, forecast=forecast
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
                position=position,
                source=index.source,
                grid=index.grid,
                time=index.time,
                forecast=index.forecast,
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

    def terrain(self, time: pd.Timestamp | str | None = None) -> np.ndarray:
        return terrain_from_file(self, time=time)

    def sample_points(
        self,
        points: Any,
        variables: str | Iterable[str],
        *,
        time: pd.Timestamp | str | None = None,
        z_kind: str = "pressure",
        method: str = "linear",
    ) -> pd.DataFrame:
        return sample_points_from_file(
            self,
            points,
            variables,
            time=time,
            z_kind=z_kind,
            method=method,
        )

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
