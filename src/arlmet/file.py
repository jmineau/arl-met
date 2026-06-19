"""File class for reading and writing ARL meteorology binary files."""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, BinaryIO, Literal, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from typing_extensions import override
from xarray.backends import CachingFileManager

from arlmet._time import ensure_timestamp
from arlmet.grid import Grid, Projection
from arlmet.header import record_length_from_grid
from arlmet.index import IndexRecord
from arlmet.record import DataRecord, _require_mode
from arlmet.recordset import RecordSet, VariableAccessor
from arlmet.vertical import VerticalAxis

if TYPE_CHECKING:
    import xarray as xr


class File:
    """
    Read or write an ARL meteorology file.

    Parameters
    ----------
    path : path-like
        Location of the ARL file on disk.
    mode : {"r", "w"}, default "r"
        File mode. Read mode scans the file immediately; write mode expects
        the caller to provide ``source``, ``grid``, and ``vertical_axis``
        before creating records.
    source : str, optional
        Four-character ARL source identifier used when writing.
    grid : Grid, optional
        Horizontal grid metadata used when writing.
    vertical_axis : VerticalAxis, optional
        Vertical axis metadata used when writing.

    Attributes
    ----------
    path : pathlib.Path
        Filesystem path for the ARL file.
    mode : {"r", "w"}
        Active file mode.
    times : list[pandas.Timestamp]
        Sorted valid times discovered in the file.
    source : str
        ARL source identifier.
    grid : Grid
        Horizontal grid metadata.
    vertical_axis : VerticalAxis
        Vertical coordinate metadata.
    variables : VariableAccessor
        Lazy accessor for variable-wise views inherited from RecordCollection.

    Methods
    -------
    create_grid(...)
        Build and attach a Grid when writing a new file.
    create_recordset(time, forecast=None)
        Create a writable RecordSet for one valid time.
    sample_points(points, variables, ...)
        Interpolate fields at arbitrary lon/lat/z sample points.
    extract_subset(destination, ...)
        Write a spatial/vertical subset to a new ARL file and return it.
    to_dataset(...)
        Project the file into the simplified analysis xarray Dataset.
    close()
        Flush pending writes and release the file handle.

    Examples
    --------
    >>> import arlmet
    >>> with arlmet.File("met.arl") as met:
    ...     met.times[0]
    Timestamp('2024-07-18 00:00:00')
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        mode: Literal["r", "w"] = "r",
        source: str | None = None,
        grid: Grid | None = None,
        vertical_axis: VerticalAxis | None = None,
    ):
        # File attrs
        self.path = Path(path)
        self.mode: Literal["r", "w"] = mode

        if self.mode not in ("r", "w"):
            raise ValueError("Mode must be 'r' (read) or 'w' (write).")

        # Open the binary file handle
        bmode = self.mode + "b"
        self._manager = CachingFileManager(open, self.path, mode=bmode)
        self._handle: BinaryIO | None = None

        # Must be consistent throughout the file
        self._source: str | None = source
        self._grid: Grid | None = grid
        self._vaxis: VerticalAxis | None = vertical_axis

        # Initialize recordsets as an ordered dict to preserve time order
        # Mapping: time -> RecordSet
        self._recordsets: OrderedDict[pd.Timestamp, RecordSet] = OrderedDict()
        self._diff_parents: dict[str, str] = {}
        self.variables = VariableAccessor(self)

        # Scan the file to populate recordsets in read mode
        if self.mode != "w":
            self._scan()

    @property
    def handle(self) -> BinaryIO:
        if self._handle is None:
            # Hot record read/write paths hit this repeatedly, so keep one
            # acquired handle per File instead of reentering the manager.
            # xarray's CachingFileManager.acquire() returns IO[Any]; opening in
            # binary mode guarantees BinaryIO at runtime.
            self._handle = cast(BinaryIO, self._manager.acquire())
        return self._handle

    @property
    def size(self) -> int:
        return self.path.stat().st_size

    @property
    def source(self) -> str:
        if self._source is None:
            raise ValueError("Source has not been set for this File.")
        return self._source

    @source.setter
    def source(self, value: str):
        _require_mode(self, "w")
        self._source = value

    @property
    def grid(self) -> Grid:
        if self._grid is None:
            raise ValueError("Grid has not been set for this File.")
        return self._grid

    @grid.setter
    def grid(self, value: Grid):
        _require_mode(self, "w")
        if not isinstance(value, Grid):
            raise TypeError("grid must be a Grid instance.")
        self._grid = value

    @property
    def vertical_axis(self) -> VerticalAxis:
        if self._vaxis is None:
            raise ValueError("Vertical axis has not been set for this File.")
        return self._vaxis

    @vertical_axis.setter
    def vertical_axis(self, value: VerticalAxis):
        _require_mode(self, "w")
        if not isinstance(value, VerticalAxis):
            raise TypeError("vertical_axis must be a VerticalAxis instance.")
        self._vaxis = value

    @property
    def times(self) -> list[pd.Timestamp]:
        """Return a sorted list of timestamps in the file."""
        return sorted(self._recordsets.keys())

    @property
    def records(self) -> list[DataRecord]:
        """List of all DataRecords in the file across all RecordSets."""
        return [
            record
            for recordset in self._recordsets.values()
            for record in recordset.records
        ]

    @property
    def record_length(self) -> int:
        return record_length_from_grid(self.grid)

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
        """
        Create and attach the horizontal grid metadata for a writable file.

        Parameters
        ----------
        nx : int
            Number of grid points in the x direction.
        ny : int
            Number of grid points in the y direction.
        pole_lat, pole_lon : float
            Projection pole definition from the ARL index record.
        tangent_lat, tangent_lon : float
            Reference latitude and longitude that define the projection.
        grid_size : float
            Grid spacing in kilometres at the projection reference point.
        orientation : float
            Rotation of the grid y-axis relative to true north.
        cone_angle : float
            Projection cone angle used for stereographic, Lambert, or
            Mercator grids.
        sync_x, sync_y : float
            One-based grid coordinates of the synchronization point.
        sync_lat, sync_lon : float
            Geographic coordinates of the synchronization point.

        Returns
        -------
        Grid
            The created grid instance, also stored on the file.
        """
        _require_mode(self, "w")
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
        position: int,
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

    def create_recordset(
        self, time: pd.Timestamp | str, *, forecast: int | None = None
    ) -> RecordSet:
        """
        Create a writable RecordSet for one valid time.

        Parameters
        ----------
        time : pandas.Timestamp or compatible datetime-like
            Valid time for the new record set.
        forecast : int, optional
            Forecast hour for the index record header.
            HYSPLIT docs are unclear on this, but conversion code appears to
            use the forecast hour from the first variable specified in the config file.
            This is brittle in `arlmet`s case, so we chose to either allow
            specifying it here or an index's forecast hour will be set to the minimum
            forecast hour among its variables (defaulting to -1 when all variables are missing data).

        Returns
        -------
        RecordSet
            Writable record set associated with ``time``.
        """
        _require_mode(self, "w")
        if self.source is None or self.grid is None:
            raise ValueError("Source and Grid must be set to create RecordSets.")

        position = -1  # New recordsets have no on-disk position yet
        source = grid = None  # skip checks in _create_recordset
        ts = ensure_timestamp(time)
        return self._create_recordset(
            position=position, source=source, grid=grid, time=ts, forecast=forecast
        )

    def register_diff_binding(self, diff_name: str, parent_name: str) -> None:
        """Record and validate the explicit parent binding for a generated DIF name."""
        _require_mode(self, "w")
        if not diff_name.startswith("DIF"):
            raise ValueError(
                f"Generated diff record names must start with 'DIF', got '{diff_name}'."
            )

        bound_parent = self._diff_parents.get(diff_name)
        if bound_parent is not None and bound_parent != parent_name:
            raise ValueError(
                f"Difference record '{diff_name}' is already bound to parent "
                f"'{bound_parent}', not '{parent_name}'."
            )

        self._diff_parents[diff_name] = parent_name

    def add_record(
        self,
        time: pd.Timestamp | str,
        variable: str,
        *,
        level: int,
        forecast: int | None = None,
        data: npt.ArrayLike | None = None,
    ) -> DataRecord:
        """Add one writable DataRecord, creating its RecordSet if needed."""
        _require_mode(self, "w")
        time = ensure_timestamp(time)

        if time in self._recordsets:
            recordset = self._recordsets[time]
        else:
            recordset = self.create_recordset(time)

        # Check if data is missing or effectively empty
        is_empty = data is None
        if not is_empty:
            # Convert to numpy to handle xarray, pandas, or lists uniformly
            arr = np.asanyarray(data)
            # Check if array is empty or all elements are NaN
            if arr.size == 0 or np.all(pd.isna(arr)):
                is_empty = True

        if is_empty:
            if forecast is None:
                forecast = -1
            elif forecast != -1:
                # Warn if a forecast hour is provided for missing data, since it will be ignored
                raise ValueError("Forecast must be -1 for missing data.")
        elif forecast is None:
            # Raise if data is valid but no forecast was supplied
            raise ValueError("forecast must be supplied when data is present")

        return recordset.create_datarecord(
            variable=variable,
            level=level,
            forecast=forecast,
            data=data,
        )

    def _scan(self) -> None:
        """Populate RecordSet objects by walking the on-disk index records."""
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
            record_length = self.record_length
            fh.seek(position + record_length)

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

                        # Keep track of previous data record for diff assignment
                        prev_dr = dr

                    position += record_length  # go to next record

            # Move file pointer to the start of the next index record
            fh.seek(position)

    def close(self) -> None:
        """Flush pending writes and close the managed binary file handle."""
        try:
            if self.mode == "w":
                for rs in self._recordsets.values():
                    if rs.position == -1:
                        if len(rs) == 0:
                            continue
                        rs._flush()
                self.handle.flush()
        finally:
            # Close the file manager — this releases the underlying file handle.
            # Any mmap objects created from it become invalid and are GC'd automatically.
            self._manager.close()
            self._handle = None

    def sample_points(
        self,
        points: pd.DataFrame | Mapping[str, Any],
        variables: str | Iterable[str],
        *,
        time: pd.Timestamp | str | None = None,
        z_kind: str = "pressure",
        method: str = "linear",
    ) -> pd.DataFrame:
        """
        Sample fields from this file at arbitrary lon/lat/z points.

        Parameters
        ----------
        points : Any
            Table-like object with ``lon``, ``lat``, ``z``, and optionally
            ``time`` columns.
        variables : str or iterable of str
            One or more ARL variables to interpolate.
        time : pandas.Timestamp or str, optional
            Default or override time when ``points`` does not include a
            ``time`` column.
        z_kind : {"pressure", "native", "agl", "msl"}, default "pressure"
            Interpretation of the ``z`` coordinate.
        method : {"linear", "nearest"}, default "linear"
            Horizontal interpolation method.

        Returns
        -------
        pandas.DataFrame
            Copy of ``points`` with one result column per requested variable.

        Examples
        --------
        >>> import pandas as pd
        >>> import arlmet
        >>> pts = pd.DataFrame({"lon": [-111.9], "lat": [40.7], "z": [850.0]})
        >>> with arlmet.File("met.arl") as met:
        ...     met.sample_points(pts, ["UWND", "VWND"])
        """
        # Delayed import: ops sit on top of file, so file's use of the sampling
        # op is lazy to avoid a file <-> ops import cycle (see File.extract_subset).
        from arlmet.ops.sample import _sample_points_from_file

        return _sample_points_from_file(
            self,
            points,
            variables,
            time=time,
            z_kind=z_kind,
            method=method,
        )

    def to_dataset(
        self,
        *,
        drop_variables: Sequence[str] | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        levels: list[int] | tuple[int, ...] | None = None,
    ) -> xr.Dataset:
        """Project this file into the simplified analysis Dataset representation."""
        from arlmet.xarray.dataset import _build_dataset_from_file

        return _build_dataset_from_file(
            self,
            drop_variables=drop_variables,
            bbox=bbox,
            levels=levels,
        )

    def extract_subset(
        self,
        destination_path: str | os.PathLike[str],
        *,
        bbox: tuple[float, float, float, float] | None = None,
        levels: Iterable[int] | None = None,
        variables: Iterable[str] | None = None,
    ) -> File:
        """
        Write a spatial/vertical subset of this file to a new ARL file.

        Parameters
        ----------
        destination_path : path-like
            Output ARL file path.
        bbox : tuple[float, float, float, float], optional
            Geographic bounding box ``(west, south, east, north)`` in degrees.
        levels : iterable of int, optional
            ARL level indices to keep. Output levels are compacted and
            renumbered from zero while preserving the selected level heights.
        variables : iterable of str, optional
            Variable names to keep. All variables are included by default.

        Returns
        -------
        File
            The newly written subset, opened in read mode. Close it when done
            (or use it as a context manager).

        Examples
        --------
        >>> import arlmet
        >>> with arlmet.File("met.arl") as met:
        ...     with met.extract_subset("subset.arl", bbox=(-114, 39, -110, 42)) as sub:
        ...         ds = sub.to_dataset()
        """
        from arlmet.ops.subset import extract_subset

        return extract_subset(
            self.path,
            destination_path,
            bbox=bbox,
            levels=levels,
            variables=variables,
        )

    def __getitem__(self, key: str | int | pd.Timestamp) -> RecordSet:
        if isinstance(key, str):
            # Allow lookup by string/int time representation
            key = ensure_timestamp(key)
        elif isinstance(key, int):
            # Allow lookup by positional index
            key = list(self._recordsets.keys())[key]
        return self._recordsets[key]

    def __iter__(self) -> Iterator[pd.Timestamp]:
        return iter(self._recordsets)

    def __len__(self) -> int:
        return len(self._recordsets)

    def __contains__(self, key: object) -> bool:
        try:
            ts = ensure_timestamp(key)
        except Exception:
            return False
        return ts in self._recordsets

    @override
    def __repr__(self) -> str:
        grid_str = (
            f"{self._grid.nx}\u00d7{self._grid.ny}"
            if self._grid is not None
            else "None"
        )
        levels_str = (
            str(len(self._vaxis._levels)) if self._vaxis is not None else "None"
        )
        return (
            f"File({self.path.name!r}, mode={self.mode!r}, "
            f"times={len(self)}, grid={grid_str}, levels={levels_str})"
        )

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
