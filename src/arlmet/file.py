"""File class for reading and writing ARL meteorology binary files."""

from __future__ import annotations

import os
import warnings
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, BinaryIO, Literal, Self, cast

import numpy.typing as npt
import pandas as pd
from typing_extensions import override
from xarray.backends import CachingFileManager
from xarray.backends.locks import SerializableLock

from arlmet._time import ensure_timestamp
from arlmet.collection import VariableAccessor
from arlmet.exceptions import ARLFormatError, ARLFormatWarning
from arlmet.grid import Grid, Projection
from arlmet.header import record_length_from_grid
from arlmet.index import IndexRecord
from arlmet.record import DataRecord, _require_mode
from arlmet.recordset import RecordSet
from arlmet.vertical import VerticalAxis

if TYPE_CHECKING:
    import xarray as xr

__all__ = ["File"]


def _open_binary(path: str | os.PathLike[str], mode: str) -> BinaryIO:
    """
    Open ``path`` in binary ``mode`` ("r", "w", or "a").

    CachingFileManager switches mode "w" to "a" after the first open so that
    reopening a writer appends instead of truncating. It only recognizes the
    bare letter, so the manager gets "r"/"w" and the "b" is added here.
    """
    # open() only narrows to BinaryIO for literal modes.
    return cast(BinaryIO, open(path, mode + "b"))


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
    __getitem__(key)
        Get a RecordSet by valid time (Timestamp or string), or by position
        in ``times`` (int).
    __iter__()
        Iterate over valid times in ``times`` order (sorted).
    create_grid(...)
        Build and attach a Grid when writing a new file.
    create_recordset(time, *, forecast=None)
        Create a writable RecordSet for one valid time.
    add_record(time, variable, level, *, forecast, data=None, diff=None)
        Add one writable DataRecord, creating its RecordSet if needed.
    flush()
        Write pending record sets to disk and release their in-memory data.
    sample_points(points, variables, ...)
        Interpolate fields at arbitrary lon/lat/z sample points.
    extract_subset(dest, ...)
        Write a spatial/vertical subset to a new ARL file and return its path.
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
        *,
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
        self._manager = CachingFileManager(_open_binary, self.path, mode=self.mode)
        self._handle: BinaryIO | None = None
        # Record reads are a seek followed by a read on the one shared handle;
        # this lock keeps the pair atomic when threads read concurrently (e.g.
        # dask chunks). SerializableLock pickles, unlike threading.Lock.
        self._lock = SerializableLock()

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
        """Open binary handle to the file, reopened if the file cache closed it."""
        # Hot record read/write paths hit this repeatedly, so keep one
        # acquired handle per File instead of reentering the manager. xarray's
        # global file cache closes the least recently used file once it holds
        # `file_cache_maxsize` (128) files, so reacquire when that happens; the
        # manager reopens the file (in append mode for writers).
        if self._handle is None or self._handle.closed:
            # CachingFileManager.acquire() returns IO[Any]; _open_binary
            # guarantees BinaryIO at runtime.
            self._handle = cast(BinaryIO, self._manager.acquire())
        return self._handle

    def _read_at(self, position: int, n_bytes: int) -> bytes:
        """Read ``n_bytes`` starting at byte ``position``; safe to call from several threads."""
        with self._lock:
            fh = self.handle
            fh.seek(position)
            return fh.read(n_bytes)

    @property
    def size(self) -> int:
        """Size of the file on disk, in bytes."""
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
        """All DataRecords in the file, in ``times`` order."""
        return [
            record for time in self.times for record in self._recordsets[time].records
        ]

    @property
    def record_length(self) -> int:
        """Length in bytes of each record in the file, from the grid."""
        return record_length_from_grid(self.grid)

    def create_grid(
        self,
        nx: int,
        ny: int,
        *,
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
        """Create a new RecordSet (internal factory)."""
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
            This is brittle in arlmet's case, so we chose to either allow
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

    def _register_diff_binding(self, diff_name: str, parent_name: str) -> None:
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
        level: int,
        *,
        forecast: int,
        data: npt.ArrayLike | None = None,
        diff: str | None = None,
    ) -> DataRecord:
        """
        Add one writable DataRecord, creating its RecordSet if needed.

        Shorthand for ``create_recordset(time)`` (when ``time`` has no record
        set yet) followed by :meth:`RecordSet.create_datarecord`. A record set
        created here derives its index-record forecast from its records; call
        :meth:`create_recordset` first to set it explicitly.

        Parameters
        ----------
        time : pandas.Timestamp or str
            Valid time of the record.
        variable : str
            Four-character ARL variable name.
        level : int
            ARL level index for the record.
        forecast : int
            Forecast hour to write into the record header.
        data : array-like, optional
            ``(ny, nx)`` field values. When omitted, assign the whole field
            later with ``record[:] = values``.
        diff : str, optional
            Name of a trailing DIF record to derive from the parent field.

        Returns
        -------
        DataRecord
            Writable data record for ``variable`` at ``level`` and ``time``.
        """
        _require_mode(self, "w")
        ts = ensure_timestamp(time)
        recordset = self._recordsets.get(ts)
        if recordset is None:
            recordset = self.create_recordset(ts)
        return recordset.create_datarecord(
            variable, level, forecast=forecast, data=data, diff=diff
        )

    def _scan(self) -> None:
        """
        Populate RecordSet objects by walking the on-disk index records.

        Raises
        ------
        ARLFormatError
            If the file is not valid ARL: an unparseable index record, a size
            that is not a whole number of records, an index record that
            declares more data records than remain, inconsistent metadata
            between index records, or a time step repeated with different
            content.
        """
        # Scan the file to populate recordsets in read mode
        fh = self.handle
        size = self.size

        # Extent (position, n_bytes) of each time step's first occurrence,
        # used to compare against any repeated copy of the same time.
        extents: dict[pd.Timestamp, tuple[int, int]] = {}

        while fh.tell() < size:
            # Get starting position of each recordset
            position = fh.tell()

            # Parse index record
            try:
                index = IndexRecord.from_position(fh, position=position)
            except EOFError:
                break  # End of file
            except ARLFormatError as exc:
                raise ARLFormatError(f"{self.path}: {exc}") from exc

            # Set metadata when reading the first index record
            if self._source is None:
                self._source = index.source
            if self._grid is None:
                self._grid = index.grid
                # The first index record fixes the record length. Every record
                # has the same length, so the file must hold a whole number.
                if size % self.record_length != 0:
                    raise ARLFormatError(
                        f"{self.path}: file size {size} bytes is not a whole "
                        f"number of {self.record_length}-byte records "
                        f"({self.grid.nx}x{self.grid.ny} grid); the file is "
                        "truncated or corrupt."
                    )
            if self._vaxis is None:
                self._vaxis = index.vertical_axis
            elif self._vaxis != index.vertical_axis:
                raise ARLFormatError(
                    f"{self.path}: vertical axis mismatch between index records "
                    f"(index record for {index.time} at byte {position})."
                )

            # The time step is the index record plus one record per variable
            # per level; it must fit in what remains of the file.
            record_length = self.record_length
            n_data = sum(len(lvl.variables) for lvl in index.levels)
            n_bytes = (1 + n_data) * record_length
            if position + n_bytes > size:
                remaining = (size - position) // record_length - 1
                raise ARLFormatError(
                    f"{self.path}: index record for {index.time} at byte "
                    f"{position} declares {n_data} data records, but only "
                    f"{remaining} remain in the file; the file is truncated."
                )

            # Some NOAA archive files repeat a whole time step. Skip a
            # byte-identical copy; anything else is ambiguous, so raise.
            if index.time in extents:
                first_position, first_n_bytes = extents[index.time]
                if first_n_bytes != n_bytes or not self._same_bytes(
                    first_position, position, n_bytes
                ):
                    raise ARLFormatError(
                        f"{self.path}: time step {index.time} is repeated with "
                        f"different content (first at byte {first_position}, "
                        f"again at byte {position})."
                    )
                warnings.warn(
                    f"{self.path}: time step {index.time} is repeated "
                    f"(byte-identical copy at byte {position}); the repeated "
                    "copy was ignored.",
                    ARLFormatWarning,
                    stacklevel=3,
                )
                fh.seek(position + n_bytes)
                continue
            extents[index.time] = (position, n_bytes)

            # Create a RecordSet for this index record (time)
            try:
                rs = self._create_recordset(
                    position=position,
                    source=index.source,
                    grid=index.grid,
                    time=index.time,
                    forecast=index.forecast,
                )
            except ValueError as exc:
                # Source/grid mismatch with the first index record.
                raise ARLFormatError(
                    f"{self.path}: index record for {index.time} at byte "
                    f"{position}: {exc}"
                ) from exc

            # Read data records for this index record
            position += record_length  # start of data records
            prev_dr = None
            for lvl in index.levels:
                for var in lvl.variables:
                    checksum = lvl.variables[var].checksum
                    reserved = lvl.variables[var].reserved

                    if var.startswith("DIF"):
                        # Assign as diff record to previous data record
                        if prev_dr is None:
                            raise ARLFormatError(
                                f"{self.path}: difference record found for "
                                f"variable '{var}' at byte {position} without "
                                "a preceding data record."
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

    def _same_bytes(self, first: int, second: int, n_bytes: int) -> bool:
        """
        Compare two ``n_bytes`` spans of the file, one record at a time.

        Reading record by record keeps memory at two records even for large
        grids (an HRRR time step is ~200 MB).
        """
        fh = self.handle
        chunk = self.record_length
        for offset in range(0, n_bytes, chunk):
            length = min(chunk, n_bytes - offset)
            fh.seek(first + offset)
            a = fh.read(length)
            fh.seek(second + offset)
            b = fh.read(length)
            if a != b:
                return False
        return True

    def flush(self) -> None:
        """
        Write pending record sets to disk and release their in-memory data.

        Record sets are held in memory until the file is flushed or closed.
        When writing many time steps, call ``flush()`` after filling each one
        so memory stays bounded by one time step instead of the whole file.
        Flushed record sets cannot be modified.

        Examples
        --------
        >>> with arlmet.File(
        ...     "out.arl", mode="w", source="TEST", grid=grid, vertical_axis=vaxis
        ... ) as arl:
        ...     for time, fields in steps:  # fields: {(name, level): array}
        ...         rs = arl.create_recordset(time, forecast=0)
        ...         for (name, level), data in fields.items():
        ...             rs.create_datarecord(name, level=level, forecast=0, data=data)
        ...         arl.flush()
        """
        _require_mode(self, "w")
        for rs in self._recordsets.values():
            if rs.position == -1 and len(rs) > 0:
                rs._flush()
        self.handle.flush()

    def close(self) -> None:
        """Flush pending writes and close the managed binary file handle."""
        try:
            if self.mode == "w":
                self.flush()
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
        z_kind: Literal["native", "pressure", "agl", "msl"] = "pressure",
        method: Literal["linear", "nearest"] = "linear",
        earth_relative: bool = False,
    ) -> pd.DataFrame:
        """
        Sample fields from this file at arbitrary lon/lat/z points.

        Equivalent to :func:`arlmet.sample_points` with this file as the only
        input.

        Parameters
        ----------
        points : pandas.DataFrame or mapping
            Table-like object with ``lon``, ``lat`` (degrees), and ``z``
            columns, and optionally a ``time`` column. Any other columns are
            carried through to the result unchanged.
        variables : str or iterable of str
            One or more ARL field names (e.g. ``"TEMP"``, ``"UWND"``), or
            ``"pressure"`` for the virtual pressure variable. A name must not
            collide with an existing column of ``points``.
        time : pandas.Timestamp or str, optional
            One valid time for every point, used when ``points`` has no
            ``time`` column. Passing both raises ``ValueError``. When neither
            is given, the file must contain exactly one time, which is used.
        z_kind : {"pressure", "native", "agl", "msl"}, default "pressure"
            Vertical coordinate system of the ``z`` values:

            - ``"native"``: fractional ARL level index.
            - ``"pressure"``: hPa. Sigma/hybrid files (flag 1/4) need
              ``PRSS``; pressure files (flag 2) use the stored levels;
              terrain-following files (flag 3) raise ``ValueError``.
            - ``"agl"``: metres above ground level. Sigma/hybrid files
              integrate hypsometrically from ``PRSS`` and ``TEMP``; pressure
              files use ``HGTS - SHGT``; terrain-following files use the
              stored levels.
            - ``"msl"``: metres above mean sea level. Sigma/hybrid files add
              ``SHGT`` to the hypsometric AGL height; pressure files use
              ``HGTS``; terrain-following files add ``SHGT`` to the stored
              levels.

            Each vertical coordinate system has exactly one method, as in
            HYSPLIT; there is no fallback between them.
        method : {"linear", "nearest"}, default "linear"
            Horizontal interpolation: bilinear or nearest grid point.
        earth_relative : bool, default False
            Rotate sampled wind pairs (``UWND``/``VWND``, ``U10M``/``V10M``)
            from the grid axes to east/north. Winds in ARL files on projected
            grids are stored grid-relative, as HYSPLIT expects. Both components
            of a pair must be requested. No effect on lat/lon grids.

        Returns
        -------
        pandas.DataFrame
            Copy of ``points`` (all columns and the index preserved) with one
            added column per requested variable. Points outside the grid or
            the vertical range are NaN.

        Raises
        ------
        ValueError
            If a required column is missing, ``time`` is given alongside a
            ``time`` column, no time is given for a multi-time file, a point
            time is not in the file, a variable name collides with a column
            of ``points``, ``z_kind`` or ``method`` is invalid, or a field
            that ``z_kind`` requires is missing.

        Examples
        --------
        >>> import pandas as pd
        >>> import arlmet
        >>> pts = pd.DataFrame({"lon": [-111.9], "lat": [40.7], "z": [850.0]})
        >>> with arlmet.File("met.arl") as met:
        ...     met.sample_points(pts, ["UWND", "VWND"], time="2024-07-18 00:00")
        """
        # Delayed import: ops sit on top of file, so file's use of the sampling
        # op is lazy to avoid a file <-> ops import cycle (see File.extract_subset).
        from arlmet.ops.sample import sample_points

        return sample_points(
            self,
            points,
            variables,
            time=time,
            z_kind=z_kind,
            method=method,
            earth_relative=earth_relative,
        )

    def to_dataset(
        self,
        *,
        drop_variables: Sequence[str] | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        levels: Iterable[int] | None = None,
    ) -> xr.Dataset:
        """
        Project this file into the simplified analysis Dataset representation.

        Parameters
        ----------
        drop_variables : sequence of str, optional
            Variable names to omit.
        bbox : tuple[float, float, float, float], optional
            Geographic bounding box ``(west, south, east, north)`` in degrees.
        levels : iterable of int, optional
            ARL level indices to keep.

        Returns
        -------
        xarray.Dataset
            See :func:`arlmet.open_dataset` for the layout.
        """
        from arlmet.xarray.dataset import _build_dataset_from_file

        return _build_dataset_from_file(
            self,
            drop_variables=drop_variables,
            bbox=bbox,
            levels=levels,
        )

    def extract_subset(
        self,
        dest: str | os.PathLike[str],
        *,
        bbox: tuple[float, float, float, float] | None = None,
        levels: Iterable[int] | None = None,
        variables: Iterable[str] | None = None,
    ) -> Path:
        """
        Write a spatial/vertical subset of this file to a new ARL file.

        Equivalent to :func:`arlmet.extract_subset` with this file's path as
        the input.

        Parameters
        ----------
        dest : path-like
            Output ARL file. Overwrites any existing file. Must not be this
            file.
        bbox : tuple[float, float, float, float], optional
            Geographic bounding box ``(west, south, east, north)`` in degrees.
        levels : iterable of int, optional
            ARL level indices to keep. Output levels are compacted and
            renumbered from zero while preserving the selected level heights.
        variables : iterable of str, optional
            Variable names to keep. All variables are included by default.

        Returns
        -------
        pathlib.Path
            The output path, ``Path(dest)``.

        Examples
        --------
        >>> import arlmet
        >>> with arlmet.File("met.arl") as met:
        ...     out = met.extract_subset("subset.arl", bbox=(-114, 39, -110, 42))
        >>> ds = arlmet.open_dataset(out)
        """
        from arlmet.ops.subset import extract_subset

        return extract_subset(
            self.path,
            dest,
            bbox=bbox,
            levels=levels,
            variables=variables,
        )

    def __getitem__(self, key: str | int | pd.Timestamp) -> RecordSet:
        if isinstance(key, str):
            # Allow lookup by string time representation
            key = ensure_timestamp(key)
        elif isinstance(key, int):
            # Positional lookup follows the sorted `times`, not file order
            key = self.times[key]
        return self._recordsets[key]

    def __iter__(self) -> Iterator[pd.Timestamp]:
        return iter(self.times)

    def __len__(self) -> int:
        return len(self._recordsets)

    def __contains__(self, key: object) -> bool:
        try:
            ts = ensure_timestamp(key)
        except (TypeError, ValueError, OverflowError):
            # Not interpretable as a timestamp
            return False
        return ts in self._recordsets

    @override
    def __repr__(self) -> str:
        grid_str = (
            f"{self._grid.nx}\u00d7{self._grid.ny}"
            if self._grid is not None
            else "None"
        )
        levels_str = str(len(self._vaxis.levels)) if self._vaxis is not None else "None"
        return (
            f"File({self.path.name!r}, mode={self.mode!r}, "
            f"times={len(self)}, grid={grid_str}, levels={levels_str})"
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
