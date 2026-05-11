
from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt

from arlmet.grid import Grid
from arlmet.record import DataRecord
from arlmet.vertical import VerticalAxis


class RecordCollection(Protocol):
    """
    Structural interface for objects exposing ARL data records.

    Attributes
    ----------
    mode : Literal["r", "w"]
        Access mode of the collection, either 'r' for read-only or 'w' for writable.
    source : str
        ARL source identifier.
    grid : Grid
        Horizontal grid metadata associated with the records.
    vertical_axis : VerticalAxis
        Vertical coordinate metadata associated with the records.
    record_length : int
        Record length in bytes for the collection, derived from the grid.
    records : list[DataRecord]
        Materialized list of records in insertion order.
    """

    @property
    def mode(self) -> Literal["r", "w"]: ...

    @property
    def source(self) -> str: ...

    @property
    def grid(self) -> Grid: ...

    @property
    def vertical_axis(self) -> VerticalAxis: ...

    @property
    def record_length(self) -> int: ...

    @property
    def records(self) -> list[DataRecord]: ...

    @property
    def variables(self) -> VariableAccessor: ...


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

        self._data: npt.NDArray | None = None

    @property
    def data(self) -> npt.NDArray:
        """The array representing the full variable view, always as 4D."""
        if self._data is not None:
            return self._data
        self._data = self._build_array()
        return self._data

    def _build_array(self) -> npt.NDArray:
        """Read all records for this variable into a (time, level, y, x) array."""
        records = [r for r in self.source.records if r.variable == self.name]
        if not records:
            raise KeyError(f"Variable '{self.name}' not found in source.")

        grid = self.source.grid
        if self._is_file_view:
            times = self.source.times  # type: ignore[attr-defined]
            time_index = {t: i for i, t in enumerate(times)}
        else:
            times = [self.source.time]  # type: ignore[attr-defined]
            time_index = {times[0]: 0}

        levels = sorted({r.level for r in records})
        level_index = {lv: i for i, lv in enumerate(levels)}

        out = np.full(
            (len(times), len(levels), grid.ny, grid.nx),
            fill_value=np.nan,
            dtype=np.float32,
        )
        for r in records:
            t_idx = time_index[r.time]
            l_idx = level_index[r.level]
            out[t_idx, l_idx] = r.read()
        return out

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
