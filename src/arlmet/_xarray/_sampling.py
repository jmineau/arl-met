"""
A Dataset in :func:`arlmet.open_dataset`'s layout as an input of :func:`arlmet.sample_points`.

:func:`arlmet.sample_points` reads an ARL file one time step at a time, through
its :class:`~arlmet.RecordSet`. :class:`DatasetMet` gives the same view of a
Dataset: the view of the file :func:`arlmet.write_dataset` would write from it.
That file has the vertical axis ``write_dataset`` writes, its upper-air levels
are numbered 1..N in the order of the ``level`` coordinate, and a variable is
stored at a level when it holds a value there at any time step, as
``write_dataset`` gives such a level a record at every time step (flagged
missing, so read as NaN, where its slice is all NaN).
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr

from arlmet._grid import Grid, GridWindow, _split_at_seam
from arlmet._time import ensure_timestamp, utc_times
from arlmet.vertical import VerticalAxis

from ._coords import (
    _check_vertical_axis_agrees,
    _extract_dataset_vertical_axis,
    grid_from_coord,
)
from .dataset import _check_grid_coords

__all__ = ["DatasetMet"]


class _DatasetField:
    """One 2D slice of a Dataset variable, read as a :class:`~arlmet.DataRecord` is."""

    def __init__(self, values: xr.DataArray, grid: Grid) -> None:
        self._values = values
        self._grid = grid

    def read(self, window: GridWindow | None = None) -> npt.NDArray[np.float32]:
        """Return the field, or the part in *window* (across a global grid's seam too), as float32."""
        if window is None:
            return np.asarray(self._values.values, dtype=np.float32)
        y_dim, x_dim = self._grid.dims
        pieces = [
            np.asarray(
                self._values.isel(
                    {
                        y_dim: slice(part.y_start, part.y_stop),
                        x_dim: slice(part.x_start, part.x_stop),
                    }
                ).values,
                dtype=np.float32,
            )
            for part in _split_at_seam(window, self._grid)
        ]
        return pieces[0] if len(pieces) == 1 else np.hstack(pieces)


class _DatasetTimeStep:
    """The fields of one time of a :class:`DatasetMet`, as a :class:`~arlmet.RecordSet` gives them."""

    def __init__(self, met: DatasetMet, index: int) -> None:
        self._met = met
        self._index = index

    @property
    def time(self) -> pd.Timestamp:
        """The valid time."""
        return self._met.times[self._index]

    @property
    def source(self) -> str:
        """The Dataset's ARL source ID."""
        return self._met.source

    @property
    def grid(self) -> Grid:
        """The Dataset's grid."""
        return self._met.grid

    @property
    def vertical_axis(self) -> VerticalAxis:
        """The vertical axis ``write_dataset`` would write."""
        return self._met.vertical_axis

    @property
    def variables(self) -> set[str]:
        """The names of the variables the Dataset holds."""
        return {name for _, name in self._met._fields}

    def _slice(self, key: tuple[int, str]) -> xr.DataArray:
        variable, position = self._met._fields[key]
        if position is None:
            return variable.isel(time=self._index)
        return variable.isel(time=self._index, level=position)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, tuple) or len(key) != 2:
            return False
        try:
            if key not in self._met._fields:
                return False
        except TypeError:  # unhashable key parts
            return False
        return self._met._is_stored(cast("tuple[int, str]", key))

    def __getitem__(self, key: tuple[int, str]) -> _DatasetField:
        if key not in self:
            raise KeyError(f"No field {key!r} at time {self.time}.")
        return _DatasetField(self._slice(key), self._met.grid)


class DatasetMet:
    """
    A Dataset in :func:`arlmet.open_dataset`'s layout, sampled as the ARL file :func:`arlmet.write_dataset` would make from it.

    Parameters
    ----------
    ds : xarray.Dataset
        A Dataset as :func:`arlmet.open_dataset` and the converters return
        it: ``attrs["source"]``, an ``arl_grid`` coordinate, a ``time``
        coordinate, a ``level`` dimension with the vertical axis in its
        coordinates, and each variable on ``(time, level, <grid dims>)`` or
        ``(time, <grid dims>)``.

    Raises
    ------
    ValueError
        If *ds* does not have that layout: it has no source, grid, time or
        levels (a surface-only Dataset has no vertical axis to sample on), its
        size differs from its grid's (a Dataset cut with ``sel``), its rows or
        columns do not increase, its coordinates disagree with its grid, or a
        variable lacks the time or grid dimensions.
    """

    def __init__(self, ds: xr.Dataset) -> None:
        source = ds.attrs.get("source")
        if not isinstance(source, str) or not source:
            raise ValueError(
                "A Dataset to sample needs attrs['source'], the ARL source ID, as "
                "arlmet.open_dataset() and the converters set it."
            )
        if "arl_grid" not in ds.coords:
            raise ValueError(
                "A Dataset to sample needs the 'arl_grid' coordinate, as "
                "arlmet.open_dataset() and the converters give it."
            )
        if "time" not in ds.coords:
            raise ValueError("A Dataset to sample needs a 'time' coordinate.")
        if "time" not in ds.dims:
            raise ValueError(
                "A Dataset to sample needs a 'time' dimension; a Dataset cut "
                "with isel(time=0) has a scalar time: use isel(time=[0])."
            )
        if "level" not in ds.coords:
            raise ValueError(
                "A Dataset to sample needs upper-air levels: a surface-only "
                "Dataset has no vertical axis to sample on."
            )
        self.source = source
        self.grid = grid_from_coord(ds.coords["arl_grid"])
        self.vertical_axis = _extract_dataset_vertical_axis(
            ds,
            fix="open the file with arlmet.open_dataset() or use a converter's output",
        )
        _check_vertical_axis_agrees(ds, self.vertical_axis)
        h_dims = self.grid.dims
        for dim, size in zip(h_dims, (self.grid.ny, self.grid.nx), strict=True):
            if ds.sizes.get(dim) != size:
                raise ValueError(
                    f"The Dataset's '{dim}' dimension has {ds.sizes.get(dim)} cells, "
                    f"but its 'arl_grid' coordinate describes {size}: a Dataset cut "
                    "with isel or sel keeps the whole grid's description. Crop the "
                    "file with open_dataset(bbox=...), or a converter's input before "
                    "converting."
                )
            if dim in ds.coords and not (np.diff(ds.coords[dim].values) > 0).all():
                raise ValueError(
                    f"The Dataset's '{dim}' coordinate must increase, as the "
                    f"grid's does; sort it first, e.g. ds.sortby('{dim}')."
                )
        # After the order check, so a descending coordinate gets its own message.
        _check_grid_coords(ds, self.grid)
        self.times = [
            ensure_timestamp(t) for t in utc_times(ds.coords["time"].to_index())
        ]
        self._fields: dict[tuple[int, str], tuple[xr.DataArray, int | None]] = {}
        for name in ds.data_vars:
            name = str(name)
            variable = ds[name]
            missing = [d for d in ("time", *h_dims) if d not in variable.dims]
            if missing:
                raise ValueError(
                    f"Variable '{name}' lacks the dimension(s) {missing}; a Dataset "
                    f"to sample holds variables on (time, [level,] {', '.join(h_dims)})."
                )
            if "level" in variable.dims:
                variable = variable.transpose("time", "level", *h_dims)
                # Numbered 1..N in coordinate order, as write_dataset numbers them.
                for position in range(ds.sizes["level"]):
                    self._fields[(position + 1, name)] = (variable, position)
            else:
                self._fields[(0, name)] = (variable.transpose("time", *h_dims), None)
        self._index = {time: i for i, time in enumerate(self.times)}
        self._stored: dict[tuple[int, str], bool] = {}

    def _is_stored(self, key: tuple[int, str]) -> bool:
        """
        Whether the written file would hold a record for *key*.

        ``write_dataset`` decides it across all time steps: a (level,
        variable) with a value at any time step gets a record at every one.
        """
        if key not in self._stored:
            variable, position = self._fields[key]
            if position is not None:
                variable = variable.isel(level=position)
            # One time step at a time, stopping at the first with a value: a
            # lazy Dataset then reads one slice of a stored field, not all of it.
            self._stored[key] = any(
                bool(variable.isel(time=i).notnull().any())
                for i in range(variable.sizes["time"])
            )
        return self._stored[key]

    def __getitem__(self, time: Any) -> _DatasetTimeStep:
        return _DatasetTimeStep(self, self._index[ensure_timestamp(time)])
