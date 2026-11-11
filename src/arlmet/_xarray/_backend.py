"""Lazy xarray BackendArray for ARL variable data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from xarray.backends.common import BackendArray
from xarray.core import indexing

if TYPE_CHECKING:
    from arlmet._grid import GridWindow
    from arlmet.record import DataRecord

__all__ = ["ArlVariableArray"]


class ArlVariableArray(BackendArray):
    """Backend-style lazy array for a single ARL variable."""

    def __init__(
        self,
        *,
        records: dict[tuple[int, int], DataRecord],
        shape: tuple[int, int, int, int],
        window: GridWindow | None = None,
    ):
        self.records = records
        self.shape = shape
        self.dtype = np.dtype(np.float32)
        self.window = window

    def __getitem__(self, key: indexing.ExplicitIndexer) -> npt.NDArray[Any]:
        # OUTER, not OUTER_1VECTOR: _getitem takes an array on every axis, so
        # xarray need not index (and copy) the result a second time.
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.OUTER,
            self._getitem,
        )

    def _getitem(self, key: tuple[Any, ...]) -> npt.NDArray[Any]:
        """Materialize the requested outer-indexed slice from the backing records."""
        if len(key) != 4:
            raise IndexError(f"ARL variable arrays expect 4 indexers, got {len(key)}.")

        t_idx, t_scalar = _normalize_backend_indexer(key[0], self.shape[0])
        z_idx, z_scalar = _normalize_backend_indexer(key[1], self.shape[1])
        y_idx, y_scalar = _normalize_backend_indexer(key[2], self.shape[2])
        x_idx, x_scalar = _normalize_backend_indexer(key[3], self.shape[3])
        # Slices keep the field's rows and columns a view; arrays copy them.
        y_key = _field_key(key[2])
        x_key = _field_key(key[3])

        out = np.empty(
            (len(t_idx), len(z_idx), len(y_idx), len(x_idx)), dtype=np.float32
        )
        for time_pos, time_idx in enumerate(t_idx):
            for level_pos, level_idx in enumerate(z_idx):
                record = self.records.get((int(time_idx), int(level_idx)))
                if record is None:
                    out[time_pos, level_pos] = np.nan
                    continue
                field = record.read(window=self.window)
                out[time_pos, level_pos] = field[y_key][:, x_key]

        scalars = (t_scalar, z_scalar, y_scalar, x_scalar)
        return np.asarray(
            out[tuple(0 if is_scalar else slice(None) for is_scalar in scalars)]
        )


def _field_key(indexer: slice | npt.NDArray[Any] | int) -> slice | npt.NDArray[Any]:
    """One axis of a 2D field's indexer, as a slice or array that keeps the axis."""
    if isinstance(indexer, slice | np.ndarray):
        return indexer
    return slice(int(indexer), int(indexer) + 1)


def _normalize_backend_indexer(
    indexer: slice | npt.NDArray[Any] | int, size: int
) -> tuple[npt.NDArray[Any], bool]:
    """Normalize one backend indexer to explicit integer indices and scalar status."""
    base = np.arange(size, dtype=int)
    if isinstance(indexer, slice):
        return np.asarray(base[indexer], dtype=int), False
    if isinstance(indexer, np.ndarray):
        return np.asarray(base[indexer], dtype=int).reshape(-1), False
    return np.asarray([base[indexer]], dtype=int), True
