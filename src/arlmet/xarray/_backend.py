from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from xarray.backends.common import BackendArray
from xarray.core import indexing

if TYPE_CHECKING:
    from arlmet.grid import GridWindow
    from arlmet.record import DataRecord


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

    def __getitem__(self, key: indexing.ExplicitIndexer) -> np.ndarray:
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.OUTER_1VECTOR,
            self._getitem,
        )

    def _getitem(self, key: tuple[Any, ...]) -> np.ndarray:
        """Materialize the requested outer-indexed slice from the backing records."""
        if len(key) != 4:
            raise IndexError(f"ARL variable arrays expect 4 indexers, got {len(key)}.")

        t_idx, t_scalar = _normalize_backend_indexer(key[0], self.shape[0])
        z_idx, z_scalar = _normalize_backend_indexer(key[1], self.shape[1])
        y_idx, y_scalar = _normalize_backend_indexer(key[2], self.shape[2])
        x_idx, x_scalar = _normalize_backend_indexer(key[3], self.shape[3])

        out = np.full(
            (len(t_idx), len(z_idx), len(y_idx), len(x_idx)),
            np.nan,
            dtype=np.float32,
        )
        for time_pos, time_idx in enumerate(t_idx):
            for level_pos, level_idx in enumerate(z_idx):
                record = self.records.get((int(time_idx), int(level_idx)))
                if record is None:
                    continue
                field = record.read(window=self.window)
                out[time_pos, level_pos] = field[np.ix_(y_idx, x_idx)]

        for axis, is_scalar in reversed(
            list(enumerate((t_scalar, z_scalar, y_scalar, x_scalar)))
        ):
            if is_scalar:
                out = np.take(out, 0, axis=axis)
        return out


def _normalize_backend_indexer(
    indexer: slice | np.ndarray | int, size: int
) -> tuple[np.ndarray, bool]:
    """Normalize one backend indexer to explicit integer indices and scalar status."""
    base = np.arange(size, dtype=int)
    if isinstance(indexer, slice):
        return np.asarray(base[indexer], dtype=int), False
    if isinstance(indexer, np.ndarray):
        return np.asarray(base[indexer], dtype=int).reshape(-1), False
    return np.asarray([base[indexer]], dtype=int), True
