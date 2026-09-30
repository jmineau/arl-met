"""Register arlmet as an xarray backend: ``xr.open_dataset(path, engine="arl")``."""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

import xarray as xr
from typing_extensions import override
from xarray.backends import BackendEntrypoint

# Bytes 14-18 of an ARL file's first record header hold its variable name, and
# every ARL file starts with an index record, whose variable is "INDX".
_INDEX_MAGIC = b"INDX"
_INDEX_MAGIC_SLICE = slice(14, 18)


class ARLBackendEntrypoint(BackendEntrypoint):
    """
    xarray backend for NOAA ARL packed meteorology files.

    Lets xarray open ARL files directly::

        ds = xr.open_dataset("met.arl", engine="arl")
        ds = xr.open_dataset("met.arl", engine="arl", bbox=(-114, 39, -110, 42))

    The result matches :func:`arlmet.open_dataset`, with the same ``bbox`` and
    ``levels`` options. Going through xarray adds its generic options, such as
    ``chunks=`` for dask. Without ``engine=``, xarray recognizes ARL files by
    their first record, so ``xr.open_dataset("met.arl")`` also works when no
    other backend claims the file first.
    """

    description = "Open NOAA ARL packed meteorology files (HYSPLIT/STILT input)"
    url = "https://jmineau.github.io/arl-met/"
    open_dataset_parameters = ("filename_or_obj", "drop_variables", "bbox", "levels")

    @override
    def open_dataset(  # type: ignore[override]
        self,
        filename_or_obj: str | os.PathLike[Any],
        *,
        drop_variables: str | Iterable[str] | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        levels: list[int] | tuple[int, ...] | None = None,
    ) -> xr.Dataset:
        """Open an ARL file; see :func:`arlmet.open_dataset`."""
        from .dataset import open_dataset

        if not isinstance(filename_or_obj, (str, os.PathLike)):
            raise TypeError(
                "The arl engine can only open files by path, "
                f"not {type(filename_or_obj).__name__}."
            )
        if isinstance(drop_variables, str):
            drop_variables = [drop_variables]
        return open_dataset(
            filename_or_obj,
            drop_variables=None if drop_variables is None else list(drop_variables),
            bbox=bbox,
            levels=levels,
        )

    @override
    def guess_can_open(self, filename_or_obj: Any) -> bool:
        """Return True if ``filename_or_obj`` is a path to a file starting with an ARL index record."""
        if not isinstance(filename_or_obj, (str, os.PathLike)):
            return False
        try:
            with open(filename_or_obj, "rb") as handle:
                head = handle.read(_INDEX_MAGIC_SLICE.stop)
        except (OSError, TypeError, ValueError):
            return False
        return head[_INDEX_MAGIC_SLICE] == _INDEX_MAGIC
