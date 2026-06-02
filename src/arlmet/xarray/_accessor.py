"""xarray accessor for ARL-specific dataset metadata."""

from __future__ import annotations

import xarray as xr

from ._coords import grid_from_coord, vaxis_from_coord


@xr.register_dataset_accessor("arl")
class ARLDatasetAccessor:
    """
    xarray accessor exposing ARL-specific metadata on a Dataset.

    Access via ``ds.arl``.

    Attributes
    ----------
    grid : Grid
        Horizontal grid and projection, reconstructed from the ``arl_grid`` coordinate.
    vertical_axis : VerticalAxis or None
        Vertical coordinate metadata, reconstructed from the ``level`` coordinate
        attrs. Returns ``None`` when the dataset has no ``level`` dimension
        (e.g. a surface-only dataset).
    source : str
        ARL source identifier from ``ds.attrs["source"]``.
    """

    def __init__(self, xarray_obj: xr.Dataset):
        self._obj = xarray_obj

    @property
    def grid(self):
        if "arl_grid" not in self._obj.coords:
            raise AttributeError(
                "Dataset has no 'arl_grid' coordinate. Open files with arlmet.open_dataset()."
            )
        return grid_from_coord(self._obj.coords["arl_grid"])

    @property
    def vertical_axis(self):
        # Dataset case: integer 'level' coord + physical non-dim coord with "surface" attr
        flag = self._obj.attrs.get("vertical_flag")
        if flag is None or "level" not in self._obj.coords:
            return None
        level_coord = self._obj.coords["level"]
        for _cname, cvar in self._obj.coords.items():
            if cvar.dims == ("level",) and "surface" in cvar.attrs:
                return vaxis_from_coord(int(flag), level_coord, cvar)
        return None

    @property
    def source(self) -> str:
        return str(self._obj.attrs.get("source", ""))
