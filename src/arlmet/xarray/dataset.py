"""Open and write ARL files as xarray Datasets."""

from __future__ import annotations

import os
import warnings
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr
from xarray.core import indexing

from arlmet._io import atomic_output, reject_same_file
from arlmet._time import ensure_timestamp
from arlmet.ops.sample import _mass_coupled
from arlmet.ops.subset import normalize_levels, resolve_window, select_records
from arlmet.vertical import VerticalAxis

from ._backend import ArlVariableArray
from ._coords import (
    _extract_dataset_forecast_hours,
    _extract_dataset_vertical_axis,
    add_cf_spatial_attrs,
    grid_from_coord,
    grid_to_coord,
    physical_coord_for_vaxis,
)

if TYPE_CHECKING:
    from arlmet.file import File
    from arlmet.record import DataRecord

__all__ = ["open_dataset", "write_dataset"]

#: What each wind of a mass-coupled WRF file holds (``AWRF`` with ``MUU0`` and
#: ``MUV0``), as its ``description`` attribute. The Dataset keeps the stored
#: values; HYSPLIT (``metinp.f``) and :func:`arlmet.sample_points` decouple them.
_MASS_COUPLED_WINDS = {
    "UWND": (
        "WRF's time-averaged x-wind coupled to the column mass, as stored. HYSPLIT "
        "uses UWND * MSFU / MUU0 (m s-1)."
    ),
    "VWND": (
        "WRF's time-averaged y-wind coupled to the column mass, as stored. HYSPLIT "
        "uses VWND * MSFV / MUV0 (m s-1)."
    ),
    "WWND": (
        "WRF's time-averaged vertical motion coupled to the column mass, as stored "
        "(DIFW added). HYSPLIT uses WWND * MSFT / 100."
    ),
}


def _build_dataset_from_file(
    met: File,
    *,
    drop_variables: Sequence[str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    levels: Iterable[int] | None = None,
) -> xr.Dataset:
    """Build an xarray Dataset from an open ARL File handle."""
    drop_variables_set: set[str] = set(drop_variables or [])
    window = resolve_window(met, bbox)
    read_window = None if bbox is None else window
    selected_grid = met.grid if bbox is None else met.grid.subset(window)
    requested_levels = (
        None if levels is None else normalize_levels(met.vertical_axis, levels)
    )
    requested_level_set = None if requested_levels is None else set(requested_levels)

    selected_recordsets: list[tuple[pd.Timestamp, int | None, list[DataRecord]]] = []
    present_levels: dict[int, None] = {}

    for time in met.times:
        recordset = met[time]
        selected_records = [
            record
            for record in select_records(recordset.records, levels=requested_level_set)
            if record.variable not in drop_variables_set
        ]
        if not selected_records:
            continue

        for record in selected_records:
            present_levels.setdefault(record.level, None)
        selected_recordsets.append(
            (ensure_timestamp(time), recordset.forecast, selected_records)
        )

    selected_levels = (
        list(requested_levels) if requested_levels is not None else list(present_levels)
    )
    level_pos = {level: pos for pos, level in enumerate(selected_levels)}

    ds = xr.Dataset(coords=selected_grid.calculate_coords())
    ds = add_cf_spatial_attrs(ds)
    # Integer 1-N level coord; surface (index 0) is handled separately below
    ds = ds.assign_coords(
        level=("level", list(selected_levels), {"standard_name": "model_level_number"})
    )

    times: list[pd.Timestamp] = []
    index_forecasts: list[int] = []
    records_by_variable: dict[str, dict[tuple[int, int], Any]] = defaultdict(dict)

    for time_pos, (time, _index_forecast, selected_records) in enumerate(
        selected_recordsets
    ):
        times.append(time)
        index_forecasts.append(0 if _index_forecast is None else int(_index_forecast))
        for record in selected_records:
            records_by_variable[record.variable][
                (time_pos, level_pos[record.level])
            ] = record

    if times:
        ds = ds.assign_coords(
            time=xr.Variable(
                "time", times, attrs={"standard_name": "time", "axis": "T"}
            )
        )
        ds = ds.assign_coords(
            forecast_hour=xr.Variable(
                "time",
                [int(value) for value in index_forecasts],
                attrs={
                    "long_name": "ARL index record forecast hour",
                    "description": (
                        "Forecast hour stored in the INDX record for each time step. "
                        "Individual variable record forecasts may differ and are not "
                        "represented in this Dataset; arlmet.File reads them."
                    ),
                    "units": "hours",
                },
            )
        )

    shape = (len(times), len(selected_levels), selected_grid.ny, selected_grid.nx)
    dims = ("time", "level", *selected_grid.dims)
    for name in sorted(records_by_variable):
        backend = ArlVariableArray(
            records=records_by_variable[name],
            shape=shape,
            window=read_window,
        )
        ds[name] = xr.Variable(dims, indexing.LazilyIndexedArray(backend))
    if met.times and _mass_coupled(met[met.times[0]]):
        for name, description in _MASS_COUPLED_WINDS.items():
            if name in ds:
                ds[name].attrs["description"] = description

    sfc_pos = level_pos.get(0)
    if sfc_pos is not None:
        sfc_only_vars = {
            name
            for name, recs in records_by_variable.items()
            if all(z == sfc_pos for (_, z) in recs)
        }
        for name in sfc_only_vars:
            ds[name] = ds[name].isel(level=sfc_pos).drop_vars("level")

        upper_level_indices = [i for i in range(len(selected_levels)) if i != sfc_pos]
        if upper_level_indices:
            ds = ds.isel(level=upper_level_indices)
        else:
            ds = ds.drop_dims("level")

    ds = ds.assign_coords(arl_grid=grid_to_coord(selected_grid))
    ds.attrs["source"] = met.source
    ds.attrs["vertical_flag"] = met.vertical_axis.flag
    if "level" in ds.coords:
        # Add physical non-dim coord for upper-air levels
        upper_level_ints = [int(v) for v in ds.coords["level"].values.tolist()]
        try:
            phys_name, phys_var = physical_coord_for_vaxis(
                met.vertical_axis, upper_level_ints
            )
            ds = ds.assign_coords({phys_name: phys_var})
        except NotImplementedError:
            warnings.warn(
                f"Physical coordinate for vertical_flag={met.vertical_axis.flag} is not implemented; "
                "the 'level' coordinate will contain raw ARL level indices only.",
                stacklevel=2,
            )
    return ds


def _read_slice(
    transposed: xr.DataArray, time_index: int, slot: int | None
) -> np.ndarray:
    """Read one 2D slice of a variable as float32 (``slot`` on ``level``)."""
    selected = (
        transposed.isel(time=time_index)
        if slot is None
        else transposed.isel(time=time_index, level=slot)
    )
    return np.asarray(selected.values, dtype=np.float32)


def open_dataset(
    filename_or_obj: str | os.PathLike[str],
    *,
    drop_variables: Sequence[str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    levels: Iterable[int] | None = None,
) -> xr.Dataset:
    """
    Open an ARL meteorology file as an xarray Dataset.

    Surface-only variables (e.g. ``SHGT``, ``T02M``, ``PRSS``) have dimensions
    ``(time, lat, lon)`` with no ``level`` dimension. Upper-air variables
    (e.g. ``UWND``, ``VWND``, ``TEMP``) have dimensions
    ``(time, level, lat, lon)``. An upper-air variable stored on only some
    levels (or times) is NaN where the file has no record for it, and so is a
    record flagged missing.

    The ``level`` coordinate holds the file's ARL level indices (surface is 0,
    so upper-air levels start at 1). With ``levels=`` the kept levels keep their
    original indices, which may therefore have gaps.

    The ``forecast_hour(time)`` coordinate holds each time step's forecast
    hour, the one its index record stores. A data record's own forecast hour
    can differ (an accumulated field taken from a 6-hour forecast may say 6
    where the index record says 0), but it is not represented here: HYSPLIT
    reads a data record's forecast hour only as the missing flag, and the hour
    it uses is the index record's. :attr:`arlmet.DataRecord.forecast` reads a
    record's own hour.

    Values are as the file stores them. On a mass-coupled WRF file (source
    ``AWRF`` with ``MUU0`` and ``MUV0``, the time-averaged winds of
    ``arw2arl``) ``UWND``, ``VWND`` and ``WWND`` are coupled to the column
    mass, and their ``description`` attribute gives the conversion HYSPLIT
    makes (``UWND * MSFU / MUU0``, ...); :func:`arlmet.sample_points` returns
    the converted winds.

    ARL metadata is accessible via the ``.arl`` accessor::

        ds = arlmet.open_dataset("met.arl")
        ds.arl.grid  # Grid object (survives isel/sel)
        ds.arl.vertical_axis  # VerticalAxis
        ds.isel(level=0)  # selects level 0 for upper vars; sfc vars unchanged

    Parameters
    ----------
    filename_or_obj : path-like
        Path to the ARL file.
    drop_variables : sequence of str, optional
        Variable names to omit from the resulting dataset.
    bbox : tuple[float, float, float, float], optional
        Geographic bounding box ``(west, south, east, north)`` in degrees.
        On a projected grid the crop is the smallest rectangle of cells
        that holds the box's part on the grid (:meth:`arlmet.Grid.window_from_bbox`).
    levels : iterable of int, optional
        ARL level indices to keep.

    Returns
    -------
    xarray.Dataset
    """
    from arlmet.file import File

    with File(filename_or_obj) as met:
        ds = met.to_dataset(
            drop_variables=drop_variables,
            bbox=bbox,
            levels=levels,
        )
    # xarray convention: remember where the data came from (write_dataset uses
    # it to refuse overwriting the file a lazy Dataset still reads from).
    ds.encoding["source"] = os.fspath(filename_or_obj)
    return ds


def write_dataset(
    ds: xr.Dataset,
    filename_or_obj: str | os.PathLike[str],
    *,
    vertical_axis: VerticalAxis | None = None,
) -> None:
    """
    Write a simple flat Dataset representation to an ARL file.

    This writer supports the common-case Dataset contract returned by
    ``open_dataset()``:

    - surface variables use dims ``(time, y, x)`` or ``(time, lat, lon)``
    - upper-air variables use dims ``(time, level, y, x)`` or ``(time, level, lat, lon)``
    - all upper-air variables share the same ``level`` coordinate

    Upper-air levels are written in ``level`` coordinate order and renumbered
    ``1..N`` (surface is 0), so a Dataset holding a subset of a file's levels
    (e.g. from ``open_dataset(levels=...)`` or ``ds.sel(level=...)``) is
    written as a compact file, like :func:`arlmet.extract_subset` does.

    Every (variable, level) that holds data at some time step is written at
    every time step, so each time step has the same records. Where its slice
    at a time step is entirely NaN (as a record flagged missing in the source
    file reads), the record is written flagged missing, as HYSPLIT's writer
    leaves a record it never fills (``pakini.f``: forecast hour -1, no values),
    and HYSPLIT skips it. A (variable, level) that is NaN at every time step
    gets no record: that is how :func:`open_dataset` represents a variable
    stored on only some levels. Finding which slices hold data takes one pass
    over the values before anything is written, so a lazy Dataset (as
    :func:`open_dataset` returns) reads each slice twice.

    The ``forecast_hour(time)`` coordinate gives each time step's index-record
    forecast hour, and every record of that time step is written with it (0
    when the coordinate is absent). A file whose records carry their own
    forecast hours is rewritten with the time step's hour in each; HYSPLIT
    reads a data record's forecast hour only as the missing flag, so this does
    not change what it reads. To write per-record hours, use
    :meth:`arlmet.RecordSet.create_datarecord`.

    A variable whose ``attrs["diff"]`` names a ``DIF*`` record is written with
    that DIF record after it on each level (see
    :meth:`arlmet.RecordSet.create_datarecord`). The name must be one
    HYSPLIT's name table pairs with the variable: ``DIFW`` for ``WWND`` or
    ``DZDT`` and ``DIFT`` for ``TKEN`` or ``TKEU`` above the surface, any
    ``DIF*`` name for ``TPP*`` or ``PRT*`` at the surface.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset in the layout returned by :func:`open_dataset`.
    filename_or_obj : path-like
        Output ARL file path. Overwrites any existing file, but must not be the
        file ``ds`` was opened from.
    vertical_axis : VerticalAxis, optional
        Vertical axis to write. By default it is rebuilt from the Dataset's
        ``level`` coordinates. Required for surface-only Datasets; otherwise it
        must have one level per ``level`` coordinate value plus the surface.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the Dataset does not match the layout above, a slice is partly NaN
        or non-finite, an ``attrs["diff"]`` name is not one the name table
        pairs with its variable, or ``filename_or_obj`` is the file ``ds`` was
        opened from.

    Notes
    -----
    The file is written to a temporary file next to ``filename_or_obj`` and
    renamed into place once complete, so an interrupted write never leaves a
    truncated file under the final name.
    """
    from arlmet.file import File

    if not isinstance(ds, xr.Dataset):
        raise TypeError("write_dataset() requires an xarray.Dataset.")
    if "source" in ds.encoding:
        reject_same_file(ds.encoding["source"], filename_or_obj)

    if "time" not in ds.coords:
        raise ValueError("Dataset must define a 'time' coordinate.")
    times = pd.to_datetime(ds.coords["time"].values)

    source = ds.attrs.get("source")
    if not isinstance(source, str) or not source:
        raise ValueError("Dataset is missing attrs['source'].")

    if "arl_grid" not in ds.coords:
        raise ValueError("Dataset is missing an 'arl_grid' coordinate.")
    grid = grid_from_coord(ds.coords["arl_grid"])
    h_dims = grid.dims

    resolved_vertical_axis = _extract_dataset_vertical_axis(ds, vertical_axis)
    n_upper_levels = ds.sizes.get("level", 0)
    if n_upper_levels and len(resolved_vertical_axis.levels) != n_upper_levels + 1:
        raise ValueError(
            f"vertical_axis has {len(resolved_vertical_axis.levels)} levels, but the "
            f"Dataset needs {n_upper_levels + 1} (the surface plus one per 'level' "
            "coordinate value)."
        )
    forecast_hours = _extract_dataset_forecast_hours(ds, times)

    data_vars = list(ds.data_vars)
    if not data_vars:
        raise ValueError("Dataset contains no data variables to write.")

    diff_names: dict[str, str] = {}
    for name in data_vars:
        da = ds[str(name)]
        diff_name = da.attrs.get("diff")
        if diff_name is None:
            continue
        if not isinstance(diff_name, str) or not diff_name:
            raise ValueError(
                f"Variable '{name}' attrs['diff'] must be a non-empty string when provided."
            )
        if not diff_name.startswith("DIF"):
            raise ValueError(
                f"Variable '{name}' attrs['diff'] must start with 'DIF', got '{diff_name}'."
            )
        diff_names[str(name)] = diff_name

    # Check each variable and list its slots: (position on the ``level``
    # dimension, ARL level index), or (None, 0) for a surface variable.
    variables: list[tuple[str, xr.DataArray, list[tuple[int | None, int]]]] = []
    for var_name in data_vars:
        var_name = str(var_name)
        if len(var_name) > 4:
            raise ValueError(
                f"Variable names must be 4 characters or fewer, got '{var_name}'."
            )
        if var_name.startswith("DIF"):
            raise ValueError(
                "Dataset DIF generation is parent-led; set attrs['diff'] on the parent variable instead of writing a DIF* data variable."
            )

        da = ds[var_name]
        for dim, size in zip(h_dims, (grid.ny, grid.nx), strict=True):
            if dim not in da.dims:
                raise ValueError(
                    f"Variable '{var_name}' is missing horizontal dimension '{dim}'."
                )
            if da.sizes[dim] != size:
                raise ValueError(
                    f"Variable '{var_name}' dimension '{dim}' has size {da.sizes[dim]}, expected {size}."
                )

        if "time" not in da.dims:
            raise ValueError(
                f"Variable '{var_name}' data must include the 'time' dimension."
            )
        data_times = pd.to_datetime(da.coords["time"].values)
        if not data_times.equals(times):
            raise ValueError(
                f"Variable '{var_name}' time coordinate must align exactly with the dataset time coordinate."
            )

        if "level" in da.dims:
            if "level" not in ds.coords:
                raise ValueError(
                    f"Variable '{var_name}' uses a 'level' dimension but the dataset has no shared level coordinate."
                )
            transposed = da.transpose("time", "level", *h_dims, missing_dims="raise")
            # Renumber upper-air levels 1..N in coordinate order.
            slots: list[tuple[int | None, int]] = [
                (slot, slot + 1) for slot in range(ds.sizes["level"])
            ]
        else:
            transposed = da.transpose("time", *h_dims, missing_dims="raise")
            slots = [(None, 0)]
        variables.append((var_name, transposed, slots))

    # First pass: a (variable, level) that holds data at any time step is a
    # slot at every time step. Each slice is read and checked here, so a
    # partly-NaN slice raises before the output file is opened.
    occupied: dict[str, list[tuple[int | None, int]]] = {}
    for var_name, transposed, slots in variables:
        has_data = [False] * len(slots)
        for time_index, time in enumerate(times):
            for i, (slot, level_index) in enumerate(slots):
                data = _read_slice(transposed, time_index, slot)
                if data.shape != (grid.ny, grid.nx):
                    raise ValueError(
                        f"Variable '{var_name}' slice has shape {data.shape}, expected {(grid.ny, grid.nx)}."
                    )
                nan_mask = np.isnan(data)
                if nan_mask.all():
                    continue
                if nan_mask.any():
                    raise ValueError(
                        f"Variable '{var_name}' contains missing values at time {pd.Timestamp(time)} "
                        f"level {level_index}; write_dataset() requires each slice to be "
                        "complete or entirely NaN."
                    )
                if not np.isfinite(data).all():
                    raise ValueError(
                        f"Variable '{var_name}' contains non-finite values that cannot be packed."
                    )
                has_data[i] = True
        occupied[var_name] = [
            slot for slot, keep in zip(slots, has_data, strict=True) if keep
        ]

    with (
        atomic_output(filename_or_obj) as tmp_path,
        File(
            tmp_path,
            mode="w",
            source=source,
            grid=grid,
            vertical_axis=resolved_vertical_axis,
        ) as arl,
    ):
        for time_index, time in enumerate(times):
            recordset = arl.create_recordset(
                pd.Timestamp(time),
                forecast=forecast_hours[time_index],
            )

            for var_name, transposed, _slots in variables:
                for slot, level_index in occupied[var_name]:
                    # An all-NaN slice in an occupied slot is written as a
                    # record flagged missing (forecast hour -1, as pakini.f).
                    recordset.create_datarecord(
                        variable=var_name,
                        level=int(level_index),
                        forecast=forecast_hours[time_index],
                        data=_read_slice(transposed, time_index, slot),
                        diff=diff_names.get(var_name),
                    )

            # Write each time step as soon as it is filled so peak memory is
            # one time step, not the whole file.
            arl.flush()
