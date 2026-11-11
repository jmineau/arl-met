"""
:func:`sample_points`: the values HYSPLIT uses at any (lon, lat, z, time).

How one time step is sampled (:func:`_sample_time`):

1. Place the points on the grid (:mod:`.horizontal`).
2. Build the column of the time step at the grid points around each point
   (:mod:`.column`), whose fields are read as HYSPLIT reads them (:mod:`.fields`).
3. Interpolate each variable's profile (:mod:`.profile`) to the point's ``z``
   at each of those grid points, on HYSPLIT's internal levels at heights
   (:mod:`.internal`), and blend the grid points' values.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any, Literal, Union, get_args

import numpy as np
import numpy.typing as npt
import pandas as pd

from arlmet._time import ensure_timestamp
from arlmet._version import __version__
from arlmet.vertical import _DEFAULT_MODEL_TOP

from .column import Column, column_for, terrain_at
from .horizontal import (
    WIND_PAIRS,
    WRF_STAGGER,
    Plan,
    SampleMethod,
    blend,
    build_plan,
    node_plan,
    rotate_winds,
)
from .internal import InternalLevels, sample_on_internal_levels
from .profile import ColumnZKind, sample_variable

if TYPE_CHECKING:
    from arlmet._file import File

__all__ = ["sample_points"]

# One sampling input: an open File or a path to an ARL file.
FileLike = Union["File", str, "os.PathLike[str]"]

#: Vertical coordinate systems accepted for the ``z`` of sample points.
ZKind = Literal["native", "pressure", "agl", "msl"]


def _normalize_points(
    points: pd.DataFrame | Mapping[str, Any],
    variable_names: Sequence[str],
    *,
    time: pd.Timestamp | str | None,
    implied_time: pd.Timestamp | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate *points* and split it into the result frame and sampling coordinates.

    Returns ``(result, coords)``. *result* is a copy of *points* as a DataFrame,
    with every column and the index preserved, that the sampled variables are
    added to. *coords* has a positional ``RangeIndex`` aligned row-for-row with
    *result* and holds float ``lon``/``lat``/``z`` and Timestamp ``time``
    columns.

    The point time comes from the ``time`` column, else from *time*, else from
    *implied_time* (the single valid time of the inputs). Giving both a
    ``time`` column and *time* is ambiguous and raises ``ValueError``.
    """
    df = points if isinstance(points, pd.DataFrame) else pd.DataFrame(points)

    missing = [name for name in ("lon", "lat", "z") if name not in df.columns]
    if missing:
        raise ValueError(
            "Point sampling requires 'lon', 'lat', and 'z' columns, "
            f"missing {', '.join(repr(name) for name in missing)}."
        )

    clashes = [name for name in variable_names if name in df.columns]
    if clashes:
        raise ValueError(
            "Requested variables collide with existing columns of points: "
            f"{', '.join(repr(name) for name in clashes)}. Rename or drop those "
            "columns before sampling."
        )

    has_time_column = "time" in df.columns
    if has_time_column and time is not None:
        raise ValueError(
            "points has a 'time' column and time= was also given; pass only one."
        )

    coords = pd.DataFrame(
        {
            "lon": np.asarray(df["lon"], dtype=float),
            "lat": np.asarray(df["lat"], dtype=float),
            "z": np.asarray(df["z"], dtype=float),
        }
    )
    if has_time_column:
        point_times = pd.to_datetime(df["time"])
        if point_times.dt.tz is not None:
            point_times = point_times.dt.tz_convert("UTC").dt.tz_localize(None)
        coords["time"] = point_times.to_numpy()
    elif time is not None:
        coords["time"] = ensure_timestamp(time)
    elif implied_time is not None:
        coords["time"] = implied_time
    else:
        raise ValueError(
            "Point sampling needs a time: add a 'time' column to points or pass "
            "time=, unless the input file(s) contain exactly one valid time."
        )
    if np.any(pd.isna(coords["time"])):
        raise ValueError("points has missing (NaT) values in its 'time' column.")

    return df.copy(), coords


def _check_wind_pairs(variable_names: Sequence[str]) -> list[tuple[str, str]]:
    """Return the wind pairs present in *variable_names*; raise if a component lacks its partner."""
    requested = set(variable_names)
    pairs: list[tuple[str, str]] = []
    for u_name, v_name in WIND_PAIRS:
        has_u = u_name in requested
        has_v = v_name in requested
        if has_u != has_v:
            missing = v_name if has_u else u_name
            present = u_name if has_u else v_name
            raise ValueError(
                f"earth_relative=True needs both wind components: '{present}' "
                f"was requested without '{missing}'."
            )
        if has_u:
            pairs.append((u_name, v_name))
    return pairs


def _map_times(files: Sequence[File]) -> dict[pd.Timestamp, File]:
    """Map each valid time to the one input File that contains it."""
    time_map: dict[pd.Timestamp, File] = {}
    for file in files:
        for file_time in file.times:
            file_time = ensure_timestamp(file_time)
            if file_time in time_map:
                raise ValueError(
                    f"Multiple input files contain meteorology for time {file_time}."
                )
            time_map[file_time] = file
    return time_map


def _describe_times(times: Sequence[pd.Timestamp]) -> str:
    """Say which valid times the input files hold, briefly, with the spacing if it is even."""
    if not times:
        return "no times"
    steps = {b - a for a, b in zip(times[:-1], times[1:], strict=True)}
    every = ""
    if len(steps) == 1:
        hours = next(iter(steps)) / pd.Timedelta(hours=1)
        every = f" (every {hours:g} hours)"
    if len(times) <= 8:
        return ", ".join(str(t) for t in times) + every
    return f"{len(times)} times from {times[0]} to {times[-1]}{every}"


def _sample_time(
    file: File,
    time: pd.Timestamp,
    coords: pd.DataFrame,
    variable_names: Sequence[str],
    *,
    z_kind: ZKind,
    method: SampleMethod,
    wind_pairs: Sequence[tuple[str, str]],
    model_top: float | None,
) -> dict[str, npt.NDArray[Any]]:
    """
    Sample variables at points (normalized by :func:`_normalize_points`) that all share one valid time.

    The method is fixed by the file's vertical axis, as HYSPLIT's, with no
    fallback between methods; :func:`sample_points` says what each one needs.
    At heights the order is HYSPLIT's (#116): the column at each of the four
    grid points around a point, with its own ground, heights and pressures, is
    interpolated to the point's ``z``, then the four values are blended over
    those that have one (:func:`~.horizontal.blend`). HYSPLIT has no pressure
    targets: at pressures and native levels each level is blended first, so a
    target near the ground is near the ground of the point's own column.

    Returns one array per variable, aligned with the rows of *coords*.
    """
    recordset = file[time]
    lon = coords["lon"].to_numpy(dtype=float)
    lat = coords["lat"].to_numpy(dtype=float)
    plan = build_plan(recordset.grid, lon, lat, method=method)
    targets = coords["z"].to_numpy(dtype=float)
    levels = None
    column_z: ColumnZKind
    if z_kind == "msl":
        # A height above sea level is the height above the ground at the point
        # itself, as HYSPLIT places a particle above the terrain there; each
        # grid point's column is then read at that height above its own ground.
        targets = targets - terrain_at(recordset, plan)
        column_z = "agl"
    else:
        column_z = z_kind
    if column_z == "agl" and model_top is not None:
        levels = InternalLevels.for_axis(
            file.vertical_axis, model_top, source=file.source
        )

    # On a WRF file UWND and VWND are on their own staggered grids: each is
    # sampled at its own array position, from the columns of the grid points
    # around it, as HYSPLIT builds each node's profile and then reads U and V
    # half a cell away (#109).
    stagger = WRF_STAGGER if file.source == "AWRF" else {}
    staggered = {name: stagger[name] for name in variable_names if name in stagger}
    each_level_first = column_z == "pressure"
    nodes: dict[tuple[float, float], tuple[Plan, npt.NDArray[Any], Column | None]] = {}
    for shift in {(0.0, 0.0), *staggered.values()}:
        shifted = (
            plan
            if shift == (0.0, 0.0)
            else build_plan(recordset.grid, lon, lat, method=method, stagger=shift)
        )
        var_nodes, weights = node_plan(shifted, each_level_first=each_level_first)
        column = column_for(recordset, var_nodes, z_kind=column_z, levels=levels)
        nodes[shift] = (var_nodes, weights, column)

    place = None
    if levels is not None:
        # The particle's terrain, interpolated from the grid points' as HYSPLIT
        # interpolates its terrain array, places it among the internal levels.
        _, weights, column = nodes[(0.0, 0.0)]
        assert column is not None
        place = levels.position(targets, blend(column.terrain, weights))

    sampled: dict[str, npt.NDArray[Any]] = {}
    for variable in variable_names:
        var_nodes, weights, column = nodes[staggered.get(variable, (0.0, 0.0))]
        if place is not None:
            assert levels is not None and column is not None
            sampled[variable] = sample_on_internal_levels(
                recordset, variable, *place, levels, var_nodes, weights, column
            )
        else:
            at_nodes = sample_variable(
                recordset,
                variable,
                np.tile(targets, len(weights)),
                z_kind=column_z,
                plan=var_nodes,
                column=column,
            )
            sampled[variable] = blend(at_nodes, weights)
    rotate_winds(sampled, file.grid, lon, lat, wind_pairs)
    return sampled


def sample_points(
    files: FileLike | Sequence[FileLike],
    points: pd.DataFrame | Mapping[str, Any],
    variables: str | Iterable[str],
    *,
    time: pd.Timestamp | str | None = None,
    z_kind: ZKind = "pressure",
    method: SampleMethod = "linear",
    earth_relative: bool = False,
    model_top: float | None = _DEFAULT_MODEL_TOP,
) -> pd.DataFrame:
    """
    Sample meteorological variables at arbitrary (lon, lat, z, time) points.

    Accepts a single ARL file or a sequence of files spanning different time
    periods. Each may be an open :class:`~arlmet._file.File` or a path to an
    ARL file; paths are opened (read mode) and closed automatically, while
    already-open Files are left open for the caller to manage. A path is
    opened and its index records read on every call, so in a loop over many
    calls on one file pass an open File. Each point is
    sampled from the file that contains its valid time. Time is not
    interpolated: each point's time must be one of the valid times of the
    files, and a point between two valid times raises ``ValueError``.

    Parameters
    ----------
    files : File, path-like, or sequence of File or path-like
        A single open :class:`~arlmet._file.File` or path, or a sequence of
        Files and/or paths. Each valid time must appear in at most one file.
    points : pandas.DataFrame or mapping
        Table-like object with ``lon``, ``lat`` (degrees), and ``z`` columns,
        and optionally a ``time`` column. Any other columns are carried
        through to the result unchanged.
    variables : str or iterable of str
        One or more ARL field names (e.g. ``"TEMP"``, ``"UWND"``), or
        ``"pressure"`` for the virtual pressure variable. A name must not
        collide with an existing column of ``points``. On a WRF file with
        mass-coupled winds (``MUU0``, ``MUV0``) ``UWND``, ``VWND`` and ``WWND``
        are decoupled as HYSPLIT does (``UWND * MSFU / MUU0``, ...), and on a
        file with ``THET`` but no ``TEMP``, ``TEMP`` is ``THET * (PRES / 1000)
        ** 0.286``.
    time : pandas.Timestamp or str, optional
        One valid time for every point, used when ``points`` has no ``time``
        column. Passing both raises ``ValueError``. When neither is given, the
        input file(s) must contain exactly one valid time, which is used. A
        time must be a valid time of a file (no interpolation in time).
    z_kind : {"pressure", "native", "agl", "msl"}, default "pressure"
        Vertical coordinate system of the ``z`` values. See
        :meth:`arlmet.File.sample_points` for the method each vertical
        coordinate system uses and the fields it requires.
    method : {"linear", "nearest"}, default "linear"
        Horizontal interpolation: bilinear or nearest grid point. At heights
        (``"agl"``, ``"msl"``), as in HYSPLIT, the column at each of the four
        grid points around a point is interpolated to the point's ``z`` first,
        with that grid point's own ground and level heights, and the four
        values are then blended; a grid point whose column ends below ``z``
        does not count, and the others are reweighted. At pressures and native
        levels each level is blended first, as before.
        On a WRF file (source ``AWRF``) ``UWND`` and ``VWND`` are read where WRF's
        staggered grid holds them, half a cell east and north of each point,
        as HYSPLIT reads them; "nearest" gives the winds at the nearest
        cell's center. The *Sampling* guide has the details.
    earth_relative : bool, default False
        Rotate sampled wind pairs (``UWND``/``VWND``, ``U10M``/``V10M``) from
        the grid axes to east/north, using the meridian convergence of each
        file's grid. Both components of a pair must be requested. No effect on
        lat/lon grids.
    model_top : float or None, default 25000
        The HYSPLIT run's top of model domain [m], the ``CONTROL`` file's
        line after the vertical motion option (PYSTILT's ``z_top``, 25000 by
        default). At heights (``"agl"``, ``"msl"``) the values are the ones a
        HYSPLIT run with that top uses: on HYSPLIT's own internal levels
        (those its ``metlvl`` picks for this file's vertical grid), which
        follow the terrain as sigma surfaces that go flat at
        ``max(model_top, 25000)`` m, linear between those levels, the first
        level's value below it (10 m, or 5 m on the finest grid), and above a
        file's top ``TEMP``, ``UWND``, ``VWND`` and pressure extrapolated as
        HYSPLIT does (other variables NaN there). A sigma, terrain-following,
        hybrid or WRF file without ``SHGT`` takes HYSPLIT's terrain estimate
        from ``PRSS``. The default matches a run with PYSTILT's default top; a run with another
        top needs its value, because below 25 km the internal levels depend on
        it (the *Sampling* guide says how far). ``None`` reads each grid
        point's column at the point's own height above that grid point's
        ground, between the file's own levels, which is not what HYSPLIT does
        (differences of up to 4.6 K, 3.5 m/s and 1.9 hPa below 5 km) and gives
        NaN above a file's top. The default applies at heights only: at
        ``"pressure"`` and ``"native"`` it is ignored, and a value passed with
        either raises.

    Returns
    -------
    pandas.DataFrame
        Copy of ``points`` (all columns and the index preserved) with one
        added column per requested variable. Points outside the grid or the
        vertical range are NaN. A point within 1e-6 cells of the grid's edge
        is on it, so a projected grid's edge nodes, given by the longitude and
        latitude :meth:`arlmet.Grid.calculate_coords` reports, have values.
        ``result.attrs["arlmet_version"]`` is the arlmet version that sampled.
        At heights (``z_kind`` ``"agl"`` or ``"msl"``) with a ``model_top``,
        ``result.attrs["model_top"]`` is the one used [m]; with ``model_top=None``
        or at pressures it is absent.

    Raises
    ------
    ValueError
        If a required column is missing, ``time`` is given alongside a
        ``time`` column, no time is given and the inputs hold more than one
        time, a point time is in none of the files, a time is in more than one
        file, a variable name collides with a column of ``points``, ``z_kind``
        or ``method`` is invalid, a field that ``z_kind`` requires is
        missing, or ``model_top`` is passed with ``z_kind`` ``"pressure"`` or
        ``"native"`` or is not a positive number.

    Examples
    --------
    >>> import pandas as pd
    >>> import arlmet
    >>> points = pd.DataFrame(
    ...     {"lon": [-111.9], "lat": [40.7], "z": [850.0], "time": ["2024-07-18 00:00"]}
    ... )
    >>> arlmet.sample_points("met.arl", points, ["UWND", "VWND"]).round(1)
         lon   lat      z              time  UWND  VWND
    0 -111.9  40.7  850.0  2024-07-18 00:00   6.2  -3.8

    Sample across files spanning different times by passing their paths:

    >>> arlmet.sample_points(
    ...     ["met_00.arl", "met_06.arl"], points, ["UWND", "VWND"]
    ... ).round(1)
         lon   lat      z              time  UWND  VWND
    0 -111.9  40.7  850.0  2024-07-18 00:00   6.2  -3.8
    """
    from arlmet._file import File  # lazy: _file imports this module

    if z_kind not in get_args(ZKind):
        raise ValueError("z_kind must be one of 'native', 'pressure', 'agl', or 'msl'.")
    if method not in get_args(SampleMethod):
        raise ValueError("method must be 'linear' or 'nearest'.")
    if model_top is not None and model_top is not _DEFAULT_MODEL_TOP:
        if z_kind not in ("agl", "msl"):
            raise ValueError(
                "model_top applies to heights: z_kind must be 'agl' or 'msl'."
            )
        if not (np.isfinite(model_top) and model_top > 0):
            raise ValueError(
                f"model_top must be a positive number of meters, got {model_top!r}."
            )
    names = (variables,) if isinstance(variables, str) else tuple(map(str, variables))
    if not names:
        raise ValueError("variables must include at least one variable name.")
    wind_pairs = _check_wind_pairs(names) if earth_relative else []
    inputs = [files] if isinstance(files, (File, str, os.PathLike)) else list(files)
    if not inputs:
        raise ValueError("sample_points requires at least one input file.")

    with ExitStack() as opened:
        # Paths are opened here and closed on return; open Files stay the caller's.
        handles = [
            f if isinstance(f, File) else opened.enter_context(File(f)) for f in inputs
        ]
        time_map = _map_times(handles)
        implied_time = next(iter(time_map)) if len(time_map) == 1 else None
        result, coords = _normalize_points(
            points, names, time=time, implied_time=implied_time
        )

        codes, point_times = pd.factorize(coords["time"])
        missing_times = [t for t in point_times if ensure_timestamp(t) not in time_map]
        if missing_times:
            raise ValueError(
                "No input file contains the requested point times: "
                + ", ".join(str(ensure_timestamp(t)) for t in missing_times)
                + ". A point time must be one of the valid times of the input "
                "file(s); sample_points does not interpolate in time. They hold "
                + _describe_times(sorted(time_map))
                + "."
            )

        columns = {name: np.full(len(coords), np.nan) for name in names}
        for code, point_time in enumerate(point_times):
            point_time = ensure_timestamp(point_time)
            rows = np.flatnonzero(codes == code)
            sampled = _sample_time(
                time_map[point_time],
                point_time,
                coords.iloc[rows],
                names,
                z_kind=z_kind,
                method=method,
                wind_pairs=wind_pairs,
                model_top=model_top,
            )
            for name, values in sampled.items():
                columns[name][rows] = values

    for name, values in columns.items():
        result[name] = values
    result.attrs["arlmet_version"] = __version__
    if z_kind in ("agl", "msl") and model_top is not None:
        result.attrs["model_top"] = float(model_top)
    return result
