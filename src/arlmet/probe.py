"""Cheap, incomplete summary of an ARL file read from two index records."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from arlmet.exceptions import ARLFormatError
from arlmet.grid import Grid
from arlmet.header import record_length_from_grid
from arlmet.indexrecord import IndexRecord

__all__ = ["Probe", "probe"]


def _read_index(fh: BinaryIO, position: int, path: Path) -> IndexRecord:
    """Read one index record, adding *path* to whatever it raises."""
    try:
        return IndexRecord.from_position(fh, position)
    except EOFError as exc:
        raise ARLFormatError(
            f"{path}: holds no index record at byte {position}; it may be "
            "empty, truncated, or not ARL."
        ) from exc
    except ARLFormatError as exc:
        raise ARLFormatError(f"{path}: {exc}") from exc


@dataclass(frozen=True)
class Probe:
    """
    Cheap summary of an ARL file, from its first two index records and its size.

    Made by :func:`probe`. ``n_steps`` and :attr:`whole` assume every time
    step is as long as the first (its index record's own count of data
    records); a file whose later time steps hold a different set of
    variables, or whose first time step is byte-identically repeated, can
    still read ``whole=True`` here with a misleading ``n_steps`` or ``step``.
    :meth:`arlmet.File.check` reads every index record and catches that;
    probe does not, so treat a probe's result as unverified until
    ``check()`` has run on the file.

    Attributes
    ----------
    path : pathlib.Path
        The probed file.
    first : pandas.Timestamp
        Valid time of the first time step.
    step : pandas.Timedelta or None
        Time between the first and second time steps, or ``None`` when the
        file holds fewer than two (by :attr:`n_steps`).
    n_steps : int
        Number of time steps the file's size divides into, assuming each is
        as long as the first.
    grid : Grid
        Horizontal grid read from the first index record.
    source : str
        Four-character ARL source identifier.
    forecast : int
        Forecast hour of the first index record.
    whole : bool
        ``True`` when the file's size is an exact multiple of the first time
        step's length; ``False`` when it leaves a remainder, which means the
        file is truncated partway through a time step.

    See Also
    --------
    probe : Builds a Probe.
    arlmet.File.check : Full validation, by reading every index record.
    arlmet.index : Indexes every time step of a whole directory.
    """

    path: Path
    first: pd.Timestamp
    step: pd.Timedelta | None
    n_steps: int
    grid: Grid
    source: str
    forecast: int
    whole: bool


def probe(path: str | os.PathLike[str]) -> Probe:
    """
    Cheaply summarize an ARL file from its first two index records and its size.

    Reads the first index record (:meth:`arlmet.indexrecord.IndexRecord.from_position` at
    byte 0) to learn the file's grid, source, forecast hour, and first valid
    time, and how many bytes one time step takes (its index record plus one
    data record per variable per level). A :func:`os.stat` then gives the
    file's size, from which the number of time steps and whether the file is
    whole or truncated follow by division; when there are at least two, the
    second index record (at the byte offset the first time step's length
    gives) gives the time step.

    This is the cost of opening the file (the few small reads over the
    network it takes) without reading every index record, which is what
    :meth:`arlmet.File.check` does for full validation. A probe never reads a
    data record, so it does not unpack or checksum any values.

    Parameters
    ----------
    path : path-like
        Location of the ARL file on disk.

    Returns
    -------
    Probe

    Raises
    ------
    ARLFormatError
        If the file holds no index record (it may be empty or not ARL), or
        the first or second index record cannot be parsed.
    OSError
        If the file cannot be opened or stat'd.

    See Also
    --------
    Probe : The returned summary.
    arlmet.File.check : Full validation, by reading every index record.
    arlmet.index : Indexes every time step of a whole directory.

    Examples
    --------
    >>> arlmet.probe("met.arl")  # doctest: +SKIP
    Probe(path=PosixPath('met.arl'), first=Timestamp('2024-07-18 00:00:00'),
    step=Timedelta('0 days 03:00:00'), n_steps=2, grid=Grid(...), source='TEST',
    forecast=0, whole=True)
    """
    path = Path(path)
    size = path.stat().st_size

    with open(path, "rb") as fh:
        first = _read_index(fh, 0, path)

        record_length = record_length_from_grid(first.grid)
        n_data = sum(len(lvl.variables) for lvl in first.levels)
        bytes_per_step = (1 + n_data) * record_length

        n_steps, remainder = divmod(size, bytes_per_step)
        whole = remainder == 0

        step: pd.Timedelta | None = None
        if n_steps >= 2:
            second = _read_index(fh, bytes_per_step, path)
            step = second.time - first.time

    return Probe(
        path=path,
        first=first.time,
        step=step,
        n_steps=n_steps,
        grid=first.grid,
        source=first.source,
        forecast=first.forecast,
        whole=whole,
    )
