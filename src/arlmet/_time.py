"""Internal timestamp normalization helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd


def ensure_timestamp(value: Any, *, floor: str | None = None) -> pd.Timestamp:
    """
    Return a concrete pandas.Timestamp without a time zone, rejecting NaT values.

    A time without a time zone is UTC (as every ARL time is); a time-zone-aware
    one is converted to UTC.
    """
    ts = value if isinstance(value, pd.Timestamp) else pd.Timestamp(value)
    if pd.isna(ts):
        raise ValueError(f"Invalid timestamp value: {value!r}")
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    if floor is not None:
        ts = ts.floor(floor)
        if pd.isna(ts):
            raise ValueError(
                f"Invalid timestamp value after floor({floor!r}): {value!r}"
            )
    return ts


def utc_times(index: Any) -> pd.DatetimeIndex:
    """
    Return times without a time zone, in UTC, as every ARL time is.

    Pass a coordinate's index (``coord.to_index()``), not ``.values``: pandas 3
    warns that ``DatetimeIndex.values`` drops the time zone of an aware index.
    """
    times = pd.DatetimeIndex(pd.to_datetime(index))
    return times.tz_convert(None) if times.tz is not None else times
