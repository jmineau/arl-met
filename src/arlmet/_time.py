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
