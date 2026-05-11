"""Internal timestamp normalization helpers."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd


def ensure_timestamp(value: Any, *, floor: str | None = None) -> pd.Timestamp:
    """Return a concrete pandas.Timestamp, rejecting NaT values."""
    ts = value if isinstance(value, pd.Timestamp) else pd.Timestamp(value)
    if pd.isna(ts):
        raise ValueError(f"Invalid timestamp value: {value!r}")
    if floor is not None:
        ts = ts.floor(floor)
        if pd.isna(ts):
            raise ValueError(
                f"Invalid timestamp value after floor({floor!r}): {value!r}"
            )
    return cast(pd.Timestamp, ts)
