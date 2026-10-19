"""Differential packing and unpacking routines for ARL data records."""

from typing import Any

import numpy as np
from numpy import typing as npt

from arlmet._pack import pack_core as _pack_core
from arlmet.grid import GridWindow

__all__ = ["pack", "unpack", "calculate_checksum"]


def calculate_checksum(packed: bytes | bytearray | memoryview) -> int:
    """
    Compute the ARL checksum for a packed payload.

    Parameters
    ----------
    packed : bytes, bytearray, or memoryview
        The packed byte array for which to calculate the checksum.

    Returns
    -------
    int
        Rolling checksum used in ARL index records.
    """
    total = int(np.frombuffer(packed, dtype=np.uint8).sum(dtype=np.uint64))
    if total == 0:
        return 0
    return ((total - 1) % 255) + 1


def pack(
    unpacked: npt.NDArray[Any],
) -> tuple[npt.NDArray[np.uint8], float, int, float]:
    """
    Pack a 2D field using the ARL differential byte encoding.

    This is the inverse of :func:`unpack` and mirrors the HYSPLIT
    ``PAKOUT`` routine.

    Parameters
    ----------
    unpacked : np.ndarray
        The 2D numpy array of shape (ny, nx) to be packed.

    Returns
    -------
    tuple[np.ndarray, float, int, float]
        Tuple of ``(packed, precision, exponent, initial_value)``.
        ``packed`` is the byte payload, ``precision`` and ``exponent`` are the
        ARL packing parameters, and ``initial_value`` is the reference value at
        the first grid cell.
    """
    unpacked_arr = np.array(unpacked, dtype=np.float32, copy=True)
    initial_value = float(unpacked_arr[0, 0])

    # Build diffs in the same row-wise order expected by unpack().
    diffs = np.empty_like(unpacked_arr)
    diffs[:, 1:] = unpacked_arr[:, 1:] - unpacked_arr[:, :-1]
    diffs[1:, 0] = unpacked_arr[1:, 0] - unpacked_arr[:-1, 0]
    diffs[0, 0] = 0.0
    rmax = float(np.max(np.abs(diffs)))

    # --- Calculate packing exponent and precision ---
    if rmax > 0.0:
        sexp = np.log(rmax) / np.log(2.0)
        exponent = int(sexp)
        if sexp >= 0.0 or sexp.is_integer():
            exponent += 1
    else:
        exponent = 0

    precision = (2.0**exponent) / 254.0
    scale = 2.0 ** (7 - exponent)
    inv_scale = 1.0 / scale

    # Zero tiny values before packing so round-trip precision matches unpack().
    unpacked_arr[np.abs(unpacked_arr) < precision] = 0.0
    initial_value = float(unpacked_arr[0, 0])

    # Feedback-loop encoder: each delta is computed against the *reconstructed*
    # running value (what unpack's cumsum will reproduce), not the original float.
    # This keeps per-cell quantisation error bounded to 0.5 * inv_scale regardless
    # of grid width.  Implemented as a C extension (_pack.c) for native speed.
    packed = _pack_core(unpacked_arr, scale, inv_scale, initial_value)

    return packed, precision, exponent, initial_value


def unpack(
    packed: bytes | bytearray,
    nx: int,
    ny: int,
    precision: float,
    exponent: int,
    initial_value: float,
    *,
    window: GridWindow | None = None,
) -> npt.NDArray[np.float32]:
    """
    Unpack an ARL differential byte stream into a 2D field.

    This is a vectorized translation of the HYSPLIT ``PAKINP`` routine and
    gives the same ``float32`` values, bit for bit: the arithmetic is in
    single precision, in ``PAKINP``'s order, and like ``PAKINP`` it sets a
    value smaller than ``precision`` to zero in every column but the first.

    Parameters
    ----------
    packed : bytes or bytearray
        The packed byte array to be unpacked.
    nx : int
        The number of columns in the full data grid.
    ny : int
        The number of rows in the full data grid.
    precision : float
        Precision of the packed data. Values with an absolute value smaller
        than this are set to zero, except in the first column (``x = 0``),
        which ``PAKINP`` leaves as it is.
    exponent : int
        The packing scaling exponent.
    initial_value : float
        The initial real value at the grid position (0,0).
    window : GridWindow, optional
        Rectangular grid window to unpack. When provided, only the requested
        subset is reconstructed.

    Returns
    -------
    numpy.ndarray
        Unpacked ``float32`` 2D array with shape ``(ny, nx)`` or the requested windowed
        shape when ``window`` is provided.
    """
    if window is None:
        window = GridWindow(x_start=0, x_stop=nx, y_start=0, y_stop=ny)
    elif window.x_stop > nx or window.y_stop > ny:
        raise ValueError("GridWindow extends beyond the packed grid bounds.")

    packed_arr = np.frombuffer(packed, dtype=np.uint8)
    if packed_arr.shape[0] != ny * nx:
        raise ValueError(
            f"Packed data length {packed_arr.shape[0]} does not match expected length {ny * nx}."
        )
    # Keep the source buffer byte-typed and cast only the rows and columns
    # that feed the window, so a crop does not promote the full grid.
    packed_arr = packed_arr.reshape((ny, nx))

    # Each byte is a difference from the previous value in units of scexp, a
    # power of 2, so converting them to float32 is exact.
    scexp = np.float32(2.0 ** (exponent - 7))

    # The first column is coded down the rows from the initial value. PAKINP
    # adds one row's difference at a time in single precision, so start the
    # running sum at the float32 initial value instead of adding it at the end.
    column0 = np.empty(window.y_stop + 1, dtype=np.float32)
    column0[0] = initial_value
    column0[1:] = packed_arr[: window.y_stop, 0]
    column0[1:] -= 127.0
    column0[1:] *= scexp
    np.cumsum(column0, out=column0)

    # Each row is then coded across x from its first-column value; cumsum along
    # a row adds one difference at a time, as PAKINP's loop over I does.
    rows = packed_arr[window.y_slice, : window.x_stop].astype(np.float32)
    rows -= 127.0
    rows *= scexp
    rows[:, 0] = column0[window.y_start + 1 : window.y_stop + 1]
    np.cumsum(rows, axis=1, out=rows)

    # PAKINP's precision test, which it applies to every column but the first.
    rest = rows[:, 1:]
    rest[np.abs(rest) < np.float32(precision)] = 0.0

    return np.ascontiguousarray(rows[:, window.x_start :])
