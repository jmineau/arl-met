"""Differential packing and unpacking routines for ARL data records."""

from typing import Any

import numpy as np
from numpy import typing as npt

from arlmet._grid import GridWindow
from arlmet._pack import pack_core as _pack_core
from arlmet._pack import unpack_core as _unpack_core

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
    data = np.frombuffer(packed, dtype=np.uint8)
    # A uint32 sum is about twice as fast as a uint64 one and exact over up to
    # 2**32 // 255 bytes, so a payload of any size is summed in such pieces.
    step = 2**32 // 255
    total = sum(
        int(data[i : i + step].sum(dtype=np.uint32)) for i in range(0, data.size, step)
    )
    if total == 0:
        return 0
    return ((total - 1) % 255) + 1


def pack(
    unpacked: npt.NDArray[Any],
) -> tuple[npt.NDArray[np.uint8], float, int, float]:
    """
    Pack a 2D field using the ARL differential byte encoding.

    This is the inverse of :func:`unpack` and a translation of the HYSPLIT
    ``PAKOUT`` routine, in single precision as ``PAKOUT`` is, so it gives the
    same bytes, exponent, precision and initial value.

    Parameters
    ----------
    unpacked : np.ndarray
        The 2D numpy array of shape (ny, nx) to be packed; it is packed as
        ``float32``.

    Returns
    -------
    tuple[np.ndarray, float, int, float]
        Tuple of ``(packed, precision, exponent, initial_value)``.
        ``packed`` is the byte payload, ``precision`` and ``exponent`` are the
        ARL packing parameters, and ``initial_value`` is the value at the
        first grid cell.

    Raises
    ------
    ValueError
        If the field is not 2D, is empty, or holds NaN or infinite values.
    """
    unpacked_arr = np.ascontiguousarray(unpacked, dtype=np.float32)
    # The C extension (_pack.c) is PAKOUT line by line, for speed; it also
    # refuses NaN and infinite values, and sums the checksum, which this drops.
    packed, precision, exponent, initial_value, _ = _pack_core(unpacked_arr)
    return packed, precision, exponent, initial_value


def unpack(
    packed: bytes | bytearray | memoryview,
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

    This is a translation of the HYSPLIT ``PAKINP`` routine and gives the
    same ``float32`` values, bit for bit: the arithmetic is in single
    precision, in ``PAKINP``'s order, and like ``PAKINP`` it sets a value
    smaller than ``precision`` to zero in every column but the first.

    Parameters
    ----------
    packed : bytes, bytearray, or memoryview
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

    n_bytes = memoryview(packed).nbytes
    if n_bytes != ny * nx:
        raise ValueError(
            f"Packed data length {n_bytes} does not match expected length {ny * nx}."
        )
    # The C extension (_pack.c) is PAKINP, reading only the rows and columns
    # that feed the window.
    return _unpack_core(
        packed,
        nx,
        ny,
        window.x_start,
        window.x_stop,
        window.y_start,
        window.y_stop,
        precision,
        exponent,
        initial_value,
    )
