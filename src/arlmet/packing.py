"""Differential packing and unpacking routines for ARL data records."""

import numpy as np
from numpy import typing as npt

from arlmet.grid import GridWindow


def calculate_checksum(packed: bytes | bytearray) -> int:
    """
    Compute the ARL checksum for a packed payload.

    Parameters
    ----------
    packed : bytes or bytearray
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
    unpacked: np.ndarray,
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

    packed = np.empty_like(unpacked_arr, dtype=np.uint8)
    packed[0, 0] = 127

    # ARL encodes each delta against the running reconstructed field, not the
    # original neighboring value. That keeps quantization error bounded instead
    # of letting it accumulate across the row.
    previous_row0 = initial_value
    for y in range(unpacked_arr.shape[0]):
        if y > 0:
            code = np.floor((unpacked_arr[y, 0] - previous_row0) * scale + 127.5)
            packed[y, 0] = np.uint8(np.clip(code, 0, 255))
            previous_row0 += (float(packed[y, 0]) - 127.0) * inv_scale

        previous_value = previous_row0
        for x in range(1, unpacked_arr.shape[1]):
            code = np.floor((unpacked_arr[y, x] - previous_value) * scale + 127.5)
            packed[y, x] = np.uint8(np.clip(code, 0, 255))
            previous_value += (float(packed[y, x]) - 127.0) * inv_scale

    return packed, precision, exponent, initial_value


def unpack(
    packed: bytes | bytearray,
    nx: int,
    ny: int,
    precision: float,
    exponent: int,
    initial_value: float,
    driver=None,
    window: GridWindow | None = None,
) -> npt.ArrayLike:
    """
    Unpack an ARL differential byte stream into a 2D field.

    This is a vectorized translation of the HYSPLIT ``PAKINP`` routine.

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
        than this will be set to zero.
    exponent : int
        The packing scaling exponent.
    initial_value : float
        The initial real value at the grid position (0,0).
    driver : module, optional
        The array library to use for computations. If None, numpy is used by
        default.
    window : GridWindow, optional
        Rectangular grid window to unpack. When provided, only the requested
        subset is reconstructed.

    Returns
    -------
    array-like
        Unpacked 2D array with shape ``(ny, nx)`` or the requested windowed
        shape when ``window`` is provided.
    """
    if window is not None:
        return _unpack_window(
            packed=packed,
            nx=nx,
            ny=ny,
            precision=precision,
            exponent=exponent,
            initial_value=initial_value,
            window=window,
        )

    if driver is None:
        driver = np

    # Convert packed bytes to an eager NumPy array for unpacking.
    packed_arr = np.frombuffer(packed, dtype=np.uint8)

    # Prepare initial value as array
    initial_arr = np.expand_dims(np.array(initial_value), axis=0)

    # Reshape the flat byte stream back to the full 2D ARL grid.
    if packed_arr.shape[0] != ny * nx:
        raise ValueError(
            f"Packed data length {packed_arr.shape[0]} does not match expected length {ny * nx}."
        )
    packed_arr = packed_arr.reshape((ny, nx))

    # Convert packed bytes to float before applying the signed differential
    # transform and cumulative sums used by the unpacker.
    # THIS IS VERY IMPORTANT!!!
    packed_arr = packed_arr.astype(np.float32)

    # Calculate the scaling exponent
    scexp = 1.0 / (2.0 ** (7 - exponent))

    # The first column is vertically differential-coded across rows; once that
    # running state is restored, the rest of each row can be recovered with a
    # standard cumulative sum across x.
    diffs = (packed_arr - 127.0) * scexp
    diffs[:, 0] = driver.cumsum(diffs[:, 0], axis=0) + initial_arr
    unpacked = driver.cumsum(diffs, axis=1)

    # Apply the precision check to the final grid.
    unpacked = driver.where(driver.abs(unpacked) < precision, 0.0, unpacked)

    # Force float32 (4 bytes per value)
    return unpacked.astype(np.float32)


def _unpack_window(
    packed: bytes | bytearray,
    nx: int,
    ny: int,
    precision: float,
    exponent: int,
    initial_value: float,
    window: GridWindow,
) -> npt.NDArray[np.float32]:
    """
    Unpack only a rectangular subset of the grid.
    """
    if window.x_stop > nx or window.y_stop > ny:
        raise ValueError("GridWindow extends beyond the packed grid bounds.")

    packed_arr = np.frombuffer(packed, dtype=np.uint8)
    if packed_arr.shape[0] != ny * nx:
        raise ValueError(
            f"Packed data length {packed_arr.shape[0]} does not match expected length {ny * nx}."
        )

    # Keep the source buffer byte-typed here and only cast the slices that feed
    # the requested window, so subset extraction does not promote the full grid.
    packed_arr = packed_arr.reshape((ny, nx))

    scexp = 1.0 / (2.0 ** (7 - exponent))
    row0_diffs = packed_arr[: window.y_stop, 0].astype(np.float32)
    row0_diffs -= 127.0
    row0_diffs *= scexp
    row0_vals = np.cumsum(row0_diffs, dtype=np.float32)
    row0_vals += initial_value

    x_start = window.x_start
    x_stop = window.x_stop
    y_slice = window.y_slice
    subset_diffs = packed_arr[y_slice, 1:x_stop].astype(np.float32)
    subset_diffs -= 127.0
    subset_diffs *= scexp
    subset_prefix = np.cumsum(subset_diffs, axis=1, dtype=np.float32)
    subset_prefix += row0_vals[y_slice, np.newaxis]

    if x_start == 0:
        # Windows that include the first column need that reconstructed origin
        # inserted explicitly before the row-wise prefixes from x=1 onward.
        subset = np.empty((window.y_stop - window.y_start, x_stop), dtype=np.float32)
        subset[:, 0] = row0_vals[y_slice]
        if x_stop > 1:
            subset[:, 1:] = subset_prefix
    else:
        subset = subset_prefix[:, x_start - 1 :]
    subset = np.where(np.abs(subset) < precision, 0.0, subset)
    return subset.astype(np.float32)
