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
    checksum = 0
    for byte_val in packed:
        checksum += byte_val
        if checksum >= 256:
            checksum -= 255
    return checksum


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
    ny, nx = unpacked.shape
    grid = unpacked.astype(np.float32, copy=True)
    initial_value = float(grid[0, 0])

    # Build diffs in the same row-wise order expected by unpack().
    diffs = np.diff(grid, axis=1, prepend=grid[:, :1])
    diffs[:, 0] = np.diff(grid[:, 0], prepend=initial_value)
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

    # Zero tiny values before packing so round-trip precision matches unpack().
    grid[np.abs(grid) < precision] = 0.0
    initial_value = float(grid[0, 0])
    diffs = np.diff(grid, axis=1, prepend=grid[:, :1])
    diffs[:, 0] = np.diff(grid[:, 0], prepend=initial_value)

    # Scale, shift, and clip to get packed integer values.
    packed = np.floor(diffs * scale + 127.5)
    packed = np.clip(packed, 0, 255)
    packed = packed.astype(np.uint8)

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
        The array library to use for computations (e.g., numpy, dask.array).
        If None, numpy is used by default.
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
        if driver is not None and getattr(driver, "__name__", "") == "dask.array":
            raise NotImplementedError("Windowed unpack is not implemented for dask.")
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

    # Convert packed bytes to array using the specified driver
    if getattr(driver, "__name__", "") == "dask.array":
        import dask
        import dask.array as da

        # Delay conversion from buffer to array
        packed_arr = da.from_delayed(
            dask.delayed(np.frombuffer)(packed, dtype=np.uint8)
        )
    else:
        # Eager conversion from buffer to array
        packed_arr = np.frombuffer(packed, dtype=np.uint8)

    # Prepare initial value as array
    initial_arr = np.expand_dims(np.array(initial_value), axis=0)

    # Convert packed to float for calculations
    # THIS IS VERY IMPORTANT!!!
    packed_arr = packed_arr.astype(np.float32)

    # Reshape to 2D grid
    if packed_arr.shape[0] != ny * nx:
        raise ValueError(
            f"Packed data length {packed_arr.shape[0]} does not match expected length {ny * nx}."
        )
    packed_arr = packed_arr.reshape((ny, nx))

    # Calculate the scaling exponent
    scexp = 1.0 / (2.0 ** (7 - exponent))

    # Calculate differential values
    diffs = (packed_arr - 127) * scexp

    # The first column is a cumulative sum of its own diffs, starting with initial_value.
    # We create an array with initial_value followed by the first column's diffs.
    first_col_vals = driver.concatenate((initial_arr, diffs[:, 0]))

    # The cumulative sum gives the unpacked values for the entire first column.
    unpacked_col0 = driver.cumsum(first_col_vals)[1:]

    # Replace the first column of diffs with these now-unpacked starting values.
    diffs = driver.concatenate([unpacked_col0[:, driver.newaxis], diffs[:, 1:]], axis=1)

    # The rest of the grid can now be unpacked by a cumulative sum along the rows (axis=1).
    # Each row starts with its correct, fully unpacked value in the first column.
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
    packed_arr = packed_arr.reshape((ny, nx)).astype(np.float32)

    scexp = 1.0 / (2.0 ** (7 - exponent))
    row0_diffs = (packed_arr[: window.y_stop, 0] - 127.0) * scexp
    row0_vals = np.cumsum(
        np.concatenate((np.array([initial_value], dtype=np.float32), row0_diffs))
    )[1:]

    subset_diffs = (packed_arr[window.y_slice, 1 : window.x_stop] - 127.0) * scexp
    subset_with_origin = np.concatenate(
        (row0_vals[window.y_slice, np.newaxis], subset_diffs),
        axis=1,
    )
    subset = np.cumsum(subset_with_origin, axis=1)[:, window.x_slice]
    subset = np.where(np.abs(subset) < precision, 0.0, subset)
    return subset.astype(np.float32)
