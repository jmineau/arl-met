import numpy as np
from numpy import typing as npt

try:
    import dask

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False


def calculate_checksum(packed: bytes | bytearray) -> int:
    """
    Calculates the checksum for a packed byte array.

    Parameters
    ----------
    packed : bytes or bytearray
        The packed byte array for which to calculate the checksum.

    Returns
    -------
    int
        The calculated checksum.
    """
    checksum = 0
    for byte_val in packed:
        checksum += byte_val
        if checksum >= 256:
            checksum -= 255
    return checksum


def pack(
    unpacked: np.ndarray,
) -> tuple[bytes, float, int, float]:
    """
    Packs a 2D numpy array using a differential packing scheme.

    This function is a vectorized Python translation of the HYSPLIT
    FORTRAN PAKOUT subroutine. It is the inverse of the `unpack` function.
    It determines the optimal packing parameters (precision, exponent)
    based on the data.

    Parameters
    ----------
    unpacked : np.ndarray
        The 2D numpy array of shape (ny, nx) to be packed.

    Returns
    -------
    tuple[bytes, float, int, float]
        A tuple containing:
        - The packed bytes.
        - The calculated precision of the packed data.
        - The packing scaling exponent.
        - The initial real value at grid position (0,0).
    """
    raise NotImplementedError("This function is not yet implemented.")
    ny, nx = unpacked.shape
    grid = unpacked.astype(np.float32)
    initial_value = grid[0, 0]

    # --- Vectorized: Find max difference to determine packing parameters ---
    # Create a flattened array with the special order of comparison
    flat_grid = grid.flatten()
    # Prepend the initial value for the first difference calculation
    rold_values = np.concatenate(([initial_value], flat_grid[:-1]))
    # The FORTRAN logic for finding rmax has a tricky dependency where rold
    # from the end of a row is not the start for the next.
    # The original loop logic is more reliable here.
    rold = initial_value
    rmax = 0.0
    for j in range(ny):
        for i in range(nx):
            rmax = max(abs(grid[j, i] - rold), rmax)
            rold = grid[j, i]
        if j + 1 < ny:
            rold = grid[j + 1, 0]

    # --- Calculate packing exponent and precision ---
    if rmax > 0.0:
        sexp = np.log(rmax) / np.log(2.0)
        exponent = int(sexp)
        if sexp >= 0.0 or sexp.is_integer():
            exponent += 1
    else:
        exponent = 0

    precision = (2.0**exponent) / 254.0
    scexp = 2.0 ** (7 - exponent)

    # --- Vectorized Packing ---
    # Apply precision and re-check initial_value
    grid[np.abs(grid) < precision] = 0.0
    initial_value = grid[0, 0]

    # Calculate differences along rows (axis=1)
    row_diffs = np.diff(grid, axis=1, prepend=grid[:, :1])

    # Calculate differences for the first column
    col0_diffs = np.diff(grid[:, 0], prepend=initial_value)

    # Combine the differences
    diffs = row_diffs
    diffs[:, 0] = col0_diffs

    # Scale, shift, and clip to get packed integer values
    packed = np.round(diffs / scexp + 127.5)
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
) -> npt.ArrayLike:
    """
    Unpacks a differentially packed 2D array into a 2D numpy array.
    This function is a vectorized Python translation of the HYSPLIT
    FORTRAN PAKINP subroutine. It uses a differential unpacking scheme
    where each value is derived from the previous one. The implementation
    is optimized using vectorized operations for high performance.

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
        If None, numpy is used by default, or dask.array if available.

    Returns
    -------
    array-like
        The unpacked 2D array with the same shape as `data`, using
        the specified `driver` array library.
    """
    if driver is None:
        if DASK_AVAILABLE:  # prefer dask if available
            import dask.array as da

            driver = da
        else:  #  default to numpy
            driver = np

    # Convert packed bytes to array using the specified driver
    if DASK_AVAILABLE and driver.__name__ == "dask.array":
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
            f"Packed data size {packed_arr.shape[0]} does not match expected size {ny * nx}."
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
