import numpy as np
from numpy.typing import NDArray

def pack_core(
    unpacked: NDArray[np.float32],
) -> tuple[NDArray[np.uint8], float, int, float]: ...
def unpack_core(
    packed: bytes | bytearray | memoryview,
    nx: int,
    ny: int,
    x_start: int,
    x_stop: int,
    y_start: int,
    y_stop: int,
    precision: float,
    exponent: int,
    initial_value: float,
    /,
) -> NDArray[np.float32]: ...
