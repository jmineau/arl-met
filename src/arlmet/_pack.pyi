import numpy as np
from numpy.typing import NDArray

def pack_core(
    unpacked: NDArray[np.float32],
) -> tuple[NDArray[np.uint8], float, int, float]: ...
