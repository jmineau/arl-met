import numpy as np
from numpy.typing import NDArray

def pack_core(
    unpacked: NDArray[np.float32],
    scale: float,
    inv_scale: float,
    initial_value: float,
) -> NDArray[np.uint8]: ...
