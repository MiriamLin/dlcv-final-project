from mask import Mask
import numpy as np
from typing import Tuple

def centroid(mask_array: np.ndarray) -> Tuple[float, float]:
    """Compute the centroid (x, y) of a mask."""
    y_indices, x_indices = np.nonzero(mask_array > 0)
    if len(x_indices) == 0:
        return (0, 0)
    return (float(np.mean(x_indices)), float(np.mean(y_indices)))

def is_left(mask_a: Mask, mask_b: Mask) -> bool:
    centroid_a = centroid(mask_a.decode_mask())
    centroid_b = centroid(mask_b.decode_mask())
    return centroid_a[0] < centroid_b[0]