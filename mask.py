from typing import Dict
import numpy as np
import pycocotools.mask as mask_utils

class Mask:
    def __init__(self, rle: Dict[str, list[int] | str]):
        self.rle = rle
    
    def decode_mask(self) -> np.ndarray:
        return mask_utils.decode(self.rle)

    def centroid(self) -> np.array:
        """Compute the centroid (x, y) of a mask."""
        mask_array = self.decode_mask()
        y_indices, x_indices = np.nonzero(mask_array > 0)
        return np.array([np.mean(x_indices), np.mean(y_indices)])

    def __lt__(self, mask):
        return self.centroid()[0] < mask.centroid()[0]
