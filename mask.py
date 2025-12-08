from typing import Dict
import numpy as np
import pycocotools.mask as mask_utils

class Mask:
    def __init__(self, rle: Dict[str, list[int] | str]):
        self.rle = rle
    
    def decode_mask(self) -> np.ndarray:
        return mask_utils.decode(self.rle)
