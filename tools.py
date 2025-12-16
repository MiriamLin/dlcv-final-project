from mask import Mask
import numpy as np

def inclusion(mask_a: Mask, mask_b: Mask) -> float:
    """
    Compute the proportion of mask_a that overlaps with mask_b.

    This is calculated as the intersection of mask_a and mask_b divided by the size of mask_a
    (i.e., intersection over size_a).
    """
    mask_a_array = mask_a.decode_mask()
    size_a = np.count_nonzero(mask_a_array)
    if size_a == 0:
        return 0
    mask_b_array = mask_b.decode_mask()
    intersection = np.count_nonzero(np.logical_and(mask_a_array, mask_b_array))
    return intersection / size_a

def proxy_dist(mask_a: Mask, mask_b: Mask) -> float:
    return np.linalg.norm(np.array(mask_a.centroid()) - np.array(mask_b.centroid()))