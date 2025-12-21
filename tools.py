from mask import Mask
import numpy as np
import json
import torch
import os
from pathlib import Path
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from agent.inside_pred.model import build_inside_model
from agent.distance_est.model import build_dist_model, DEVICE

def inclusion(mask_a: Mask, mask_b: Mask) -> float:
    """
    Compute the proportion of mask_a that overlaps with mask_b.

    This is calculated as the intersection of mask_a and mask_b divided by the size of mask_a
    (i.e., intersection over size_a).
    """
    mask_a_array = mask_a.decode_mask()
    size_a = np.count_nonzero(mask_a_array)
    mask_b_array = mask_b.decode_mask()
    intersection = np.count_nonzero(np.logical_and(mask_a_array, mask_b_array))
    return intersection / size_a

def proxy_dist(mask_a: Mask, mask_b: Mask) -> float:
    return np.linalg.norm(np.array(mask_a.centroid()) - np.array(mask_b.centroid()))


STR_USE_RGB = "use_rgb"
STR_USE_DEPTH = "use_depth"
STR_IMAGES = "images"
STR_DEPTHS = "depths"

class InclusionModel():
    def __init__(self, config_path: Path):
        with open(config_path, "r") as f:
            config = json.load(f)

        config["input_channels"] = 2 + config[STR_USE_DEPTH] + 3 * config[STR_USE_RGB]

        self.config = config
        self.model = build_inside_model(config)
        self.to_tensor = transforms.ToTensor()
        resize = (360, 640)
        self.resize_tf = transforms.Resize(resize, interpolation=Image.BILINEAR)

    def predict(self, mask_a: Mask, mask_b: Mask, image: str | None = None, data_dir: Path | None = None) -> float:
        r = inclusion(mask_a, mask_b)
        if r > 0.9 or r < 0.01: return r
        with torch.no_grad():
            components = []

            if self.config[STR_USE_RGB]:
                assert image is not None and data_dir is not None
                img_path = os.path.join(data_dir, STR_IMAGES, image)
                components.append(self.to_tensor(self.resize_tf(Image.open(img_path).convert("RGB"))))

            if self.config[STR_USE_DEPTH]:
                assert image is not None and data_dir is not None
                depth_path = os.path.join(data_dir, STR_DEPTHS, image.replace(".png", "_depth.png"))
                depth = self.to_tensor(self.resize_tf(Image.open(depth_path)))
                components.append(depth)
            
            def mask_to_ndarray(mask: Mask) -> np.ndarray:
                return self.to_tensor(self.resize_tf(Image.fromarray((mask.decode_mask() * 255).astype(np.uint8))))
            
            components.append(mask_to_ndarray(mask_b))
            components.append(mask_to_ndarray(mask_a))
            
            return float(
                np.array(
                    torch.sigmoid(
                        self.model(
                            torch.cat(
                                components
                            ).to(DEVICE).unsqueeze(0)
                        ).cpu()
                    )
                )[0]
            )

class DistModel():
    def __init__(self, config_path: Path):
        with open(config_path, "r") as f:
            config = json.load(f)
        
        config["input_channels"] = 2 + config[STR_USE_DEPTH] + 3 * config[STR_USE_RGB]
        
        self.config = config
        self.model = build_dist_model(config)
    
    def predict(self, mask_a: Mask, mask_b: Mask, image: str | None = None, data_dir: Path | None = None, check_inclusion: bool = False, inclusion_model: InclusionModel | None = None) -> float:
        if check_inclusion and inclusion_model.predict(mask_a, mask_b, image, data_dir) > 0.5:
            return 0.0
        resize = (360, 640)
        with torch.no_grad():
            components = []

            if self.config[STR_USE_RGB]:
                assert image is not None and data_dir is not None
                img_path = os.path.join(data_dir, STR_IMAGES, image)
                components.append(np.array(F.resize(Image.open(img_path).convert('RGB'), resize)).astype(np.float32) / 255.0)

            if self.config[STR_USE_DEPTH]:
                assert image is not None and data_dir is not None
                depth_path = os.path.join(data_dir, STR_DEPTHS, image.replace(".png", "_depth.png"))
                depth = np.array(F.resize(Image.open(depth_path), resize, interpolation=Image.BILINEAR)).astype(np.float32) / 255.0
                components.append(depth[..., None])
            
            def mask_to_ndarray(mask: Mask) -> np.ndarray:
                return np.array(F.resize(Image.fromarray(mask.decode_mask().astype(np.float32)), resize, interpolation=Image.NEAREST)).astype(np.float32)[..., None]
            
            components.append(mask_to_ndarray(mask_a))
            components.append(mask_to_ndarray(mask_b))
            
            return float(
                np.array(
                    self.model(
                        torch.tensor(
                            np.concatenate(components, axis=-1)
                        ).permute(2, 0, 1).to(DEVICE).unsqueeze(0)
                    ).cpu()
                )[0]
            )