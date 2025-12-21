import os
import json
import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.nn.functional as F
import pycocotools.mask as mask_utils

ImageFile.LOAD_TRUNCATED_IMAGES = True

class InsideDataset(Dataset):
    def __init__(self, json_path, image_dir, depth_dir=None, resize=(360, 640), use_depth=True, use_rgb: bool=True):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.use_rgb = use_rgb
        self.use_depth = use_depth
        self.resize = resize
        self.to_tensor = transforms.ToTensor()
        self.resize_tf = transforms.Resize(resize, interpolation=Image.BILINEAR)
        
    def __len__(self):
        return len(self.data)
    
    def decode_rle(self, rle):
        mask = mask_utils.decode(rle).astype(np.float32)
        return mask
    
    def __getitem__(self, idx):
        item = self.data[idx]
        components = []

        if self.use_rgb:
            assert self.image_dir is not None
            img_path = os.path.join(self.image_dir, item['image'])
            img = Image.open(img_path).convert('RGB')
            img = self.resize_tf(img)
            img = self.to_tensor(img)  # 3 x H x W
            components.append(img)
        
        # Optional: Depth
        if self.use_depth and self.depth_dir is not None:
            depth_path = os.path.join(self.depth_dir, item['image'].replace('.png', '_depth.png'))
            depth = Image.open(depth_path)
            depth = self.resize_tf(depth)
            depth = self.to_tensor(depth)  # 1 x H x W
            components.append(depth)

        # Masks
        buffer_mask = self.decode_rle(item['buffer_rle'])  # (H, W)
        obj_mask = self.decode_rle(item['obj_rle'])        # (H, W)

        # Resize masks to target shape
        buffer_mask = Image.fromarray((buffer_mask * 255).astype(np.uint8))
        buffer_mask = self.resize_tf(buffer_mask)
        buffer_mask = self.to_tensor(buffer_mask)  # 1 x H x W
        components.append(buffer_mask)

        obj_mask = Image.fromarray((obj_mask * 255).astype(np.uint8))
        obj_mask = self.resize_tf(obj_mask)
        obj_mask = self.to_tensor(obj_mask)  # 1 x H x W
        components.append(obj_mask)

        x = torch.cat(components)
        
        y = torch.tensor(item['inside'], dtype=torch.float32)

        return x, y