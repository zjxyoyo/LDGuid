# Version 3.0

import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import tifffile as tiff
from torchvision.transforms import functional as F

class Manual_OSCD_Dataset(Dataset):
    """
    FINAL-FIXED VERSION 3.0: Correctly handles multi-channel RGB/RGBA masks
    by converting them to single-channel binary masks.
    """
    def __init__(self, root_dir, split='train', output_size=(512, 512), transforms=None):
        self.root_dir = root_dir
        self.split = split
        self.output_size = output_size
        self.transforms = transforms
        self.samples = self._find_samples()
        if not self.samples:
            raise RuntimeError(f"Found 0 samples for split '{self.split}' in {self.root_dir}.")

    def _find_samples(self):
        # This method is correct and does not need changes.
        samples = []
        split_file_path = os.path.join(self.root_dir, f"{self.split}.txt")
        try:
            with open(split_file_path, 'r') as f:
                content = f.read()
                city_names = [city.strip() for city in content.split(',') if city.strip()]
        except FileNotFoundError: return []
        for city in city_names:
            pre_change_path = os.path.join(self.root_dir, city, 'imgs_1_rect')
            post_change_path = os.path.join(self.root_dir, city, 'imgs_2_rect')
            mask_path = os.path.join(self.root_dir, city, 'cm', 'cm.png')
            if os.path.exists(mask_path) and os.path.exists(pre_change_path) and os.path.exists(post_change_path):
                 samples.append({'pre_path': pre_change_path, 'post_path': post_change_path, 'mask_path': mask_path})
        return samples

    def _read_multiband_tiff(self, folder_path):
        # This method is correct and does not need changes.
        bands = []
        for i in range(1, 13):
            band_path = os.path.join(folder_path, f'B{i:02d}.tif')
            bands.append(tiff.imread(band_path))
        return np.stack(bands, axis=-1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_paths = self.samples[idx]

        pre_image_np = self._read_multiband_tiff(sample_paths['pre_path'])
        post_image_np = self._read_multiband_tiff(sample_paths['post_path'])
        
        # === THE FINAL FIX IS HERE: ROBUST MASK PROCESSING ===
        mask_pil = Image.open(sample_paths['mask_path'])
        mask_np = np.array(mask_pil)

        # If the mask is multi-channel (like RGB or RGBA), take only the first channel.
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]
        
        # Create a new, clean binary mask.
        # Initialize an array of zeros with the same height and width.
        binary_mask = np.zeros_like(mask_np, dtype=np.uint8)
        
        # Where the original mask has any non-zero value, set our new mask's value to 1.
        # This robustly handles 255 and any other artifact values.
        binary_mask[mask_np > 0] = 1

        pre_image_np = pre_image_np.astype(np.float32) / 10000.0
        post_image_np = post_image_np.astype(np.float32) / 10000.0
        
        pre_image_tensor = torch.from_numpy(pre_image_np).permute(2, 0, 1)
        post_image_tensor = torch.from_numpy(post_image_np).permute(2, 0, 1)
        # Create tensor from the guaranteed-binary 2D mask, then add the channel dimension.
        mask_tensor = torch.from_numpy(binary_mask).long().unsqueeze(0)

        # Resizing (this will now work correctly on the clean mask)
        pre_image_tensor = F.resize(pre_image_tensor, self.output_size, interpolation=F.InterpolationMode.BILINEAR)
        post_image_tensor = F.resize(post_image_tensor, self.output_size, interpolation=F.InterpolationMode.BILINEAR)
        mask_tensor = F.resize(mask_tensor, self.output_size, interpolation=F.InterpolationMode.NEAREST)
        
        if self.transforms:
            pre_image_tensor, post_image_tensor, mask_tensor = self.transforms(
                pre_image_tensor, post_image_tensor, mask_tensor
            )

        return {
            "pre_image": pre_image_tensor,
            "post_image": post_image_tensor,
            "mask": mask_tensor,
        }