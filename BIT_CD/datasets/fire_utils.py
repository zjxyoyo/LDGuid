import random
import numpy as np
import torch
import torch.nn.functional as F

# Note: All PIL and torchvision imports have been removed as we will operate on NumPy arrays directly.

class CDDataAugmentation:
    def __init__(
            self,
            img_size,
            with_random_hflip=False,
            with_random_vflip=False,
            with_random_crop=False, # We will use a simplified random crop
            with_scale_random_crop=False, # We will use a simplified random crop
            with_random_blur=False, # This is removed as it's PIL-specific
    ):
        self.img_size = img_size
        self.with_random_hflip = with_random_hflip
        self.with_random_vflip = with_random_vflip
        # Consolidate crop flags into one
        self.with_random_crop = with_random_crop or with_scale_random_crop

    def transform(self, imgs, labels, to_tensor=True):
        """
        :param imgs: List of NumPy arrays [H, W, C]
        :param labels: List of NumPy arrays [H, W]
        :return: List of Tensors, List of Tensors
        """

        # --- All operations are now on NumPy arrays ---

        if self.with_random_hflip and random.random() > 0.5:
            imgs = [np.ascontiguousarray(np.fliplr(img)) for img in imgs]
            labels = [np.ascontiguousarray(np.fliplr(label)) for label in labels]

        if self.with_random_vflip and random.random() > 0.5:
            imgs = [np.ascontiguousarray(np.flipud(img)) for img in imgs]
            labels = [np.ascontiguousarray(np.flipud(label)) for label in labels]
            
        # --- Simplified Random Crop for NumPy Arrays ---
        if self.with_random_crop:
            h, w, _ = imgs[0].shape
            if h > self.img_size and w > self.img_size:
                top = random.randint(0, h - self.img_size)
                left = random.randint(0, w - self.img_size)
                
                imgs = [img[top:top + self.img_size, left:left + self.img_size, :] for img in imgs]
                labels = [label[top:top + self.img_size, left:left + self.img_size] for label in labels]

        # --- Manual Conversion to Tensor ---
        if to_tensor:
            # Get the number of channels from the first image
            num_channels = imgs[0].shape[2] if len(imgs) > 0 else 3

            # Transpose from (H, W, C) to (C, H, W) for PyTorch
            imgs = [torch.from_numpy(img.transpose((2, 0, 1))).float() for img in imgs]
            labels = [torch.from_numpy(np.array(label, np.uint8)).long() for label in labels]

            # --- Updated Normalization for N-Channels ---
            # Create a mean and std for the number of channels in the image
            mean = [0.5] * num_channels
            std = [0.5] * num_channels
            
            # Manually apply normalization
            for i in range(len(imgs)):
                for c in range(num_channels):
                    imgs[i][c] = (imgs[i][c] - mean[c]) / std[c]

        return imgs, labels