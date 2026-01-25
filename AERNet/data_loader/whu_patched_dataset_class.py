import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F

class Patched_WHU_Dataset(Dataset):
    """
    A custom PyTorch Dataset for the patched WHU-CD dataset.
    Reads PNG patches from the A, B, and label folders.
    """
    def __init__(self, root_dir, split='train', output_size=(512, 512), transforms=None):
        """
        Args:
            root_dir (str): The path to the WHU-CD-Patches directory.
            split (str): 'train' or 'test'.
            output_size (tuple): The desired final output size.
            transforms (callable, optional): Optional augmentations.
        """
        self.root_dir = root_dir
        self.split = split
        self.output_size = output_size
        self.transforms = transforms
        
        image_folder_A = os.path.join(self.root_dir, self.split, 'A')
        if not os.path.isdir(image_folder_A):
            raise FileNotFoundError(f"Directory not found: {image_folder_A}.")
        
        self.image_filenames = sorted(os.listdir(image_folder_A))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        
        pre_image_path = os.path.join(self.root_dir, self.split, 'A', filename)
        post_image_path = os.path.join(self.root_dir, self.split, 'B', filename)
        mask_path = os.path.join(self.root_dir, self.split, 'label', filename)

        pre_image_pil = Image.open(pre_image_path).convert('RGB')
        post_image_pil = Image.open(post_image_path).convert('RGB')
        mask_pil = Image.open(mask_path)

        pre_image_np = np.array(pre_image_pil)
        post_image_np = np.array(post_image_pil)
        mask_np = np.array(mask_pil)
        
        pre_image_tensor = torch.from_numpy(pre_image_np).float().permute(2, 0, 1) / 255.0
        post_image_tensor = torch.from_numpy(post_image_np).float().permute(2, 0, 1) / 255.0
        
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]
        binary_mask = np.zeros_like(mask_np, dtype=np.uint8)
        binary_mask[mask_np > 0] = 1
        mask_tensor = torch.from_numpy(binary_mask).long().unsqueeze(0)

        # The resize step is kept for consistency, even if patches are already 512x512.
        # It ensures all outputs are guaranteed to be the correct size.
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