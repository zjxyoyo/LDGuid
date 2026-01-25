import torch
import torch.nn.functional as F

def resize_mask(mask, size):
    """
    Resize mask to the specified size using bilinear interpolation.
    Args:
        mask (torch.Tensor): The mask tensor of shape (B, H, W, 1).
        size (tuple): The target size (C, H, W).
    Returns:
        torch.Tensor: The resized mask tensor of shape (B, C, H, W).
    """
    # Change shape from (B, H, W, 1) to (B, 1, H, W)
    mask = mask.permute(0, 3, 1, 2)
    # Resize the mask
    resized_mask = F.interpolate(mask.float(), size=size[1:], mode='bilinear', align_corners=False)
    return resized_mask

# Example usage
mask = torch.randn(8, 512, 512, 1)
target_size = (1, 64, 64)
resized_mask = resize_mask(mask, target_size)
print(resized_mask.shape)  # Should be torch.Size([8, 1, 64, 64])