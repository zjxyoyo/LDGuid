import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from scipy.ndimage import gaussian_filter, map_coordinates

class DoubleCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, pre_fire_img, post_fire_img, mask):
        for t in self.transforms:
            pre_fire_img, post_fire_img, mask = t(pre_fire_img, post_fire_img, mask)
        return pre_fire_img, post_fire_img, mask

class DoubleToTensor:
    def __call__(self, pre_fire_img, post_fire_img, mask):
        return (
            torch.tensor(pre_fire_img, dtype=torch.float32).permute(2, 0, 1),
            torch.tensor(post_fire_img, dtype=torch.float32).permute(2, 0, 1),
            torch.tensor(mask).permute(2, 0, 1)
        )

class DoubleHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, pre_fire_img, post_fire_img, mask):
        if random.random() < self.p:
            pre_fire_img = TF.hflip(pre_fire_img)
            post_fire_img = TF.hflip(post_fire_img)
            mask = TF.hflip(mask)
        return pre_fire_img, post_fire_img, mask

class DoubleVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, pre_fire_img, post_fire_img, mask):
        if random.random() < self.p:
            pre_fire_img = TF.vflip(pre_fire_img)
            post_fire_img = TF.vflip(post_fire_img)
            mask = TF.vflip(mask)
        return pre_fire_img, post_fire_img, mask

class DoubleAffine(transforms.RandomAffine):
    def forward(self, pre_fire_img, post_fire_img, mask):
        fill_img = self.fill
        fill_mask = 0
        channels, height, width = TF.get_dimensions(pre_fire_img)
        channels_mask = 1

        if isinstance(pre_fire_img, torch.Tensor):
            if isinstance(fill_img, (int, float)):
                fill_img = [float(fill_img)] * channels
            else:
                fill_img = [float(f) for f in fill_img]

        if isinstance(mask, torch.Tensor):
            fill_mask = [float(fill_mask)] * channels_mask

        img_size = [width, height]
        params = self.get_params(self.degrees, self.translate, self.scale, self.shear, img_size)
        pre_fire_img = TF.affine(pre_fire_img, *params, interpolation=self.interpolation, fill=fill_img, center=self.center)
        post_fire_img = TF.affine(post_fire_img, *params, interpolation=self.interpolation, fill=fill_img, center=self.center)
        mask = TF.affine(mask, *params, interpolation=self.interpolation, fill=fill_mask, center=self.center)

        return pre_fire_img, post_fire_img, mask

class DoubleElasticTransform:
    def __init__(self, alpha=250, sigma=10, p=0.5):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, pre_fire_img, post_fire_img, mask):
        if random.random() < self.p:
            random_state = np.random.RandomState(None)
            shape = pre_fire_img.shape[1:]

            dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
            dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha

            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

            pre_fire_img = torch.tensor([map_coordinates(channel, indices, order=1).reshape(shape) for channel in pre_fire_img.numpy()])
            post_fire_img = torch.tensor([map_coordinates(channel, indices, order=1).reshape(shape) for channel in post_fire_img.numpy()])
            mask = torch.tensor([map_coordinates(channel, indices, order=1).reshape(shape) for channel in mask.numpy()])

        return pre_fire_img, post_fire_img, mask

