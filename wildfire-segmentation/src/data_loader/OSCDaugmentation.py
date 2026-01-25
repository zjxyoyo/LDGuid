import torch
import random
from torchvision.transforms import functional as F

class JointCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, pre_image, post_image, mask):
        for t in self.transforms:
            pre_image, post_image, mask = t(pre_image, post_image, mask)
        return pre_image, post_image, mask

class JointRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, pre_image, post_image, mask):
        if random.random() < self.p:
            pre_image = F.hflip(pre_image)
            post_image = F.hflip(post_image)
            mask = F.hflip(mask)
        return pre_image, post_image, mask

class JointRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, pre_image, post_image, mask):
        if random.random() < self.p:
            pre_image = F.vflip(pre_image)
            post_image = F.vflip(post_image)
            mask = F.vflip(mask)
        return pre_image, post_image, mask

class JointRandomRotation:
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, pre_image, post_image, mask):
        angle = random.choice(self.degrees)
        pre_image = F.rotate(pre_image, angle)
        post_image = F.rotate(post_image, angle)
        mask = F.rotate(mask, angle)
        return pre_image, post_image, mask