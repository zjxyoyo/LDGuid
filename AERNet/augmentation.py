# Augmentation script for trainings.
import torch
import random
from torchvision.transforms import functional as F

class JointCompose:
    """
    Composes several joint transforms together.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, pre_image, post_image, mask):
        for t in self.transforms:
            pre_image, post_image, mask = t(pre_image, post_image, mask)
        return pre_image, post_image, mask

class JointRandomHorizontalFlip:
    """
    Applies a random horizontal flip identically to all input tensors.
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, pre_image, post_image, mask):
        if random.random() < self.p:
            pre_image = F.hflip(pre_image)
            post_image = F.hflip(post_image)
            mask = F.hflip(mask)
        return pre_image, post_image, mask

class JointRandomVerticalFlip:
    """
    Applies a random vertical flip identically to all input tensors.
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, pre_image, post_image, mask):
        if random.random() < self.p:
            pre_image = F.vflip(pre_image)
            post_image = F.vflip(post_image)
            mask = F.vflip(mask)
        return pre_image, post_image, mask

class JointRandomRotation:
    """
    Applies a random rotation from a given list of degrees identically to all input tensors.
    """
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, pre_image, post_image, mask):
        angle = random.choice(self.degrees)
        if angle != 0:
            pre_image = F.rotate(pre_image, angle)
            post_image = F.rotate(post_image, angle)
            mask = F.rotate(mask, angle)
        return pre_image, post_image, mask