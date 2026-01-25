import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data_loader.augmentation import (
    CustomColorJitter,
    DoubleAffine,
    DoubleCompose,
    DoubleElasticTransform,
    DoubleHorizontalFlip,
    DoubleToTensor,
    DoubleVerticalFlip,
    GaussianNoise,
)


# Helper function to set up data transforms and get the data loader
def get_loader(data_set, is_train, loader_args):
    # NOTE: This is for pre-post-fire dataset! Hard mean and std for now
    mean = [
        x / 10000
        for x in [
            529.8350,
            670.5348,
            891.0950,
            1077.3064,
            1355.6924,
            1803.6713,
            2002.1877,
            2129.0054,
            2197.2866,
            2246.2820,
            2313.7886,
            1721.2771,
        ]
    ]

    std = [
        x / 10000
        for x in [
            662.8983,
            705.7620,
            733.9877,
            836.4911,
            847.7748,
            837.1517,
            878.2544,
            910.7749,
            913.5588,
            975.5820,
            1125.1262,
            961.8238,
        ]
    ]

    if is_train:
        image_mask_transform = DoubleCompose(
            [
                DoubleToTensor(),
                DoubleElasticTransform(alpha=250, sigma=10),
                DoubleHorizontalFlip(),
                DoubleVerticalFlip(),
                DoubleAffine(degrees=(-15, 15), translate=(0.15, 0.15), scale=(0.8, 1)),
            ]
        )
        image_transform = transforms.Compose(
            [
                CustomColorJitter(brightness=[0.5, 1.5], contrast=[0.8, 1.2], saturation=[0.8, 1.2]),
                transforms.Normalize(mean, std),
                GaussianNoise(mean=0, std=0.2),
            ]
        )
    else:
        image_mask_transform = DoubleCompose([DoubleToTensor()])
        image_transform = transforms.Compose([transforms.Normalize(mean, std)])

    data_set.__set_transforms__(image_mask_transform, image_transform)

    loader = DataLoader(data_set, **loader_args)
    return loader


# Helper function to calculate means
def get_mean_std(trainLoader):
    imgs = None
    for batch in trainLoader:
        images = batch["image"]
        if imgs is None:
            imgs = images
        else:
            imgs = torch.cat((imgs, images), dim=0)
    mean = imgs.mean(dim=(0, 2, 3))
    std = imgs.std(dim=(0, 2, 3))

    print(mean, std)
    return mean, std
