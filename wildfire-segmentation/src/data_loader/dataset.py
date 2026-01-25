import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
from src.data_loader.augmentation import DoubleToTensor, DoubleCompose, DoubleElasticTransform, DoubleHorizontalFlip, DoubleVerticalFlip, DoubleAffine, CustomColorJitter, GaussianNoise
import torch
import h5py


class BaseDataset(Dataset):
    mask_key = "mask"
    pre_fire_key = "pre_fire"
    post_fire_key = "post_fire"

    def __init__(self, datasets, keys, include_pre_fire=False):
        self._datasets = datasets
        self.keys = keys
        self.include_pre_fire = include_pre_fire
        self.image_mask_transform = None
        self.image_transform = None

    def __len__(self):
        return len(self.keys)

    def __set_transforms__(self, image_mask_transforms, image_transforms):
        self.image_transform = image_transforms
        self.image_mask_transform = image_mask_transforms

    def __getitem__(self, idx):
        key, img_index = self.keys[idx]
        data = self._datasets[key][img_index]
        post_fire_img = np.array(data[self.post_fire_key]) / 10000
        mask = np.array(data[self.mask_key])

        if self.include_pre_fire:
            pre_fire_img = np.array(data[self.pre_fire_key]) / 10000
        else:
            pre_fire_img = None

        if self.image_mask_transform:
            post_fire_img, mask = self.image_mask_transform(post_fire_img, mask)
            if self.include_pre_fire:
                pre_fire_img, _ = self.image_mask_transform(pre_fire_img, mask)

        if self.image_transform:
            post_fire_img = self.image_transform(post_fire_img)
            if self.include_pre_fire:
                pre_fire_img = self.image_transform(pre_fire_img)

        result = {
            "post_fire_image": post_fire_img,
            "mask": mask,
        }
        if self.include_pre_fire:
            result["pre_fire_image"] = pre_fire_img

        return result




class WildfireDataset:
    def __init__(self, path_to_data: Union[str, Path], transforms, include_pre_fire=False):
        self.path_to_data = Path(path_to_data)
        self.transforms = transforms
        self.include_pre_fire = include_pre_fire
        self._datasets = load_from_disk(str(self.path_to_data.absolute()))

        self.keys: List[Tuple[str, int]] = [
            (k, i) for k in self._datasets.data.keys() for i in range(len(self._datasets[k]))
        ]

        self.idxs = range(len(self.keys))

        n_val_test = int(len(self.keys) * 0.2)
        n_train = len(self.keys) - n_val_test
        self.train_idxs = self.idxs[:n_train]
        val_and_test_idxs = self.idxs[n_train:]

        mid_index = n_val_test // 2

        self.val_idxs = val_and_test_idxs[:mid_index]
        self.test_idxs = val_and_test_idxs[mid_index:]

        # Initialize
        # by default, just convert the NP images to Tensor
        self.image_mask_transform = DoubleToTensor()
        self.image_transform = None

        self.train_pre_fire = BaseDataset(self._datasets, [self.keys[i] for i in self.train_idxs], include_pre_fire=True)
        self.train_post_fire = BaseDataset(self._datasets, [self.keys[i] for i in self.train_idxs], include_pre_fire=False)
        self.val_pre_fire = BaseDataset(self._datasets, [self.keys[i] for i in self.val_idxs], include_pre_fire=True)
        self.val_post_fire = BaseDataset(self._datasets, [self.keys[i] for i in self.val_idxs], include_pre_fire=False)
        self.test_pre_fire = BaseDataset(self._datasets, [self.keys[i] for i in self.test_idxs], include_pre_fire=True)
        self.test_post_fire = BaseDataset(self._datasets, [self.keys[i] for i in self.test_idxs], include_pre_fire=False)

    def extract_features(self, autoencoder, dataloader, device, is_pre_fire=False):
        autoencoder.eval()
        features = []
        with torch.no_grad():
            for data in dataloader:
                images = data['pre_fire_image' if is_pre_fire else 'post_fire_image'].to(device)
                # Ensure the images are in the correct shape [batch_size, channels, height, width]
                if images.ndim == 4 and images.shape[1] == 512:
                    images = images.permute(0, 3, 1, 2)
                latent, _ = autoencoder(images)
                features.append(latent.cpu())
        return torch.cat(features, dim=0)

    def get_feature_diff_dataloader(self, autoencoder, batch_size, device):
        pre_fire_loader = DataLoader(self.train_pre_fire, batch_size=batch_size, shuffle=False, num_workers=4)
        post_fire_loader = DataLoader(self.train_post_fire, batch_size=batch_size, shuffle=False, num_workers=4)

        pre_fire_features = self.extract_features(autoencoder, pre_fire_loader, device, is_pre_fire=True)
        post_fire_features = self.extract_features(autoencoder, post_fire_loader, device)

        # masks = []
        # for data in post_fire_loader:
        #     masks.append(data['mask'])
        # masks = torch.cat(masks, dim=0)

        masks = []
        for data in post_fire_loader:
            mask = data['mask'].unsqueeze(1)  # Ensure masks have a single channel
            masks.append(mask)
        masks = torch.cat(masks, dim=0).squeeze(1)  # Ensure the masks have the correct shape [batch_size, H, W]
        print(f"Pre-fire feature size: {pre_fire_features.size()}, Post-fire feature size: {post_fire_features.size()}, Mask size: {masks.size()}")
        feature_diff_dataset = FeatureDiffDataset(pre_fire_features, post_fire_features, masks)
        return DataLoader(feature_diff_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


class CombinedDataset(Dataset):
    def __init__(self, dataset, transforms=None):
        self.train_pre_fire = dataset.train_pre_fire
        self.train_post_fire = dataset.train_post_fire
        self.keys = self.train_post_fire.keys  # Use the keys from one of the datasets (assuming they are the same)
        self.transforms = transforms

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key, img_index = self.keys[idx]
        data_pre = self.train_pre_fire._datasets[key][img_index]
        data_post = self.train_post_fire._datasets[key][img_index]
        
        pre_fire_img = np.array(data_pre[self.train_pre_fire.pre_fire_key]) / 10000
        post_fire_img = np.array(data_post[self.train_post_fire.post_fire_key]) / 10000
        mask = np.array(data_post[self.train_post_fire.mask_key])
        # if self.transforms:
        #     pre_fire_img, mask = self.transforms(pre_fire_img, mask)
        #     post_fire_img, mask = self.transforms(post_fire_img, mask)

        # return post_fire_img, pre_fire_img, mask
        if self.transforms:
            pre_fire_img, post_fire_img, mask = self.transforms(pre_fire_img, post_fire_img, mask)
        return {
            "pre_fire_image": pre_fire_img,
            "post_fire_image": post_fire_img,
            "mask": mask,
        }


class FeatureDiffDataset(Dataset):
    def __init__(self, pre_fire_features, post_fire_features, masks):
        self.pre_fire_features = pre_fire_features
        self.post_fire_features = post_fire_features
        self.masks = masks

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        pre_feature = self.pre_fire_features[idx]
        post_feature = self.post_fire_features[idx]
        mask = self.masks[idx]
        feature_diff = post_feature - pre_feature
        return {'feature_diff': feature_diff, 'mask': mask}