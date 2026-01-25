import os
import tifffile  
import numpy as np
from torch.utils import data
from datasets.fire_utils import CDDataAugmentation

IMG_FOLDER_NAME = "A"
IMG_POST_FOLDER_NAME = 'B'
LIST_FOLDER_NAME = 'list'
ANNOT_FOLDER_NAME = "label"

def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=str)
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list

def get_img_post_path(root_dir, img_name):
    # Ensure the file extension is correct for .tif files
    return os.path.join(root_dir, IMG_POST_FOLDER_NAME, img_name.replace('.png', '.tif'))

def get_img_path(root_dir, img_name):
    # Ensure the file extension is correct for .tif files
    return os.path.join(root_dir, IMG_FOLDER_NAME, img_name.replace('.png', '.tif'))

def get_label_path(root_dir, img_name):
    # Ensure the file extension is correct for .tif files
    return os.path.join(root_dir, ANNOT_FOLDER_NAME, img_name.replace('.png', '.tif'))



class FireDataset(data.Dataset):
    def __init__(self, root_dir, split='train', img_size=256, is_train=True, to_tensor=True, label_transform=None):
        super(FireDataset, self).__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split
        self.list_path = os.path.join(self.root_dir, LIST_FOLDER_NAME, self.split + '.txt')
        self.img_name_list = load_img_name_list(self.list_path)
        self.to_tensor = to_tensor
        self.label_transform = label_transform
        
        if is_train:
            self.augm = CDDataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
                with_scale_random_crop=True,
                with_random_blur=True,
            )
        else:
            self.augm = CDDataAugmentation(img_size=self.img_size)

    def __getitem__(self, index):
        name = self.img_name_list[index]
        A_path = get_img_path(self.root_dir, name)
        B_path = get_img_post_path(self.root_dir, name)
        L_path = get_label_path(self.root_dir, name)

        # Use tifffile to read the 12-channel images
        img = tifffile.imread(A_path)
        img_B = tifffile.imread(B_path)
        label = tifffile.imread(L_path)

        # Ensure label is a 2D array
        if label.ndim == 3:
            label = label[:, :, 0]

        # Normalize label pixels from 0-255 to 0-1
        if self.label_transform == 'norm':
            label = label // 255

        # Apply augmentations
        [img, img_B], [label] = self.augm.transform([img, img_B], [label], to_tensor=self.to_tensor)
        
        return {'name': name, 'A': img, 'B': img_B, 'L': label}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.img_name_list)

