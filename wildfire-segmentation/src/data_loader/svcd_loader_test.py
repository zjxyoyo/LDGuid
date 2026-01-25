# svcd_loader_test.py

import os
import torch
from torch.utils.data import DataLoader

# Import the new SVCD_Dataset class from the other file
from .svcd_dataset_class import SVCD_Dataset

def main():
    """A simple test function for the SVCD DataLoader."""
    print("--- Starting SVCD DataLoader Test ---")
    
    # Define the path to your dataset's root directory
    # This should point to the folder containing the train/, val/, and test/ subdirectories
    base_path = '~/projects/def-bereyhia/zjxeric/data'
    DATASET_ROOT_PATH = os.path.join(os.path.expanduser(base_path), "SVCD_dataset")
    
    print(f"Attempting to load data from: {DATASET_ROOT_PATH}")

    try:
        # 1. Instantiate the custom PyTorch Dataset for the 'train' split
        train_dataset = SVCD_Dataset(root_dir=DATASET_ROOT_PATH, split='train')
        
        # 2. Create the DataLoader
        # We use a small batch size for testing and num_workers=0 for stability
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
        
        print(f"Successfully created DataLoader. Found {len(train_dataset)} training samples.")

        # 3. Fetch one batch of data to test
        print("\n--- Fetching one sample batch... ---")
        sample_batch = next(iter(train_loader))
        print("Successfully fetched one batch!")
        
        # 4. Check the data shapes and types
        pre_images = sample_batch['pre_image']
        post_images = sample_batch['post_image']
        masks = sample_batch['mask']

        print("\n--- Batch Data Shape and Type Verification ---")
        print(f"Pre-image batch shape:  {pre_images.shape},  DType: {pre_images.dtype}")
        print(f"Post-image batch shape: {post_images.shape},  DType: {post_images.dtype}")
        print(f"Mask batch shape:       {masks.shape},        DType: {masks.dtype}")
        
        print("\n--- SVCD DataLoader Test SUCCESSFUL! ---")

    except Exception as e:
        print(f"\n--- SVCD DataLoader Test FAILED ---")
        print(f"An error occurred: {e}")
        print("Please check the DATASET_ROOT_PATH and the dataset's directory structure.")

if __name__ == "__main__":
    main()