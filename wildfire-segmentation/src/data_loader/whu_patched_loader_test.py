import os
import torch
from torch.utils.data import DataLoader


from .whu_patched_dataset_class import Patched_WHU_Dataset

def main():
    """A simple test function for the patched WHU-CD DataLoader."""
    print("--- Starting Patched WHU-CD DataLoader Test ---")
    
    # Define the path to your new patched dataset directory
    base_path = '~/projects/def-bereyhia/zjxeric/data'
    DATASET_ROOT_PATH = os.path.join(os.path.expanduser(base_path), "WHU-CD-Patches")
    
    print(f"Attempting to load data from: {DATASET_ROOT_PATH}")

    try:
        train_dataset = Patched_WHU_Dataset(root_dir=DATASET_ROOT_PATH, split='train')
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
        
        print(f"Successfully created DataLoader. Found {len(train_dataset)} training samples.")

        print("\n--- Fetching one sample batch... ---")
        sample_batch = next(iter(train_loader))
        print("Successfully fetched one batch!")
        
        pre_images = sample_batch['pre_image']
        post_images = sample_batch['post_image']
        masks = sample_batch['mask']

        print("\n--- Batch Data Shape and Type Verification ---")
        print(f"Pre-image batch shape:  {pre_images.shape},  DType: {pre_images.dtype}")
        print(f"Post-image batch shape: {post_images.shape},  DType: {post_images.dtype}")
        print(f"Mask batch shape:       {masks.shape},        DType: {masks.dtype}")
        
        print("\n--- Patched WHU-CD DataLoader Test SUCCESSFUL! ---")

    except Exception as e:
        print(f"\n--- Patched WHU-CD DataLoader Test FAILED ---")
        print(f"An error occurred: {e}")
        print("Please check the DATASET_ROOT_PATH and the dataset's directory structure.")

if __name__ == "__main__":
    main()