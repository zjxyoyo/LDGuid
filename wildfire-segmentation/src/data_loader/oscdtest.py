# OSCDloader.py (Diagnostic Version)

import os
from oscd_dataset_class import Manual_OSCD_Dataset

def main():
    base_path = '~/projects/def-bereyhia/zjxeric/data'
    DATASET_ROOT_PATH = os.path.join(os.path.expanduser(base_path), "Onera Satellite Change Detection")
    
    print("--- Initializing Dataset for diagnostics ---")
    # We only need to create the dataset object
    train_dataset = Manual_OSCD_Dataset(root_dir=DATASET_ROOT_PATH, split='train')
    
    print(f"\nDataset initialized. Found {len(train_dataset)} samples.")
    print("="*60)
    
    # --- Manually fetch the first two samples to see the detailed logs ---
    print("Fetching sample 0...")
    sample_0 = train_dataset[0]
    
    print("\nFetching sample 1...")
    sample_1 = train_dataset[1]
    
    print("="*60)
    print("Diagnostics finished. Check the shapes printed above.")

if __name__ == "__main__":
    main()