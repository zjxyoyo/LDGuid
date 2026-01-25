import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
from sklearn.metrics import f1_score, jaccard_score

# --- Import the necessary modules ---
from src.data_loader.whu_patched_dataset_class import Patched_WHU_Dataset
from src.model.unet import UNet
from .augmentation import JointCompose, JointRandomHorizontalFlip, JointRandomVerticalFlip, JointRandomRotation

def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- 1. Configuration ---
# ==============================================================================
DATASET_ROOT_PATH = os.path.expanduser('~/projects/def-bereyhia/zjxeric/data/WHU-CD-Patches')
# Default to seed 42 if no argument is given
try:
    # Read the seed from the first command-line argument
    SEED = int(sys.argv[1])
except (IndexError, ValueError):
    print("No seed argument provided. Defaulting to 42.")
    SEED = 42


SAVE_DIR = f"./results/whu_unet_benchmark_seed{SEED}" 
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "best_unet_benchmark_whu.pth")
LOG_CSV_PATH = os.path.join(SAVE_DIR, "benchmark_whu_log.csv")
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_EPOCHS = 150 # Increased epochs
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
N_CHANNELS = 3   # WHU-CD is 3-channel RGB
N_CLASSES = 2
# ==============================================================================

# --- 2. Helper Functions (Metrics only) ---
def calculate_metrics(outputs, labels):
    preds = torch.argmax(outputs, dim=1)
    preds_np, labels_np = preds.cpu().numpy().flatten(), labels.cpu().numpy().flatten()
    f1 = f1_score(labels_np, preds_np, average='binary', zero_division=0)
    iou = jaccard_score(labels_np, preds_np, average='binary', zero_division=0)
    return f1, iou

# --- 3. Main Training Function ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading with Augmentation ---
    print("Loading Patched WHU-CD dataset with augmentation...")
    
    # --- Defining the augmentation pipeline for the training set ---
    train_transforms = JointCompose([
        JointRandomHorizontalFlip(p=0.5),
        JointRandomVerticalFlip(p=0.5),
        JointRandomRotation(degrees=[0, 90, 180, 270]),
    ])

    train_dataset = Patched_WHU_Dataset(root_dir=DATASET_ROOT_PATH, split='train', transforms=train_transforms)
    val_dataset = Patched_WHU_Dataset(root_dir=DATASET_ROOT_PATH, split='test', transforms=None)
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(SEED)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=g
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=g
    )
    print(f"Data loaded. Train: {len(train_dataset)}, Validation: {len(val_dataset)}")

    # --- Model, Optimizer, and Loss ---
    print("Initializing U-Net model...")
    model = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES, enable_dropout=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # --- Using simple, unweighted CrossEntropyLoss as requested ---
    criterion = nn.CrossEntropyLoss()
    
    # --- Training Loop ---
    with open(LOG_CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_iou', 'val_f1'])
    
    best_val_f1 = 0.0
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            images, masks = batch['post_image'].to(device), batch['mask'].to(device)
            squeezed_masks = masks.squeeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, squeezed_masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validation Loop ---
        model.eval()
        val_loss, val_iou, val_f1 = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                images, masks = batch['post_image'].to(device), batch['mask'].to(device)
                outputs = model(images)
                squeezed_masks = masks.squeeze(1)
                loss = criterion(outputs, squeezed_masks)
                val_loss += loss.item()
                f1, iou = calculate_metrics(outputs, squeezed_masks)
                val_f1 += f1
                val_iou += iou
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        avg_val_f1 = val_f1 / len(val_loader)
        
        print(f"Epoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f} | Val F1: {avg_val_f1:.4f}")
        
        with open(LOG_CSV_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, avg_val_loss, avg_val_iou, avg_val_f1])
        
        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Validation F1-score improved to {best_val_f1:.4f}. Model saved.")

    print("\n--- WHU-CD Benchmark training finished! ---")

if __name__ == "__main__":
    main()