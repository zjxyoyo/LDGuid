# Weight Loss Added
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
import numpy as np
from sklearn.metrics import f1_score, jaccard_score
from src.helper.losses import DiceLoss
from src.data_loader.OSCDaugmentation import JointCompose, JointRandomHorizontalFlip, JointRandomVerticalFlip, JointRandomRotation

from src.data_loader.oscd_dataset_class import Manual_OSCD_Dataset
from src.model.unet import UNet

# --- CONFIGURATION ---
DATASET_ROOT_PATH = os.path.expanduser('~/projects/def-bereyhia/zjxeric/data/Onera Satellite Change Detection')
SAVE_DIR = "./results/unet_benchmark"
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "best_unet_benchmark.pth")
LOG_CSV_PATH = os.path.join(SAVE_DIR, "benchmark_log.csv")
os.makedirs(SAVE_DIR, exist_ok=True)
NUM_EPOCHS = 300
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
N_CHANNELS = 12
N_CLASSES = 2

# --- HELPER FUNCTIONS ---
def calculate_metrics(outputs, labels):
    preds = torch.argmax(outputs, dim=1)
    preds_np = preds.cpu().numpy().flatten()
    labels_np = labels.cpu().numpy().flatten()
    f1 = f1_score(labels_np, preds_np, average='binary', zero_division=0)
    iou = jaccard_score(labels_np, preds_np, average='binary', zero_division=0)
    return f1, iou

def calculate_class_weights(dataloader, n_classes):
    print("Calculating class weights to handle imbalance...")
    class_counts = torch.zeros(n_classes, dtype=torch.float64)
    for batch in tqdm(dataloader, desc="Calculating Weights"):
        masks = batch['mask']
        masks = masks.clamp(0, n_classes - 1)
        class_counts += torch.bincount(masks.flatten(), minlength=n_classes).double()
    
    # Calculate weights as inverse frequency, add a small epsilon to avoid division by zero
    total_pixels = class_counts.sum()
    class_weights = total_pixels / (class_counts + 1e-6)
    
    print(f"Pixel counts: {class_counts.tolist()}")
    print(f"Calculated class weights: {class_weights.tolist()}")
    return class_weights

def main():
    # --- SETUP ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- DATA LOADING ---
    print("Loading datasets with augmentation...")
    # As the paper suggests: all possible flips and 90-degree rotations
    train_transforms = JointCompose([
        JointRandomHorizontalFlip(p=0.5),
        JointRandomVerticalFlip(p=0.5),
        JointRandomRotation(degrees=[0, 90, 180, 270]),
    ])
    val_transforms = None

    train_dataset = Manual_OSCD_Dataset(root_dir=DATASET_ROOT_PATH, split='train', transforms=train_transforms)
    val_dataset = Manual_OSCD_Dataset(root_dir=DATASET_ROOT_PATH, split='test', transforms=val_transforms)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Data loaded. Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # --- CALCULATE CLASS WEIGHTS ---
    # We use the training loader to calculate weights
    class_weights = calculate_class_weights(train_loader, N_CLASSES).to(device, dtype=torch.float)
    
    # --- MODEL, OPTIMIZER, AND LOSS ---
    print("Initializing U-Net model...")
    model = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES, enable_dropout=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    ce_loss_criterion = nn.CrossEntropyLoss(weight=class_weights)
    dice_loss_criterion = DiceLoss(n_classes=N_CLASSES)
    
    # --- TRAINING LOOP ---
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
            # --- Calculate the combined loss ---
            ce_loss = ce_loss_criterion(outputs, squeezed_masks)
            # DiceLoss needs probabilities, so we apply softmax to the model's output logits
            dice_loss = dice_loss_criterion(outputs, squeezed_masks, softmax=True)
            loss = 0.5 * ce_loss + 0.5 * dice_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss, val_iou, val_f1 = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                images, masks = batch['post_image'].to(device), batch['mask'].to(device)
                outputs = model(images)
                squeezed_masks = masks.squeeze(1)
                loss = ce_loss_criterion(outputs, squeezed_masks)
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

    print("\n--- Benchmark training finished! ---")

if __name__ == "__main__":
    main()