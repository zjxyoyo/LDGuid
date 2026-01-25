import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
from sklearn.metrics import f1_score, jaccard_score

import random
import numpy as np
from argparse import ArgumentParser

from models.network import zh_net 

# --- Import your Dataset Class ---
from data_loader.levir_dataset_class import LEVIR_Dataset

# --- 1. Configuration ---
DATASET_ROOT_PATH = os.path.expanduser('~/projects/def-bereyhia/zjxeric/data/LEVIR_CD')
BASE_SAVE_DIR = "./results/levir_aernet_benchmark"


NUM_EPOCHS = 100
BATCH_SIZE = 16 
LEARNING_RATE = 1e-4
N_CLASSES = 2
IMG_SIZE = (512, 512) # Match the size used in LEVIR_Dataset

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- 2. Helper Functions (Metrics only) ---
def calculate_metrics(outputs, labels):
    preds = torch.argmax(outputs, dim=1) # Assumes model output is (Batch, Classes, H, W)
    preds_np, labels_np = preds.cpu().numpy().flatten(), labels.cpu().numpy().flatten()
    f1 = f1_score(labels_np, preds_np, average='binary', zero_division=0)
    iou = jaccard_score(labels_np, preds_np, average='binary', zero_division=0)
    return f1, iou

# --- 3. Main Training Function ---
def main():
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')
    args = parser.parse_args()
    

    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    SAVE_DIR = f"{BASE_SAVE_DIR}_seed{args.seed}"
    MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "best_aernet_benchmark_levir.pth")
    LOG_CSV_PATH = os.path.join(SAVE_DIR, "benchmark_aernet_levir_log.csv")
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Saving checkpoints and logs to: {SAVE_DIR}")

    # --- Data Loading (No Augmentation) ---
    print("Loading LEVIR-CD dataset...")
    train_dataset = LEVIR_Dataset(root_dir=DATASET_ROOT_PATH, split='train', output_size=IMG_SIZE, transforms=None)
    val_dataset = LEVIR_Dataset(root_dir=DATASET_ROOT_PATH, split='val', output_size=IMG_SIZE, transforms=None)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # --- Model, Optimizer, and Loss ---
    print("Initializing AERNet (zh_net) model...")
    model = zh_net().to(device) # Initialize AERNet
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # --- Using simple, unweighted CrossEntropyLoss ---
    criterion = nn.CrossEntropyLoss()
    
    # --- Training Loop ---
    with open(LOG_CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_iou', 'val_f1'])
    
    best_val_f1 = -1.0
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            # --- MODIFIED: Load both pre and post images ---
            pre_images, post_images, masks = batch['pre_image'].to(device), batch['post_image'].to(device), batch['mask'].to(device)
            squeezed_masks = masks.squeeze(1) # Remove channel dim: (B, 1, H, W) -> (B, H, W)
            
            optimizer.zero_grad()
            # --- MODIFIED: Pass both images to the model ---
            outputs_tuple = model(pre_images, post_images)
            
            # --- MODIFIED: Select the final output for loss ---
            # AERNet returns multiple outputs (al1,al2,al3,al4,result,seg)
            # 'result' seems to be the final refined output based on network.py
            final_output = outputs_tuple[4] 
            
            # --- Convert final_output to (Batch, Classes, H, W) for CrossEntropy ---

            change_prob = final_output.squeeze(1) # -> (B, H, W)
            # Create (B, 2, H, W) tensor for CrossEntropyLoss
            logits_output = torch.stack([(1 - change_prob), change_prob], dim=1) 
            
            loss = criterion(logits_output, squeezed_masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation Loop
        model.eval()
        val_loss, val_iou, val_f1 = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                pre_images, post_images, masks = batch['pre_image'].to(device), batch['post_image'].to(device), batch['mask'].to(device)
                
                outputs_tuple = model(pre_images, post_images)
                final_output = outputs_tuple[4]
                squeezed_masks = masks.squeeze(1)

                # Convert output for loss and metrics
                change_prob = final_output.squeeze(1) 
                logits_output = torch.stack([(1 - change_prob), change_prob], dim=1)

                loss = criterion(logits_output, squeezed_masks)
                val_loss += loss.item()
                
                # --- Use logits_output for metrics ---
                f1, iou = calculate_metrics(logits_output, squeezed_masks)
                val_f1 += f1
                val_iou += iou
                
        avg_val_loss, avg_val_iou, avg_val_f1 = val_loss / len(val_loader), val_iou / len(val_loader), val_f1 / len(val_loader)
        print(f"Epoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f} | Val F1: {avg_val_f1:.4f}")
        
        with open(LOG_CSV_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, avg_val_loss, avg_val_iou, avg_val_f1])
        
        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Validation F1-score improved to {best_val_f1:.4f}. Model saved.")

    print("\n--- LEVIR-CD AERNet Benchmark training finished! ---")

if __name__ == "__main__":
    main()