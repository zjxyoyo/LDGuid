import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv


from models.fire_networks import BASE_Transformer 
from models.losses import cross_entropy
from datasets.fire_dataset import FireDataset 
from misc.metric_tool import ConfuseMatrixMeter



# ===================================================================
# Configuration 
# ===================================================================
# --- Paths ---
DATASET_ROOT_PATH = "/home/zjxeric/projects/def-bereyhia/zjxeric/data/CaBuAr_12ch" 
# NOTE: AE_WEIGHTS_PATH is removed.
SAVE_DIR = "./checkpoints/CD_Benchmark_Fire"
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "best_benchmark_fire.pth")
LOG_CSV_PATH = os.path.join(SAVE_DIR, "log_benchmark_fire.csv")
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Hyperparameters ---
NUM_EPOCHS = 200
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
IMG_SIZE = 256
IMG_CHANNELS = 12 # 
BACKBONE_CHANNELS = 32 
N_CLASSES = 2
GPU_ID = 0

# ===================================================================
# Main Training Function 
# ===================================================================
def main():
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    print("Loading CaBuAr (Fire) 12-channel dataset...")
    train_dataset = FireDataset(root_dir=DATASET_ROOT_PATH, split='train', img_size=IMG_SIZE, is_train=True, label_transform='norm')
    val_dataset = FireDataset(root_dir=DATASET_ROOT_PATH, split='val', img_size=IMG_SIZE, is_train=False, label_transform='norm')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6)

    # --- Model Initialization (MODIFIED for Benchmark) ---
    print("Initializing benchmark BIT model...")
    # The model is just the BASE_Transformer itself, no wrapper.
    model = BASE_Transformer(input_nc=IMG_CHANNELS, output_nc=N_CLASSES, with_pos='learned', 
                             resnet_stages_num=4).to(device)

    # --- Optimizer (MODIFIED for Benchmark) ---
    # The optimizer now only trains the main model's parameters.
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = cross_entropy

    # --- Logging and Training Loop (Identical to your other script) ---
    with open(LOG_CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_iou', 'val_f1'])
    
    metric_calculator = ConfuseMatrixMeter(n_class=N_CLASSES)
    best_val_f1 = -1.0

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            pre_images, post_images, masks = batch['A'].to(device), batch['B'].to(device), batch['L'].to(device)
            
            optimizer.zero_grad()
            outputs = model(pre_images, post_images) 
            loss = criterion(outputs, masks.long())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        metric_calculator.clear()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                pre_images, post_images, masks = batch['A'].to(device), batch['B'].to(device), batch['L'].to(device)
                
                outputs = model(pre_images, post_images)
                loss = criterion(outputs, masks.long())
                val_loss += loss.item()
                
                pred_masks = torch.argmax(outputs, dim=1)
                metric_calculator.update_cm(pr=pred_masks.cpu().numpy(), gt=masks.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_metrics = metric_calculator.get_scores()
        val_f1 = val_metrics['mf1']
        val_iou = val_metrics['miou']

        print(f"Epoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val IoU: {val_iou:.4f} | Val F1: {val_f1:.4f}")
        
        with open(LOG_CSV_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, avg_val_loss, val_iou, val_f1])
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # Save the state_dict of the model directly.
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Validation F1-score improved to {best_val_f1:.4f}. Model saved.")

    print(f"\n--- Training finished! Best F1: {best_val_f1:.4f} ---")

if __name__ == "__main__":
    main()