import os
import sys 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
import numpy as np

from src.data_loader.svcd_dataset_class import SVCD_Dataset
from src.model.ae_model import ChangeDetectionAE 

# --- 1. Configuration ---
DATASET_ROOT_PATH = os.path.expanduser('~/projects/def-bereyhia/zjxeric/data/SVCD_dataset')
BASE_SAVE_DIR = "./results/svcd_autoencoder_sweep"
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

NUM_EPOCHS = 100 
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

# --- MODIFIED: Read the adversarial weight from the command line ---
try:
    adv_weight = float(sys.argv[1])
except IndexError:
    print("ERROR: No adversarial weight provided.")
    print("Usage: python -m src.ablation.ae_svcd_single_run <alpha_value>")
    sys.exit(1)
# ==============================================================================


def main(current_adv_weight):
    # --- 2. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- 3. Data Loading ---
    print("Loading SVCD dataset...")
    train_dataset = SVCD_Dataset(root_dir=DATASET_ROOT_PATH, split='train')
    val_dataset = SVCD_Dataset(root_dir=DATASET_ROOT_PATH, split='val')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"Data loaded. Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # --- 4. This is now a single run, no outer loop ---
    print(f"\n{'='*20} TESTING ADVERSARIAL WEIGHT: {current_adv_weight} {'='*20}")

    # --- Create specific save directories ---
    SAVE_DIR = os.path.join(BASE_SAVE_DIR, f"alpha_{current_adv_weight}")
    MODEL_SAVE_PATH = os.path.join(SAVE_DIR, f"best_ae_svcd_alpha_{current_adv_weight}.pth")
    LOG_CSV_PATH = os.path.join(SAVE_DIR, f"ae_svcd_log_alpha_{current_adv_weight}.csv")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # --- Initialize Model, Optimizer, and Loss ---
    autoencoder = ChangeDetectionAE(in_channels=3).to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    with open(LOG_CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_recon_loss'])
    
    best_val_loss = float('inf')
    best_train_loss_for_best_val = 0.0

    for epoch in range(NUM_EPOCHS):
        # --- Training Phase ---
        autoencoder.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Training (Alpha={current_adv_weight})", leave=False):
            pre_images = batch['pre_image'].to(device)
            post_images = batch['post_image'].to(device)
            
            optimizer.zero_grad()
            
            latent_change, reconstructed_post = autoencoder(pre_images, post_images)
            reconstruction_loss = criterion(reconstructed_post, post_images)

            zero_context_input = torch.zeros_like(pre_images)
            zero_context_features = autoencoder.context_encoder(zero_context_input)
            adversarial_decoder_input = torch.cat((latent_change.detach(), zero_context_features), dim=1)
            adversarial_reconstruction = autoencoder.decoder(adversarial_decoder_input)
            adversarial_loss = criterion(adversarial_reconstruction, post_images)
            
            # --- Use the current adversarial weight ---
            final_loss = reconstruction_loss - current_adv_weight * adversarial_loss
            final_loss.backward()
            optimizer.step()
            
            train_loss += final_loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validation Phase ---
        autoencoder.eval()
        val_recon_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validating (Alpha={current_adv_weight})", leave=False):
                pre_images = batch['pre_image'].to(device)
                post_images = batch['post_image'].to(device)
                
                _, reconstructed = autoencoder(pre_images, post_images)
                loss = criterion(reconstructed, post_images)
                val_recon_loss += loss.item()
        
        avg_val_recon_loss = val_recon_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} [Alpha={current_adv_weight}]: Train Loss: {avg_train_loss:.6f} | Val Recon Loss: {avg_val_recon_loss:.6f}")
        
        with open(LOG_CSV_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, avg_val_recon_loss])
        
        if avg_val_recon_loss < best_val_loss:
            best_val_loss = avg_val_recon_loss
            best_train_loss_for_best_val = avg_train_loss 
            torch.save(autoencoder.state_dict(), MODEL_SAVE_PATH)
            # print(f"Validation loss improved. Model saved.")

    print(f"--- Finished training for alpha={current_adv_weight}. Best Val Loss: {best_val_loss:.6f} (Train Loss: {best_train_loss_for_best_val:.6f}) ---")

if __name__ == "__main__":
    # --- MODIFIED: Pass the command-line argument to main ---
    main(current_adv_weight=adv_weight)