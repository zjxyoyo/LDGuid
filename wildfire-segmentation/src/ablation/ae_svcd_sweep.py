import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
import numpy as np

# Import the new dataset class and the updated AE model
from src.data_loader.svcd_dataset_class import SVCD_Dataset
from src.model.ae_model import ChangeDetectionAE 

# --- 1. Configuration ---
DATASET_ROOT_PATH = os.path.expanduser('~/projects/def-bereyhia/zjxeric/data/SVCD_dataset')
BASE_SAVE_DIR = "./results/svcd_autoencoder_sweep"
SUMMARY_CSV_PATH = os.path.join(BASE_SAVE_DIR, "sweep_summary.csv") # Path for the final summary
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

NUM_EPOCHS = 100 
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

# --- List of adversarial weights (alpha) to test ---
ADVERSARIAL_WEIGHTS = np.array([0.00001, 0.001, 0.01, 1.2, 1.5, 2, 5]) # [0.1, 0.2, ..., 0.9]
# ==============================================================================


def main():
    # --- 2. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- 3. Data Loading ---
    print("Loading SVCD dataset...")
    train_dataset = SVCD_Dataset(root_dir=DATASET_ROOT_PATH, split='train')
    val_dataset = SVCD_Dataset(root_dir=DATASET_ROOT_PATH, split='val')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Data loaded. Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # --- Dictionary to store results ---
    results_summary = {}

    # --- 4. Loop over Adversarial Weights ---
    for adv_weight in ADVERSARIAL_WEIGHTS:

        print(f"\n{'='*20} TESTING ADVERSARIAL WEIGHT: {adv_weight} {'='*20}")

        # --- Create specific save directories ---
        SAVE_DIR = os.path.join(BASE_SAVE_DIR, f"alpha_{adv_weight}")
        MODEL_SAVE_PATH = os.path.join(SAVE_DIR, f"best_ae_svcd_alpha_{adv_weight}.pth")
        LOG_CSV_PATH = os.path.join(SAVE_DIR, f"ae_svcd_log_alpha_{adv_weight}.csv")
        os.makedirs(SAVE_DIR, exist_ok=True)

        # --- Re-initialize Model, Optimizer, and Loss for each run ---
        autoencoder = ChangeDetectionAE(in_channels=3).to(device)
        optimizer = optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()
        
        with open(LOG_CSV_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_recon_loss'])
        
        best_val_loss = float('inf')
        # --- Variable to store the train loss from the best epoch ---
        best_train_loss_for_best_val = 0.0

        for epoch in range(NUM_EPOCHS):
            # --- Training Phase ---
            autoencoder.train()
            train_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Training (Alpha={adv_weight})", leave=False):
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
                final_loss = reconstruction_loss - adv_weight * adversarial_loss
                final_loss.backward()
                optimizer.step()
                
                train_loss += final_loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # --- Validation Phase ---
            autoencoder.eval()
            val_recon_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Validating (Alpha={adv_weight})", leave=False):
                    pre_images = batch['pre_image'].to(device)
                    post_images = batch['post_image'].to(device)
                    
                    _, reconstructed = autoencoder(pre_images, post_images)
                    loss = criterion(reconstructed, post_images)
                    val_recon_loss += loss.item()
            
            avg_val_recon_loss = val_recon_loss / len(val_loader)
            
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} [Alpha={adv_weight}]: Train Loss: {avg_train_loss:.6f} | Val Recon Loss: {avg_val_recon_loss:.6f}")
            
            with open(LOG_CSV_PATH, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, avg_train_loss, avg_val_recon_loss])
            
            if avg_val_recon_loss < best_val_loss:
                best_val_loss = avg_val_recon_loss
                # --- Store the corresponding train loss ---
                best_train_loss_for_best_val = avg_train_loss 
                torch.save(autoencoder.state_dict(), MODEL_SAVE_PATH)
                # print(f"Validation loss improved. Model saved.") # Quieter logging

        # --- Store both results in the summary ---
        results_summary[adv_weight] = (best_val_loss, best_train_loss_for_best_val)
        print(f"--- Finished training for alpha={adv_weight}. Best Val Loss: {best_val_loss:.6f} (Train Loss: {best_train_loss_for_best_val:.6f}) ---")

    # --- 5. Print and Save Final Summary  ---
    print("\n{'='*20} HYPERPARAMETER SWEEP SUMMARY {'='*20}")
    print("Alpha | Best Val Recon Loss | Corresponding Train Loss")
    print("-" * 65)
    
    # Sort results by alpha for clarity
    sorted_alphas = sorted(results_summary.keys())
    
    with open(SUMMARY_CSV_PATH, 'w', newline='') as f_summary:
        writer = csv.writer(f_summary)
        # --- MODIFIED: Update header ---
        header = ["Adversarial_Weight_Alpha", "Best_Validation_Recon_Loss", "Corresponding_Train_Loss"]
        writer.writerow(header)
        
        for alpha in sorted_alphas:
            # --- MODIFIED: Unpack both values ---
            best_val, best_train = results_summary[alpha]
            print(f" {alpha:<7} |      {best_val:.6f}     |         {best_train:.6f}")
            writer.writerow([f"{alpha}", f"{best_val:.6f}", f"{best_train:.6f}"])
            
    print("=" * 65)
    print(f"Summary results saved to: {SUMMARY_CSV_PATH}")
    print("\n--- AE hyperparameter sweep on SVCD finished! ---")

if __name__ == "__main__":
    main()