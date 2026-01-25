# Autoencoder Training

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import csv

# Make sure these files are in the same directory or accessible via sys.path
from src.data_loader.oscd_dataset_class import Manual_OSCD_Dataset
from src.model.OSCDAutoEncoder import OSCDAE

# --- 1. Configuration ---
# ==============================================================================
# Paths and Hyperparameters
DATASET_ROOT_PATH = os.path.expanduser('~/projects/def-bereyhia/zjxeric/data/Onera Satellite Change Detection')
MODEL_SAVE_PATH = "./model_weights"
LOG_CSV_PATH = os.path.join(MODEL_SAVE_PATH, "AEtraining_log.csv") # <-- Path for the log file
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)



NUM_EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
# ==============================================================================


def main():
    # --- 2. Setup ---
    # ==============================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # ==============================================================================


    # --- 3. Data Loading ---
    # ==============================================================================
    # We are using our robust Manual_OSCD_Dataset
    print("Loading datasets...")
    # Create training and validation datasets
    train_dataset = Manual_OSCD_Dataset(root_dir=DATASET_ROOT_PATH, split='train')
    
    # For validation, we'll use the 'test' split defined in test.txt
    val_dataset = Manual_OSCD_Dataset(root_dir=DATASET_ROOT_PATH, split='test')

    # Create DataLoaders
    # We use num_workers=0 as we know it's stable.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Data loaded. Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    # ==============================================================================
    

    # --- 4. Model, Optimizer, and Loss Function ---
    # ==============================================================================
    print("Initializing model...")
    autoencoder = OSCDAE(in_channels=12).to(device) # Or ModifiedAutoencoder().to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
    # The base loss for reconstruction
    criterion = nn.MSELoss()
    # ==============================================================================


    # --- 5. Training and Validation Loop ---
    # ==============================================================================
    best_val_loss = float('inf') # To save the best model

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        # --- Training Phase ---
        autoencoder.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc="Training"):
            # Our new DataLoader returns tensors in the correct format and shape
            pre_images = batch['pre_image'].to(device)
            post_images = batch['post_image'].to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # 1. Normal forward pass to get the standard reconstruction
            latent_change, reconstructed_post = autoencoder(pre_images, post_images)
            
            # This is the primary loss: how well did we reconstruct the post-image?
            reconstruction_loss = criterion(reconstructed_post, post_images)

            # 2. Adversarial forward pass (as you designed it)
            # Create a zero-tensor with the same shape as the pre-image
            zero_context_input = torch.zeros_like(pre_images)
            
            # Get the "context features" from this zero input. This should be ~zeros.
            zero_context_features = autoencoder.context_encoder(zero_context_input)
            
            # Combine the REAL change latent with the ZERO context
            adversarial_decoder_input = torch.cat((latent_change, zero_context_features), dim=1)
            
            # Try to reconstruct the post-image. This SHOULD fail (i.e., have high loss).
            adversarial_reconstruction = autoencoder.decoder(adversarial_decoder_input)
            
            # This is the loss from the adversarial case. We want this to be HIGH.
            adversarial_loss = criterion(adversarial_reconstruction, post_images)
            
            # 3. Your Final Combined Loss Function
            # loss = reconstruction_loss - 0.5 * adversarial_loss
            # This formula correctly MINIMIZES reconstruction_loss while MAXIMIZING adversarial_loss.
            # It pushes the model to reconstruct well with context, and poorly without it.
            final_loss = reconstruction_loss - 0.5 * adversarial_loss

            # Backward pass and optimization on the final combined loss
            final_loss.backward()
            optimizer.step()
            
            # Keep track of the total loss for reporting
            train_loss += final_loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validation Phase ---
        autoencoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                pre_images = batch['pre_image'].to(device)
                post_images = batch['post_image'].to(device)
                
                _, reconstructed = autoencoder(pre_images, post_images)
                loss = criterion(reconstructed, post_images)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1} Summary: Avg Train Loss: {avg_train_loss:.6f} | Avg Val Loss: {avg_val_loss:.6f}")

        # --- Append results to CSV Log File ---
        with open(LOG_CSV_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, avg_val_loss])
        
        # --- Save the best model ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(autoencoder.state_dict(), os.path.join(MODEL_SAVE_PATH, "oscd_autoencoder.pth"))
            print(f"Validation loss improved. Model saved to {MODEL_SAVE_PATH}/oscd_autoencoder.pth")
    # ==============================================================================

    print("\n--- Training finished! ---")


if __name__ == "__main__":
    main()