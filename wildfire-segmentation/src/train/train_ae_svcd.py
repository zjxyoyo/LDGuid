import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv

# Import the new dataset class and the updated AE model
from src.data_loader.svcd_dataset_class import SVCD_Dataset
from src.model.ae_model import ChangeDetectionAE 

# --- 1. Configuration ---
DATASET_ROOT_PATH = os.path.expanduser('~/projects/def-bereyhia/zjxeric/data/SVCD_dataset')
SAVE_DIR = "./results/svcd_autoencoder"
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "best_ae_svcd.pth")
LOG_CSV_PATH = os.path.join(SAVE_DIR, "ae_svcd_log.csv")
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_EPOCHS = 100 
BATCH_SIZE = 16  # larger size for ae training
LEARNING_RATE = 1e-4
# ==============================================================================


def main():
    # --- 2. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- 3. Data Loading ---
    print("Loading SVCD dataset...")
    # Using our new SVCD_Dataset class
    train_dataset = SVCD_Dataset(root_dir=DATASET_ROOT_PATH, split='train')
    val_dataset = SVCD_Dataset(root_dir=DATASET_ROOT_PATH, split='val') # SVCD has a 'val' split

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Data loaded. Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # --- 4. Model, Optimizer, and Loss Function ---
    print("Initializing model...")
    # ** Instantiate the model with in_channels=3 **
    autoencoder = ChangeDetectionAE(in_channels=3).to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # --- 5. Training and Validation Loop ---
    with open(LOG_CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_recon_loss'])
    
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        # --- Training Phase ---
        autoencoder.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            pre_images = batch['pre_image'].to(device)
            post_images = batch['post_image'].to(device)
            
            optimizer.zero_grad()
            
            # custom adversarial loss logic
            latent_change, reconstructed_post = autoencoder(pre_images, post_images)
            reconstruction_loss = criterion(reconstructed_post, post_images)

            zero_context_input = torch.zeros_like(pre_images)
            zero_context_features = autoencoder.context_encoder(zero_context_input)
            adversarial_decoder_input = torch.cat((latent_change.detach(), zero_context_features), dim=1)
            adversarial_reconstruction = autoencoder.decoder(adversarial_decoder_input)
            adversarial_loss = criterion(adversarial_reconstruction, post_images)
            
            final_loss = reconstruction_loss - 0.5 * adversarial_loss
            final_loss.backward()
            optimizer.step()
            
            train_loss += final_loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validation Phase ---
        autoencoder.eval()
        val_recon_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                pre_images = batch['pre_image'].to(device)
                post_images = batch['post_image'].to(device)
                
                _, reconstructed = autoencoder(pre_images, post_images)
                loss = criterion(reconstructed, post_images)
                val_recon_loss += loss.item()
        
        avg_val_recon_loss = val_recon_loss / len(val_loader)
        
        print(f"Epoch {epoch+1} Summary: Avg Train Loss: {avg_train_loss:.6f} | Avg Val Recon Loss: {avg_val_recon_loss:.6f}")
        
        with open(LOG_CSV_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, avg_val_recon_loss])
        
        if avg_val_recon_loss < best_val_loss:
            best_val_loss = avg_val_recon_loss
            torch.save(autoencoder.state_dict(), MODEL_SAVE_PATH)
            print(f"Validation loss improved. Model saved to {MODEL_SAVE_PATH}")

    print("\n--- AE training on SVCD finished! ---")

if __name__ == "__main__":
    main()