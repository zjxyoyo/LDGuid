import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import tifffile
import csv

"""
All in one training script for Autoencoder focused on NBR Channels.
"""

# ===================================================================
# 1. AE
# ===================================================================
class ChangeDetectionAE(nn.Module):
    def __init__(self, in_channels=1): # Default to 1 for single-channel NBR
        super().__init__()
        
        # Change encoder takes 2 * in_channels
        self.change_encoder = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        # Context encoder takes in_channels
        self.context_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder takes the concatenated features
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, in_channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, pre_image, post_image):
        latent_change = self.change_encoder(torch.cat((pre_image, post_image), dim=1))
        context_features = self.context_encoder(pre_image)
        combined_features_for_decoder = torch.cat((latent_change, context_features), dim=1)
        reconstructed_post = self.decoder(combined_features_for_decoder)
        return latent_change, reconstructed_post

# ===================================================================
# 2. Custom Dataset for NBR Calculation
# ===================================================================
class CaBuAr_NBR_Dataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        self.split = split
        list_path = os.path.join(root_dir, "list", f"{split}.txt")
        with open(list_path, 'r') as f:
            self.filenames = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.filenames)

    def calculate_nbr(self, image_12_channel):
        """Calculates NBR from a 12-channel Sentinel-2 image."""
        # For Sentinel-2, NIR is Band 8 (index 7) and SWIR is Band 12 (index 11)
        nir = image_12_channel[:, :, 7].astype(np.float32)
        swir = image_12_channel[:, :, 11].astype(np.float32)
        
        # Add a small epsilon to avoid division by zero
        numerator = nir - swir
        denominator = nir + swir + 1e-8
        
        nbr = numerator / denominator
        # Normalize NBR from [-1, 1] to [0, 1] for the Sigmoid output of the AE
        nbr_normalized = (nbr + 1) / 2.0
        return nbr_normalized

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        pre_path = os.path.join(self.root_dir, "A", filename)
        post_path = os.path.join(self.root_dir, "B", filename)
        
        pre_img_12ch = tifffile.imread(pre_path)
        post_img_12ch = tifffile.imread(post_path)
        
        pre_nbr = self.calculate_nbr(pre_img_12ch)
        post_nbr = self.calculate_nbr(post_img_12ch)
        
        # Add channel dimension: (H, W) -> (1, H, W)
        pre_nbr_tensor = torch.from_numpy(pre_nbr).unsqueeze(0).float()
        post_nbr_tensor = torch.from_numpy(post_nbr).unsqueeze(0).float()

        return {'pre_nbr': pre_nbr_tensor, 'post_nbr': post_nbr_tensor}

# ===================================================================
# 3. Configuration
# ===================================================================
DATASET_ROOT_PATH = "/home/zjxeric/projects/def-bereyhia/zjxeric/data/CaBuAr_12ch"
SAVE_DIR = "./results/fire_ae_nbr"
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "best_ae_fire_nbr.pth")
LOG_CSV_PATH = os.path.join(SAVE_DIR, "log_ae_fire_nbr.csv")
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
IN_CHANNELS = 1 # We are using 1-channel NBR images

# ===================================================================
# 4. Main Training Function
# ===================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    print("Loading and preparing CaBuAr NBR dataset...")
    train_dataset = CaBuAr_NBR_Dataset(root_dir=DATASET_ROOT_PATH, split='train')
    val_dataset = CaBuAr_NBR_Dataset(root_dir=DATASET_ROOT_PATH, split='val')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # --- Model Initialization ---
    print("Initializing models...")
    model = ChangeDetectionAE(in_channels=IN_CHANNELS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss() 
    # --- Training Loop ---
    with open(LOG_CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss'])
    
    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
    # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            pre_nbr = batch['pre_nbr'].to(device)
            post_nbr = batch['post_nbr'].to(device)
            
            optimizer.zero_grad()
            
            # --- Custom Adversarial Loss Logic ---
            # 1. Standard forward pass and reconstruction loss
            latent_change, reconstructed_post = model(pre_nbr, post_nbr)
            reconstruction_loss = criterion(reconstructed_post, post_nbr)

            # 2. Adversarial pass with zeroed-out context
            zero_context_input = torch.zeros_like(pre_nbr)
            zero_context_features = model.context_encoder(zero_context_input)
            
            # Combine REAL change with FAKE context (detach latent_change)
            adversarial_decoder_input = torch.cat((latent_change.detach(), zero_context_features), dim=1)
            adversarial_reconstruction = model.decoder(adversarial_decoder_input)
            
            # 3. Calculate adversarial loss (we want this to be HIGH)
            adversarial_loss = criterion(adversarial_reconstruction, post_nbr)
            
            # 4. Final loss: minimize reconstruction_loss, maximize adversarial_loss
            final_loss = reconstruction_loss - 0.5 * adversarial_loss
            
            final_loss.backward()
            optimizer.step()
            
            train_loss += final_loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                pre_nbr = batch['pre_nbr'].to(device)
                post_nbr = batch['post_nbr'].to(device)
                _, reconstructed_post = model(pre_nbr, post_nbr)
                loss = criterion(reconstructed_post, post_nbr)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1} Summary: Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        with open(LOG_CSV_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, avg_val_loss])
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Validation loss improved to {best_val_loss:.6f}. Model saved.")

    print("\n--- NBR Autoencoder training finished! ---")

if __name__ == "__main__":
    main()