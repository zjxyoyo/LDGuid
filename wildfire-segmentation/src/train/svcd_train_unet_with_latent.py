# train_unet_with_latent_svcd.py

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
from sklearn.metrics import f1_score, jaccard_score


from src.data_loader.svcd_dataset_class import SVCD_Dataset
from src.model.unet import UNet
from src.model.ae_model import ChangeDetectionAE 


# --- Wrapper Model to combine AE and U-Net ---
class LatentUNet(nn.Module):
    def __init__(self, pretrained_ae, unet_model, latent_compression_layer):
        super().__init__()
        self.autoencoder = pretrained_ae
        self.unet = unet_model
        self.compression = latent_compression_layer
        
        # Freeze the Autoencoder weights - we use it only as a feature extractor
        self.autoencoder.eval()
        for param in self.autoencoder.parameters():
            param.requires_grad = False

    def forward(self, pre_image, post_image):
        # 1. Extract latent feature with the frozen AE
        with torch.no_grad():
            latent_change, _ = self.autoencoder(pre_image, post_image)

        # 2. Compress the latent feature
        compressed_latent = self.compression(latent_change)
        
        # 3. Upsample latent feature to match the post-image size
        upsampled_latent = F.interpolate(compressed_latent, size=post_image.shape[2:], mode='bilinear', align_corners=False)
        
        # 4. Concatenate post-image and latent feature
        combined_input = torch.cat((post_image, upsampled_latent), dim=1)
        
        # 5. Feed the combined input to the U-Net
        output = self.unet(combined_input)
        
        return output

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

# --- Configuration ---
# ==============================================================================
# --- Paths ---
DATASET_ROOT_PATH = os.path.expanduser('~/projects/def-bereyhia/zjxeric/data/SVCD_dataset')
AE_WEIGHTS_PATH = "./results/svcd_autoencoder/best_ae_svcd.pth" 
# Default to seed 42 if no argument is given
try:
    # Read the seed from the first command-line argument
    SEED = int(sys.argv[1])
except (IndexError, ValueError):
    print("No seed argument provided. Defaulting to 42.")
    SEED = 42
SAVE_DIR = f"./results/svcd_unet_latent_seed{SEED}"
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "best_unet_with_latent_svcd.pth")
LOG_CSV_PATH = os.path.join(SAVE_DIR, "latent_svcd_log.csv")
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Hyperparameters ---
NUM_EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

# --- Model Parameters ---
IMG_CHANNELS = 3
AE_LATENT_CHANNELS = 128      # The channel output of your AE's encoder
LATENT_CHANNELS_COMPRESSED = 4 # The desired compressed channel size
UNET_INPUT_CHANNELS = IMG_CHANNELS + LATENT_CHANNELS_COMPRESSED # -> 7
N_CLASSES = 2
# ==============================================================================

# --- Helper Functions ---
def calculate_metrics(outputs, labels):
    preds = torch.argmax(outputs, dim=1)
    preds_np, labels_np = preds.cpu().numpy().flatten(), labels.cpu().numpy().flatten()
    f1 = f1_score(labels_np, preds_np, average='binary', zero_division=0)
    iou = jaccard_score(labels_np, preds_np, average='binary', zero_division=0)
    return f1, iou

# --- Main Training Function ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading (No Augmentation) ---
    print("Loading SVCD dataset...")
    train_dataset = SVCD_Dataset(root_dir=DATASET_ROOT_PATH, split='train', transforms=None)
    val_dataset = SVCD_Dataset(root_dir=DATASET_ROOT_PATH, split='val', transforms=None)

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

    # --- Model Initialization ---
    print("Initializing models...")
    # 1. Load the pre-trained Autoencoder
    autoencoder = ChangeDetectionAE(in_channels=IMG_CHANNELS)
    autoencoder.load_state_dict(torch.load(AE_WEIGHTS_PATH, map_location=device))
    
    # 2. Create a new U-Net with 7 input channels
    unet = UNet(n_channels=UNET_INPUT_CHANNELS, n_classes=N_CLASSES, enable_dropout=True)
    
    # 3. Create the compression layer (128 -> 4 channels)
    compression_layer = nn.Conv2d(AE_LATENT_CHANNELS, LATENT_CHANNELS_COMPRESSED, kernel_size=1)

    # 4. Assemble the final "wrapper" model
    model = LatentUNet(autoencoder, unet, compression_layer).to(device)
    
    # 5. Define the optimizer to train ONLY the U-Net and the new compression layer
    optimizer = optim.Adam(
        list(model.unet.parameters()) + list(model.compression.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-5
    )
    
    # 6. Define the simple, unweighted loss function
    criterion = nn.CrossEntropyLoss()
    
    # --- Training Loop ---
    with open(LOG_CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_iou', 'val_f1'])
    
    best_val_f1 = 0.0
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        model.train()
        model.unet.train()
        model.compression.train()

        train_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            pre_images = batch['pre_image'].to(device)
            post_images = batch['post_image'].to(device)
            masks = batch['mask'].to(device)
            squeezed_masks = masks.squeeze(1)
            
            optimizer.zero_grad()
            outputs = model(pre_images, post_images)
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
                pre_images = batch['pre_image'].to(device)
                post_images = batch['post_image'].to(device)
                masks = batch['mask'].to(device)
                squeezed_masks = masks.squeeze(1)
                
                outputs = model(pre_images, post_images)
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
            torch.save({
                'unet_state_dict': model.unet.state_dict(),
                'compression_state_dict': model.compression.state_dict()
            }, MODEL_SAVE_PATH)
            print(f"Validation F1-score improved to {best_val_f1:.4f}. Model saved.")

    print("\n--- Final Experiment Training Finished! ---")

if __name__ == "__main__":
    main()