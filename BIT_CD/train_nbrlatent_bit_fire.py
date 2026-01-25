import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv

# --- BIT_CD Project Imports ---
from models.fire_networks_latent import BASE_Transformer 
from models.losses import cross_entropy
from datasets.fire_dataset import FireDataset 
from misc.metric_tool import ConfuseMatrixMeter


from models.ae_model import ChangeDetectionAE 

# ===================================================================
# 1. Model Wrapper for Feature Injection 
# ===================================================================
class LatentBIT(nn.Module):
    def __init__(self, pretrained_ae, bit_model, compression_layer):
        super().__init__()
        self.autoencoder = pretrained_ae
        self.bit_model = bit_model
        self.compression = compression_layer
        
        self.autoencoder.eval()
        for param in self.autoencoder.parameters():
            param.requires_grad = False
            
    def _calculate_nbr(self, image_12_channel):
        """Calculates NBR from a 12-channel PyTorch tensor."""
        # For Sentinel-2, NIR is Band 8 (index 7) and SWIR is Band 12 (index 11)
        nir = image_12_channel[:, 7:8, :, :]  # Use slicing to keep channel dimension
        swir = image_12_channel[:, 11:12, :, :]
        
        numerator = nir - swir
        denominator = nir + swir + 1e-8
        
        nbr = numerator / denominator
        # Normalize NBR from [-1, 1] to [0, 1] for the AE
        return (nbr + 1) / 2.0

    def forward(self, pre_image_12ch, post_image_12ch):
        # --- On-the-fly NBR Calculation ---
        # The BIT model gets the full 12-channel images,
        # but the AE needs 1-channel NBR images. We create them here.
        pre_nbr = self._calculate_nbr(pre_image_12ch)
        post_nbr = self._calculate_nbr(post_image_12ch)
        
        # --- Generate Latent Feature using NBR images ---
        with torch.no_grad():
            latent_change, _ = self.autoencoder(pre_nbr, post_nbr)
        
        compressed_latent = self.compression(latent_change)

        # --- The rest of the logic uses the original 12-channel images ---
        feature_A = self.bit_model.forward_single(pre_image_12ch)
        feature_B = self.bit_model.forward_single(post_image_12ch)
        
        aligned_latent = F.interpolate(compressed_latent, size=feature_A.shape[2:], mode='bilinear', align_corners=False)
        
        enriched_feature_A = torch.cat((feature_A, aligned_latent), dim=1)
        enriched_feature_B = torch.cat((feature_B, aligned_latent), dim=1)

        output = self.bit_model.forward_transformer(enriched_feature_A, enriched_feature_B)
        return output

# ===================================================================
# 2. Configuration 
# ===================================================================
# --- Paths ---
DATASET_ROOT_PATH = "/home/zjxeric/projects/def-bereyhia/zjxeric/data/CaBuAr_12ch" 
# --- MODIFIED: Point to your new NBR-trained AE weights ---
AE_WEIGHTS_PATH = "/home/zjxeric/projects/def-bereyhia/zjxeric/BIT_CD/results/fire_ae_nbr/best_ae_fire_nbr.pth" 
SAVE_DIR = "./checkpoints/CD_LatentBIT_Fire_NBR_AE_Experiment"
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "best_latent_bit_fire_nbr_ae.pth")
LOG_CSV_PATH = os.path.join(SAVE_DIR, "log_latent_bit_fire_nbr_ae.csv")
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Hyperparameters ---
NUM_EPOCHS = 200
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
IMG_SIZE = 256
IMG_CHANNELS = 12 # The BIT model uses the full 12 channels
AE_IN_CHANNELS = 1 # The AE model was trained on 1-channel NBR
AE_LATENT_CHANNELS = 128 # The latent space from ChangeDetectionAE
LATENT_CHANNELS_COMPRESSED = 4
BACKBONE_CHANNELS = 32
TOKENIZER_IN_CHANNELS = BACKBONE_CHANNELS + LATENT_CHANNELS_COMPRESSED # 32 + 4 = 36
N_CLASSES = 2
GPU_ID = 0

# ===================================================================
# 3. Main Training Function 
# ===================================================================
def main():
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    # Use the FireDataset which loads the full 12-channel images
    print("Loading CaBuAr (Fire) 12-channel dataset...")
    train_dataset = FireDataset(root_dir=DATASET_ROOT_PATH, split='train', img_size=IMG_SIZE, is_train=True, label_transform='norm')
    val_dataset = FireDataset(root_dir=DATASET_ROOT_PATH, split='val', img_size=IMG_SIZE, is_train=False, label_transform='norm')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6)

    # --- Model Initialization ---
    print("Initializing models...")
    # Initialize the correct AE with correct channels (1 for NBR)
    autoencoder = ChangeDetectionAE(in_channels=AE_IN_CHANNELS)
    autoencoder.load_state_dict(torch.load(AE_WEIGHTS_PATH, map_location=device))
    
    # Initialize the 12-channel BIT model
    bit_model = BASE_Transformer(input_nc=IMG_CHANNELS, output_nc=N_CLASSES, with_pos='learned', 
                                 resnet_stages_num=4,
                                 tokenizer_in_channels=TOKENIZER_IN_CHANNELS,
                                 dim=TOKENIZER_IN_CHANNELS).to(device)

    # The compression layer takes the latent dimension from the AE
    compression_layer = nn.Conv2d(AE_LATENT_CHANNELS, LATENT_CHANNELS_COMPRESSED, kernel_size=1).to(device)

    # Create the final wrapper model
    model = LatentBIT(autoencoder, bit_model, compression_layer).to(device)

    # --- Optimizer, Loss, and Training Loop (No changes needed below) ---
    optimizer = optim.Adam(
        list(model.bit_model.parameters()) + list(model.compression.parameters()),
        lr=LEARNING_RATE
    )
    criterion = cross_entropy

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
            torch.save({
                'bit_model_state_dict': model.bit_model.state_dict(),
                'compression_state_dict': model.compression.state_dict()
            }, MODEL_SAVE_PATH)
            print(f"Validation F1-score improved to {best_val_f1:.4f}. Model saved.")

    print(f"\n--- Training finished! Best F1: {best_val_f1:.4f} ---")

if __name__ == "__main__":
    main()