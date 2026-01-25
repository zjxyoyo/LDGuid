import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
import random
import numpy as np
from argparse import ArgumentParser

# --- BIT_CD Project Imports ---
from models.networks_new import define_G, BASE_Transformer
from models.losses import cross_entropy

from datasets.CD_dataset import CDDataset
from misc.metric_tool import ConfuseMatrixMeter


from models.ae_model import ChangeDetectionAE 

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

    def forward(self, pre_image, post_image):
        # Generate latent feature from your AE
        with torch.no_grad():
            latent_change, _ = self.autoencoder(pre_image, post_image)
        
        compressed_latent = self.compression(latent_change)

        # --- Use the hooks we created in the BIT model ---
        # Part 1: Pass images through the Siamese CNN backbone
        feature_A = self.bit_model.forward_single(pre_image)
        feature_B = self.bit_model.forward_single(post_image)
        
        # Align the latent feature's size
        aligned_latent = F.interpolate(compressed_latent, size=feature_A.shape[2:], mode='bilinear', align_corners=False)
        
        # Inject the latent feature by concatenating to both streams
        enriched_feature_A = torch.cat((feature_A, aligned_latent), dim=1)
        enriched_feature_B = torch.cat((feature_B, aligned_latent), dim=1)

        # Part 2: Pass the enriched features to the rest of the BIT model
        output = self.bit_model.forward_transformer(enriched_feature_A, enriched_feature_B)

        return output

# ===================================================================
# 2. Configuration
# ===================================================================
# --- Paths ---
DATASET_ROOT_PATH = "/home/zjxeric/projects/def-bereyhia/zjxeric/data/WHU_CD" 
AE_WEIGHTS_PATH = "/home/zjxeric/projects/def-bereyhia/zjxeric/wildfire-segmentation/results/whu_autoencoder/best_ae_whu.pth" 
#/home/zjxeric/projects/def-bereyhia/zjxeric/wildfire-segmentation/results/levir_autoencoder/best_ae_levir.pth
#/home/zjxeric/projects/def-bereyhia/zjxeric/wildfire-segmentation/results/whu_autoencoder/best_ae_whu.pth
BASE_SAVE_DIR = "./checkpoints/CD_LatentBIT_WHU_CD_Experiment" #
# --- Hyperparameters ---
NUM_EPOCHS = 200
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
IMG_SIZE = 256
IMG_CHANNELS = 3
AE_LATENT_CHANNELS = 128
LATENT_CHANNELS_COMPRESSED = 4
BACKBONE_CHANNELS = 32 # The ResNet backbone in BIT outputs 32 channels
TOKENIZER_IN_CHANNELS = BACKBONE_CHANNELS + LATENT_CHANNELS_COMPRESSED # 32 + 4 = 36
N_CLASSES = 2
GPU_ID = 0

# ===================================================================
# 3. Main Training Function (Corrected Version)
# ===================================================================
def main():

    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')
    args = parser.parse_args()
    
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    SAVE_DIR = f"{BASE_SAVE_DIR}_seed{args.seed}"
    MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "best_latent_bit_whu_cd.pth")
    LOG_CSV_PATH = os.path.join(SAVE_DIR, "log_latent_bit_whu_cd.csv")
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Saving checkpoints and logs to: {SAVE_DIR}")

    # --- Data Loading (Using BIT_CD's loader) ---
    print("Loading WHU_CD dataset...")
    # The BIT data loader uses an 'args' object for configuration
    class Args:
        data_name = 'WHU_CD'
        img_size = IMG_SIZE
        split = 'train'
        split_val = 'val'
        batch_size = BATCH_SIZE
        num_workers = 6 # Adjust based on your system
        dataset = 'CDDataset'
        
    loader_args = Args()
    

    root_dir = DATASET_ROOT_PATH 
    label_transform = 'norm' # This was the default value in the config file

    train_dataset = CDDataset(root_dir=root_dir, split=loader_args.split, img_size=loader_args.img_size, is_train=True, label_transform=label_transform)
    val_dataset = CDDataset(root_dir=root_dir, split=loader_args.split_val, img_size=loader_args.img_size, is_train=False, label_transform=label_transform)


    train_loader = DataLoader(train_dataset, batch_size=loader_args.batch_size, shuffle=True, num_workers=loader_args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=loader_args.batch_size, shuffle=False, num_workers=loader_args.num_workers)

    # --- Model Initialization ---
    print("Initializing models...")
    autoencoder = ChangeDetectionAE(in_channels=IMG_CHANNELS)
    autoencoder.load_state_dict(torch.load(AE_WEIGHTS_PATH, map_location=device))
    
    class BitArgs:
        net_G='base_transformer_pos_s4_dd8'
        resnet_stages_num=4
        
    bit_args = BitArgs()
    bit_model = BASE_Transformer(input_nc=3, output_nc=N_CLASSES, with_pos='learned', 
                                 resnet_stages_num=bit_args.resnet_stages_num,
                                 tokenizer_in_channels=TOKENIZER_IN_CHANNELS,dim=TOKENIZER_IN_CHANNELS).to(device)

    compression_layer = nn.Conv2d(AE_LATENT_CHANNELS, LATENT_CHANNELS_COMPRESSED, kernel_size=1).to(device)

    model = LatentBIT(autoencoder, bit_model, compression_layer).to(device)

    # --- Optimizer and Loss ---
    optimizer = optim.Adam(
        list(model.bit_model.parameters()) + list(model.compression.parameters()),
        lr=LEARNING_RATE
    )
    criterion = cross_entropy

    # --- Setup Logging and Metrics ---
    with open(LOG_CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_iou', 'val_f1'])
    
    metric_calculator = ConfuseMatrixMeter(n_class=N_CLASSES)
    best_val_f1 = -1.0

    # --- Training & Validation Loop ---
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