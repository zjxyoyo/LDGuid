import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F # Needed for interpolate
from tqdm import tqdm
import csv
from sklearn.metrics import f1_score, jaccard_score

import random
import numpy as np
from argparse import ArgumentParser

from models.aernet_network_latent import zh_net 


from models.ae_model import ChangeDetectionAE 
from data_loader.whu_patched_dataset_class import Patched_WHU_Dataset
from augmentation import JointCompose, JointRandomHorizontalFlip, JointRandomVerticalFlip, JointRandomRotation

# --- Import Augmentations if used ---
# from augmentation import JointCompose, JointRandomHorizontalFlip, JointRandomVerticalFlip, JointRandomRotation

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
class LatentAERNet(nn.Module):
    def __init__(self, pretrained_ae, aernet_model, compression_layer):
        super().__init__()
        self.autoencoder = pretrained_ae
        self.aernet_model = aernet_model
        self.compression = compression_layer
        
        self.autoencoder.eval()
        for param in self.autoencoder.parameters():
            param.requires_grad = False

    def forward(self, pre_image, post_image):
        with torch.no_grad():
            latent_change, _ = self.autoencoder(pre_image, post_image)
        compressed_latent = self.compression(latent_change)

        features_A = self.aernet_model.encoder(pre_image)
        features_B = self.aernet_model.encoder(post_image)
        
        deep_A = features_A[-1] 
        deep_B = features_B[-1]

        aligned_latent = F.interpolate(compressed_latent, size=deep_A.shape[2:], mode='bilinear', align_corners=False)
        
        enriched_deep_A = torch.cat((deep_A, aligned_latent), dim=1)
        enriched_deep_B = torch.cat((deep_B, aligned_latent), dim=1)

        features_A[-1] = enriched_deep_A
        features_B[-1] = enriched_deep_B
        
        output_tuple = self.aernet_model.decoder(features_A, features_B)
        return output_tuple

# ===================================================================
# 2. Configuration 
# ===================================================================
DATASET_ROOT_PATH = os.path.expanduser('~/projects/def-bereyhia/zjxeric/data/WHU-CD-Patches')
# --- IMPORTANT: Update this path to your AE weights trained on WHU-CD ---
AE_WEIGHTS_PATH = "/home/zjxeric/projects/def-bereyhia/zjxeric/AERNet/weights/best_ae_whu.pth" 
BASE_SAVE_DIR = "./results/whu_aernet_latent"

NUM_EPOCHS = 150 # Match benchmark script
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
IMG_SIZE = (512, 512) # Match dataset loader
IMG_CHANNELS = 3      # WHU-CD is RGB
AE_IN_CHANNELS = 3
AE_LATENT_CHANNELS = 128 
LATENT_CHANNELS_COMPRESSED = 64 
AERNET_DEEPEST_CHANNELS = 512
DECODER_INPUT_CHANNELS = (AERNET_DEEPEST_CHANNELS + LATENT_CHANNELS_COMPRESSED) * 2
N_CLASSES = 2

# ===================================================================
# 3. Helper Functions (Metrics only) 
# ===================================================================
def calculate_metrics(outputs, labels):
    preds = torch.argmax(outputs, dim=1)
    preds_np, labels_np = preds.cpu().numpy().flatten(), labels.cpu().numpy().flatten()
    f1 = f1_score(labels_np, preds_np, average='binary', zero_division=0)
    iou = jaccard_score(labels_np, preds_np, average='binary', zero_division=0)
    return f1, iou

# ===================================================================
# 4. Main Training Function 
# ===================================================================
def main():
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    SAVE_DIR = f"{BASE_SAVE_DIR}_seed{args.seed}"
    MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "best_aernet_latent_whu.pth")
    LOG_CSV_PATH = os.path.join(SAVE_DIR, "log_aernet_latent_whu.csv")
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Saving checkpoints and logs to: {SAVE_DIR}")

    # --- Data Loading (Using Patched_WHU_Dataset, optional augmentation) ---
    print("Loading Patched WHU-CD dataset...")
    
    # Define augmentations if needed (keep consistent with benchmark)
    #train_transforms = None # Or use the same JointCompose as benchmark script
    train_transforms = JointCompose([
        JointRandomHorizontalFlip(p=0.5),
        JointRandomVerticalFlip(p=0.5),
        JointRandomRotation(degrees=[0, 90, 180, 270]),
    ])

    train_dataset = Patched_WHU_Dataset(root_dir=DATASET_ROOT_PATH, split='train', output_size=IMG_SIZE, transforms=train_transforms)
    val_dataset = Patched_WHU_Dataset(root_dir=DATASET_ROOT_PATH, split='test', output_size=IMG_SIZE, transforms=None) 
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"Data loaded. Train: {len(train_dataset)}, Validation: {len(val_dataset)}")
    
    # --- Model Initialization ---
    print("Initializing LatentAERNet model...")
    autoencoder = ChangeDetectionAE(in_channels=AE_IN_CHANNELS)
    autoencoder.load_state_dict(torch.load(AE_WEIGHTS_PATH, map_location=device))
    
    aernet_model = zh_net(decoder_input_channels=DECODER_INPUT_CHANNELS).to(device)
    compression_layer = nn.Conv2d(AE_LATENT_CHANNELS, LATENT_CHANNELS_COMPRESSED, kernel_size=1).to(device)

    model = LatentAERNet(autoencoder, aernet_model, compression_layer).to(device)

    # --- Optimizer and Loss (Consistent with benchmark) ---
    optimizer = optim.Adam(
        list(model.aernet_model.parameters()) + list(model.compression.parameters()), # Only train AERNet and compression
        lr=LEARNING_RATE, 
        weight_decay=1e-5 # Added weight decay like benchmark
    )
    criterion = nn.CrossEntropyLoss()
    
    # --- Training Loop (Structure identical to benchmark) ---
    with open(LOG_CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_iou', 'val_f1'])
    
    best_val_f1 = 0.0 # Start at 0.0 like benchmark
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            pre_images, post_images, masks = batch['pre_image'].to(device), batch['post_image'].to(device), batch['mask'].to(device)
            squeezed_masks = masks.squeeze(1)
            
            optimizer.zero_grad()
            outputs_tuple = model(pre_images, post_images)
            final_output = outputs_tuple[4] 

            change_prob = final_output.squeeze(1) 
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

                change_prob = final_output.squeeze(1) 
                logits_output = torch.stack([(1 - change_prob), change_prob], dim=1)

                loss = criterion(logits_output, squeezed_masks)
                val_loss += loss.item()
                
                f1, iou = calculate_metrics(logits_output, squeezed_masks)
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
                'aernet_state_dict': model.aernet_model.state_dict(),
                'compression_state_dict': model.compression.state_dict()
            }, MODEL_SAVE_PATH)
            print(f"Validation F1-score improved to {best_val_f1:.4f}. Model saved.")

    print("\n--- WHU-CD AERNet Latent Feature training finished! ---")

if __name__ == "__main__":
    main()