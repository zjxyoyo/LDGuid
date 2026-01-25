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
from data_loader.levir_dataset_class import LEVIR_Dataset

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
        # Generate latent feature from your AE
        with torch.no_grad():
            latent_change, _ = self.autoencoder(pre_image, post_image)
        
        compressed_latent = self.compression(latent_change)

        # Pass images through the Siamese Encoder part of AERNet
        features_A = self.aernet_model.encoder(pre_image)
        features_B = self.aernet_model.encoder(post_image)
        
        # Extract the deepest features (output of layer4)
        deep_A = features_A[-1] # Last element from the list returned by ResNet
        deep_B = features_B[-1]

        # Align the latent feature's size
        aligned_latent = F.interpolate(compressed_latent, size=deep_A.shape[2:], mode='bilinear', align_corners=False)
        
        # Inject the latent feature by concatenating to both deep features
        enriched_deep_A = torch.cat((deep_A, aligned_latent), dim=1)
        enriched_deep_B = torch.cat((deep_B, aligned_latent), dim=1)

        # Replace the original deep features in the lists
        features_A[-1] = enriched_deep_A
        features_B[-1] = enriched_deep_B
        
        # Pass the modified feature lists to the AERNet decoder
        output_tuple = self.aernet_model.decoder(features_A, features_B)
        
        return output_tuple

# ===================================================================
# 2. Configuration 
# ===================================================================
DATASET_ROOT_PATH = os.path.expanduser('~/projects/def-bereyhia/zjxeric/data/LEVIR_CD')
AE_WEIGHTS_PATH = "/home/zjxeric/projects/def-bereyhia/zjxeric/AERNet/weights/best_ae_levir.pth"
BASE_SAVE_DIR = "./results/levir_aernet_latent"

NUM_EPOCHS = 100
BATCH_SIZE = 16 
LEARNING_RATE = 1e-4
N_CLASSES = 2
IMG_SIZE = (512, 512)
IMG_CHANNELS = 3
AE_IN_CHANNELS = 3
AE_LATENT_CHANNELS = 128 # Latent space from ChangeDetectionAE
LATENT_CHANNELS_COMPRESSED = 64 
AERNET_DEEPEST_CHANNELS = 512 # ResNet34 layer4 output channels

# Calculate the input channels for the modified decoder
DECODER_INPUT_CHANNELS = (AERNET_DEEPEST_CHANNELS + LATENT_CHANNELS_COMPRESSED) * 2

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
    MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "best_aernet_latent_levir.pth")
    LOG_CSV_PATH = os.path.join(SAVE_DIR, "log_aernet_latent_levir.csv")
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Saving checkpoints and logs to: {SAVE_DIR}")

    # --- Data Loading ---
    print("Loading LEVIR-CD dataset...")
    train_dataset = LEVIR_Dataset(root_dir=DATASET_ROOT_PATH, split='train', output_size=IMG_SIZE, transforms=None)
    val_dataset = LEVIR_Dataset(root_dir=DATASET_ROOT_PATH, split='val', output_size=IMG_SIZE, transforms=None)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # --- Model Initialization ---
    print("Initializing LatentAERNet model...")
    autoencoder = ChangeDetectionAE(in_channels=AE_IN_CHANNELS)
    autoencoder.load_state_dict(torch.load(AE_WEIGHTS_PATH, map_location=device))
    
    # Initialize the MODIFIED AERNet, passing the new channel count
    aernet_model = zh_net(decoder_input_channels=DECODER_INPUT_CHANNELS).to(device)
    
    compression_layer = nn.Conv2d(AE_LATENT_CHANNELS, LATENT_CHANNELS_COMPRESSED, kernel_size=1).to(device)

    # Create the final wrapper model
    model = LatentAERNet(autoencoder, aernet_model, compression_layer).to(device)

    # --- Optimizer and Loss ---
    optimizer = optim.Adam(
        list(model.aernet_model.parameters()) + list(model.compression.parameters()),
        lr=LEARNING_RATE
    )
    criterion = nn.CrossEntropyLoss()
    
    # --- Training Loop ---
    with open(LOG_CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_iou', 'val_f1'])
    
    best_val_f1 = -1.0
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            pre_images, post_images, masks = batch['pre_image'].to(device), batch['post_image'].to(device), batch['mask'].to(device)
            squeezed_masks = masks.squeeze(1)
            
            optimizer.zero_grad()
            outputs_tuple = model(pre_images, post_images)
            final_output = outputs_tuple[4] # 'result'

            # Convert final_output to (B, C, H, W) logits for CrossEntropyLoss
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
                
        avg_val_loss, avg_val_iou, avg_val_f1 = val_loss / len(val_loader), val_iou / len(val_loader), val_f1 / len(val_loader)
        print(f"Epoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f} | Val F1: {avg_val_f1:.4f}")
        
        with open(LOG_CSV_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, avg_val_loss, avg_val_iou, avg_val_f1])
        
        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1
            # Save the state_dict of the AERNet model and the compression layer
            torch.save({
                'aernet_state_dict': model.aernet_model.state_dict(),
                'compression_state_dict': model.compression.state_dict()
            }, MODEL_SAVE_PATH)
            print(f"Validation F1-score improved to {best_val_f1:.4f}. Model saved.")

    print("\n--- LEVIR-CD AERNet Latent Feature training finished! ---")

if __name__ == "__main__":
    main()