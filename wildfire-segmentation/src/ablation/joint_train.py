import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
from sklearn.metrics import f1_score, jaccard_score
import itertools # We need this to cycle through the dataloader for the AE inner loop

# Import your custom modules
from src.data_loader.svcd_dataset_class import SVCD_Dataset
from src.model.ae_model import ChangeDetectionAE 
from src.model.unet import UNet

# --- Wrapper Model (No changes needed structurally) ---
class LatentUNet(nn.Module):
    def __init__(self, autoencoder, unet_model, latent_compression_layer):
        super().__init__()
        self.autoencoder = autoencoder
        self.unet = unet_model
        self.compression = latent_compression_layer
        # AE is NOT frozen here

    def forward(self, pre_image, post_image):
        # Forward pass through AE to get latent and initial reconstruction
        latent_change, reconstructed_post_ae = self.autoencoder(pre_image, post_image)
        # Compress latent
        compressed_latent = self.compression(latent_change)
        # Upsample latent
        upsampled_latent = F.interpolate(compressed_latent, size=post_image.shape[2:], mode='bilinear', align_corners=False)
        # Combine for U-Net input
        combined_input = torch.cat((post_image, upsampled_latent), dim=1)
        # Get U-Net output
        output_unet = self.unet(combined_input)
        
        # Return components needed for both losses
        return latent_change, reconstructed_post_ae, output_unet

# --- Configuration ---
# ==============================================================================
DATASET_ROOT_PATH = os.path.expanduser('~/projects/def-bereyhia/zjxeric/data/SVCD_dataset')
AE_WEIGHTS_PATH = "./results/svcd_autoencoder/best_ae_svcd.pth" # Path to PRE-TRAINED AE weights (optional starting point)
SAVE_DIR = "./results/svcd_joint_training"
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "best_joint_model_svcd.pth")
LOG_CSV_PATH = os.path.join(SAVE_DIR, "joint_svcd_log.csv")
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_OUTER_EPOCHS = 100 
AE_STEPS_PER_UNET = 10 # Train AE 10 times for every 1 U-Net step
BATCH_SIZE = 8 
LEARNING_RATE_AE = 1e-5 
LEARNING_RATE_UNET = 1e-4 # Keep U-Net LR potentially higher
IMG_CHANNELS = 3
AE_LATENT_CHANNELS = 128
LATENT_CHANNELS_COMPRESSED = 4
UNET_INPUT_CHANNELS = IMG_CHANNELS + LATENT_CHANNELS_COMPRESSED # -> 7
N_CLASSES = 2
LOAD_PRETRAINED_AE = False # Flag to load initial weights for AE
# ==============================================================================

# --- Helper Functions (Unchanged) ---
def calculate_metrics(outputs, labels):
    # ... (same as before) ...
    preds = torch.argmax(outputs, dim=1)
    preds_np, labels_np = preds.cpu().numpy().flatten(), labels.cpu().numpy().flatten()
    f1 = f1_score(labels_np, preds_np, average='binary', zero_division=0)
    iou = jaccard_score(labels_np, preds_np, average='binary', zero_division=0)
    return f1, iou

# --- Main Training Function ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading (No Augmentation as per previous request) ---
    print("Loading SVCD dataset...")
    train_dataset = SVCD_Dataset(root_dir=DATASET_ROOT_PATH, split='train', transforms=None)
    val_dataset = SVCD_Dataset(root_dir=DATASET_ROOT_PATH, split='val', transforms=None)
    # Use persistent_workers=True if possible to speed up dataloader restarts
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True if torch.cuda.is_available() else False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True if torch.cuda.is_available() else False)
    print(f"Data loaded. Train: {len(train_dataset)}, Validation: {len(val_dataset)}")

    # --- Model Initialization ---
    print("Initializing models...")
    autoencoder = ChangeDetectionAE(in_channels=IMG_CHANNELS)
    if LOAD_PRETRAINED_AE:
        try:
            autoencoder.load_state_dict(torch.load(AE_WEIGHTS_PATH, map_location='cpu')) # Load to CPU first
            print(f"Loaded pre-trained AE weights from {AE_WEIGHTS_PATH}")
        except FileNotFoundError:
            print(f"WARNING: Pre-trained AE weights not found at {AE_WEIGHTS_PATH}. Starting AE from scratch.")
        except Exception as e:
            print(f"WARNING: Error loading AE weights: {e}. Starting AE from scratch.")

    unet = UNet(n_channels=UNET_INPUT_CHANNELS, n_classes=N_CLASSES, enable_dropout=True)
    compression_layer = nn.Conv2d(AE_LATENT_CHANNELS, LATENT_CHANNELS_COMPRESSED, kernel_size=1)
    
    # Assemble the final model - AE is NOT frozen
    model = LatentUNet(autoencoder, unet, compression_layer).to(device)

    # --- Optimizers ---
    optimizer_ae = optim.Adam(model.autoencoder.parameters(), lr=LEARNING_RATE_AE)
    optimizer_unet = optim.Adam(
        list(model.unet.parameters()) + list(model.compression.parameters()),
        lr=LEARNING_RATE_UNET,
        weight_decay=1e-5
    )

    # --- Loss Functions ---
    criterion_ae_recon = nn.MSELoss()
    criterion_unet_seg = nn.CrossEntropyLoss() # Unweighted as requested

    # --- Training Loop ---
    with open(LOG_CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss_ae', 'train_loss_unet', 'val_loss_unet', 'val_iou', 'val_f1'])

    best_val_f1 = 0.0
    # Use itertools.cycle to endlessly loop through the dataloader for the inner AE loop
    train_loader_iter = itertools.cycle(train_loader)

    print("Starting joint training...")
    for epoch in range(NUM_OUTER_EPOCHS):
        print(f"\n--- Outer Epoch {epoch+1}/{NUM_OUTER_EPOCHS} ---")
        
        model.train() # Set the entire model to train mode
        
        total_ae_loss_epoch = 0.0
        total_unet_loss_epoch = 0.0
        num_unet_steps = 0

        # We iterate roughly through the dataset once for the U-Net updates
        pbar = tqdm(range(len(train_loader)), desc=f"Epoch {epoch+1} Training")
        for i in pbar:
            
            # --- Inner Loop: Train AE for AE_STEPS_PER_UNET steps ---
            ae_loss_step_sum = 0.0
            for _ in range(AE_STEPS_PER_UNET):
                try:
                    ae_batch = next(train_loader_iter)
                except StopIteration: # Should not happen with itertools.cycle but added for safety
                    train_loader_iter = itertools.cycle(train_loader)
                    ae_batch = next(train_loader_iter)
                
                pre_images_ae = ae_batch['pre_image'].to(device)
                post_images_ae = ae_batch['post_image'].to(device)

                optimizer_ae.zero_grad()
                
                # AE forward pass (only need AE components)
                latent_change_ae, reconstructed_post_ae = model.autoencoder(pre_images_ae, post_images_ae)
                reconstruction_loss_ae = criterion_ae_recon(reconstructed_post_ae, post_images_ae)

                # Adversarial loss calculation for AE
                zero_context_input_ae = torch.zeros_like(pre_images_ae)
                zero_context_features_ae = model.autoencoder.context_encoder(zero_context_input_ae)
                adv_decoder_input_ae = torch.cat((latent_change_ae.detach(), zero_context_features_ae), dim=1)
                adv_reconstruction_ae = model.autoencoder.decoder(adv_decoder_input_ae)
                adversarial_loss_ae = criterion_ae_recon(adv_reconstruction_ae, post_images_ae)
                
                final_loss_ae = reconstruction_loss_ae - 0.5 * adversarial_loss_ae
                final_loss_ae.backward()
                optimizer_ae.step()
                ae_loss_step_sum += final_loss_ae.item()

            avg_ae_loss_inner = ae_loss_step_sum / AE_STEPS_PER_UNET
            total_ae_loss_epoch += ae_loss_step_sum

            # --- Outer Step: Train U-Net + Compression Layer once ---
            # We can use the last batch fetched by the AE loop or fetch a new one
            # Using the last batch (ae_batch) is slightly more efficient
            pre_images_unet = ae_batch['pre_image'].to(device)
            post_images_unet = ae_batch['post_image'].to(device)
            masks_unet = ae_batch['mask'].to(device)
            squeezed_masks_unet = masks_unet.squeeze(1)

            optimizer_unet.zero_grad()

            # Full forward pass through the wrapper model
            # Gradients WILL flow back through AE from this step
            _, _, output_unet = model(pre_images_unet, post_images_unet)
            
            loss_unet = criterion_unet_seg(output_unet, squeezed_masks_unet)
            loss_unet.backward()
            optimizer_unet.step()
            total_unet_loss_epoch += loss_unet.item()
            num_unet_steps += 1
            
            pbar.set_postfix(ae_loss=f"{avg_ae_loss_inner:.4f}", unet_loss=f"{loss_unet.item():.4f}")

        avg_train_loss_ae = total_ae_loss_epoch / (num_unet_steps * AE_STEPS_PER_UNET)
        avg_train_loss_unet = total_unet_loss_epoch / num_unet_steps

        # --- Validation Phase (Validate U-Net performance) ---
        model.eval() # Set the entire model to eval mode
        val_loss_unet, val_iou, val_f1 = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                pre_images = batch['pre_image'].to(device)
                post_images = batch['post_image'].to(device)
                masks = batch['mask'].to(device)
                squeezed_masks = masks.squeeze(1)
                
                # Full forward pass needed for validation
                _, _, output_unet_val = model(pre_images, post_images)
                
                loss_val = criterion_unet_seg(output_unet_val, squeezed_masks)
                val_loss_unet += loss_val.item()
                f1, iou = calculate_metrics(output_unet_val, squeezed_masks)
                val_f1 += f1
                val_iou += iou
        
        avg_val_loss_unet = val_loss_unet / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        avg_val_f1 = val_f1 / len(val_loader)
        
        print(f"Epoch {epoch+1} Summary: Avg Train AE Loss: {avg_train_loss_ae:.6f} | Avg Train U-Net Loss: {avg_train_loss_unet:.4f} | Val U-Net Loss: {avg_val_loss_unet:.4f} | Val IoU: {avg_val_iou:.4f} | Val F1: {avg_val_f1:.4f}")
        
        with open(LOG_CSV_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss_ae, avg_train_loss_unet, avg_val_loss_unet, avg_val_iou, avg_val_f1])
        
        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1
            # Save the entire model's state dict
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Validation F1-score improved to {best_val_f1:.4f}. Model saved to {MODEL_SAVE_PATH}")

    print("\n--- Joint training finished! ---")

if __name__ == "__main__":
    main()