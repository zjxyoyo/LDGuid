# train_unet_with_latent.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
from sklearn.metrics import f1_score, jaccard_score


from src.helper.losses import DiceLoss
from src.data_loader.OSCDaugmentation import JointCompose, JointRandomHorizontalFlip, JointRandomVerticalFlip, JointRandomRotation

from src.data_loader.oscd_dataset_class import Manual_OSCD_Dataset
from src.model.unet import UNet
from src.model.OSCDAutoEncoder import OSCDAE

# Define a wrapped class for training.
# ==============================================================================
class LatentUNet(nn.Module):
    def __init__(self, pretrained_ae, unet_model, latent_compression_layer):
        super().__init__()
        self.autoencoder = pretrained_ae
        self.unet = unet_model
        self.compression = latent_compression_layer
        
        self.autoencoder.eval()
        for param in self.autoencoder.parameters():
            param.requires_grad = False

    def forward(self, pre_image, post_image):
        # Use frozen AE to extract Latent Feature
        with torch.no_grad():
            latent_change, _ = self.autoencoder(pre_image, post_image)

        # Compress channels
        compressed_latent = self.compression(latent_change)
        
        # Make sure the size is the same
        upsampled_latent = F.interpolate(compressed_latent, size=post_image.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate
        combined_input = torch.cat((post_image, upsampled_latent), dim=1)
        
        # Put it to U-Net
        output = self.unet(combined_input)
        
        return output
# ==============================================================================


# Configuration
DATASET_ROOT_PATH = os.path.expanduser('~/projects/def-bereyhia/zjxeric/data/Onera Satellite Change Detection')
AE_WEIGHTS_PATH = "./model_weights/oscd_autoencoder.pth" 
SAVE_DIR = "./results/unet_with_latent"
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "best_unet_with_latent.pth")
LOG_CSV_PATH = os.path.join(SAVE_DIR, "latent_training_log.csv")
os.makedirs(SAVE_DIR, exist_ok=True)

# Hyperparameter
NUM_EPOCHS = 300 # Same as benchmark training
BATCH_SIZE = 8
LEARNING_RATE = 1e-4

# Model parameter
IMG_CHANNELS = 12
LATENT_CHANNELS_COMPRESSED = 4 # same as before
UNET_INPUT_CHANNELS = IMG_CHANNELS + LATENT_CHANNELS_COMPRESSED # -> 16
N_CLASSES = 2
# ==============================================================================

# Helper Function
def calculate_metrics(outputs, labels):
    preds = torch.argmax(outputs, dim=1)
    preds_np, labels_np = preds.cpu().numpy().flatten(), labels.cpu().numpy().flatten()
    f1 = f1_score(labels_np, preds_np, average='binary', zero_division=0)
    iou = jaccard_score(labels_np, preds_np, average='binary', zero_division=0)
    return f1, iou

def calculate_class_weights(dataloader, n_classes):
    print("Calculating class weights...")
    class_counts = torch.zeros(n_classes, dtype=torch.float64)
    for batch in tqdm(dataloader, desc="Calculating Weights"):
        masks = batch['mask'].clamp(0, n_classes - 1)
        class_counts += torch.bincount(masks.flatten(), minlength=n_classes).double()
    total_pixels = class_counts.sum()
    class_weights = total_pixels / (class_counts + 1e-6)
    print(f"Pixel counts: {class_counts.tolist()}")
    print(f"Calculated class weights: {class_weights.tolist()}")
    return class_weights

# Training
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # Data loading
    print("Loading datasets with augmentation...")
    train_transforms = JointCompose([
        JointRandomHorizontalFlip(p=0.5),
        JointRandomVerticalFlip(p=0.5),
        JointRandomRotation(degrees=[0, 90, 180, 270]),
    ])
    # No augmentation for the validation set
    val_transforms = None


    # Create datasets, passing the transforms
    train_dataset = Manual_OSCD_Dataset(root_dir=DATASET_ROOT_PATH, split='train', transforms=train_transforms)
    val_dataset = Manual_OSCD_Dataset(root_dir=DATASET_ROOT_PATH, split='test', transforms=val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Loss function
    class_weights = calculate_class_weights(train_loader, N_CLASSES).to(device, dtype=torch.float)
    ce_loss_criterion = nn.CrossEntropyLoss(weight=class_weights)
    dice_loss_criterion = DiceLoss(n_classes=N_CLASSES)

    # Model initialization
    print("Initializing models...")
    # Load AE
    pretrained_ae = OSCDAE(in_channels=IMG_CHANNELS)
    pretrained_ae.load_state_dict(torch.load(AE_WEIGHTS_PATH, map_location=device))
    
    # Initialize U-Net
    unet = UNet(n_channels=UNET_INPUT_CHANNELS, n_classes=N_CLASSES, enable_dropout=True)
    
    # Create compress Layer
    # 128->4
    latent_compression = nn.Conv2d(128, LATENT_CHANNELS_COMPRESSED, kernel_size=1).to(device)

    # Make a wrapped class
    model = LatentUNet(pretrained_ae, unet, latent_compression).to(device)
    
    # Optimizer
    optimizer = optim.Adam(
        list(model.unet.parameters()) + list(model.compression.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-5
    )
    
    # Train
    with open(LOG_CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_iou', 'val_f1'])
    
    best_val_f1 = 0.0
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        model.train()
        # 
        model.unet.train()
        model.compression.train()

        train_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            pre_images = batch['pre_image'].to(device)
            post_images = batch['post_image'].to(device)
            masks = batch['mask'].to(device)
            squeezed_masks = masks.squeeze(1)

            optimizer.zero_grad()
            
            # Call the wrapped model
            outputs = model(pre_images, post_images)
            
            ce_loss = ce_loss_criterion(outputs, squeezed_masks)
            dice_loss = dice_loss_criterion(outputs, squeezed_masks, softmax=True)
            loss = ce_loss + dice_loss
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss, val_iou, val_f1 = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                pre_images = batch['pre_image'].to(device)
                post_images = batch['post_image'].to(device)
                masks = batch['mask'].to(device)
                squeezed_masks = masks.squeeze(1)
                
                outputs = model(pre_images, post_images)
                
                loss = ce_loss_criterion(outputs, squeezed_masks)
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
            # Save the weights
            torch.save({
                'unet_state_dict': model.unet.state_dict(),
                'compression_state_dict': model.compression.state_dict()
            }, MODEL_SAVE_PATH)
            print(f"Validation F1-score improved to {best_val_f1:.4f}. Model saved.")

    print("\n--- Latent feature training finished! ---")

if __name__ == "__main__":
    main()