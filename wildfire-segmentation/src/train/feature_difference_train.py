import os
import sys

sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.load_data import POST_FIRE_DIR, PRE_POST_FIRE_DIR
from src.model.unet import UNet
from src.model.autoencoder import Autoencoder
from src.data_loader.dataset import WildfireDataset, FeatureDiffDataset

# Parameter
BATCH_SIZE = 8

def resize_mask(mask, size):
    """
    Resize mask to the specified size using bilinear interpolation.
    Args:
        mask (torch.Tensor): The mask tensor of shape (B, 1, H, W).
        size (tuple): The target size (H, W).
    Returns:
        torch.Tensor: The resized mask tensor of shape (B, 1, H, W).
    """
    # Change shape from (B, H, W, 1) to (B, 1, H, W)
    mask = mask.permute(0, 3, 1, 2)

    resized_mask = F.interpolate(mask.float(), size=size, mode='bilinear', align_corners=False)
   #resized_mask = F.interpolate(mask, size=size, mode='bilinear', align_corners=False)
    return resized_mask

def train_segmentation_network(autoencoder, dataloader, device, epochs=50, lr=0.001):
    autoencoder.eval()
    feature_diff_loader = dataloader.get_feature_diff_dataloader(autoencoder, BATCH_SIZE, device)
    print("Feature difference loaded")
    # Initialize segmentation model
    model = UNet(n_channels=autoencoder.encoder_out_channels, n_classes=1, bilinear=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Split dataloader into train and validation
    n_val = int(len(feature_diff_loader.dataset) * 0.2)
    n_train = len(feature_diff_loader.dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(feature_diff_loader.dataset, [n_train, n_val])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        num_train_samples = 0
        train_loop = tqdm(train_loader, leave=True)
        for batch in train_loop:
            feature_diff = batch['feature_diff'].to(device)
            masks = batch['mask'].to(device)
            print(f"Mask Size: {masks.size()}")
            resized_masks = resize_mask(masks, feature_diff.size()[-2:])
            print(f"Feature Diff Size: {feature_diff.size()}, Resized Mask Size: {resized_masks.size()}")
            optimizer.zero_grad()
            outputs = model(feature_diff)
            loss = criterion(outputs, resized_masks.float())  
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * feature_diff.size(0)
            num_train_samples += feature_diff.size(0)
            train_loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            train_loop.set_postfix(loss=train_loss / num_train_samples)

        avg_train_loss = train_loss / num_train_samples
        print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {avg_train_loss:.4f}')

        # Validation loop
        model.eval()
        val_loss = 0
        num_val_samples = 0
        val_loop = tqdm(val_loader, leave=True, desc="Validation")
        with torch.no_grad():
            for batch in val_loop:
                feature_diff = batch['feature_diff'].to(device)
                masks = batch['mask'].to(device)
                resized_masks = resize_mask(masks, feature_diff.size()[-2:])
                outputs = model(feature_diff)
                loss = criterion(outputs, resized_masks.float())  
                val_loss += loss.item() * feature_diff.size(0)
                num_val_samples += feature_diff.size(0)
                val_loop.set_postfix(loss=val_loss / num_val_samples)

        avg_val_loss = val_loss / num_val_samples
        print(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_val_loss:.4f}')

    return model

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:0")
    print(device)

    # Prepare dataset and dataloader
    dataset = WildfireDataset(PRE_POST_FIRE_DIR, transforms=None, include_pre_fire=True)

    # Initialize and train autoencoder
    autoencoder = Autoencoder()
    autoencoder.to(device)
    # Load the pretrained autoencoder named trained_autoencoder.
    trained_autoencoder = Autoencoder()
    trained_autoencoder.load_state_dict(torch.load("model_weights/autoencoder.pth"))
    trained_autoencoder.to(device)

    # Train segmentation network using feature differences
    trained_segmentation_model = train_segmentation_network(trained_autoencoder, dataset, device, epochs=50, lr=0.001)

    # Save the trained models
    torch.save(trained_autoencoder.state_dict(), "trained_autoencoder.pth")
    torch.save(trained_segmentation_model.state_dict(), "trained_segmentation_model.pth")

