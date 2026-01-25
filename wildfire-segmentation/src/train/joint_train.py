import os
import sys

sys.path.append(os.getcwd())

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from src.data_loader.dataset import WildfireDataset, CombinedDataset
from src.model.modified_autoencoder import ModifiedAutoencoder
from src.model.simple_network import SimpleSegmentationNetwork
from src.data_loader.augmentation import (
    CustomColorJitter,
    DoubleAffine,
    DoubleCompose,
    DoubleElasticTransform,
    DoubleHorizontalFlip,
    DoubleToTensor,
    DoubleVerticalFlip,
    GaussianNoise,
)
from data.load_data import POST_FIRE_DIR, PRE_POST_FIRE_DIR

def resize_mask(mask, size):
    if mask.dim() == 4 and mask.size(2) == 1:  # Check if mask is [B, H, 1, W]
        mask = mask.permute(0, 2, 1, 3)  # Change to [B, 1, H, W]
    resized_mask = torch.nn.functional.interpolate(mask.float(), size=size, mode='bilinear', align_corners=False)
    return resized_mask



def train_joint_model(autoencoder, segmentation_model, train_loader, val_loader, device, epochs=5, lr=1e-3):
    optimizer = optim.Adam(list(autoencoder.parameters()) + list(segmentation_model.parameters()), lr=lr)
    reconstruction_criterion = nn.MSELoss()
    segmentation_criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # writer = SummaryWriter()

    best_val_loss = float('inf')

    for epoch in range(epochs):
        autoencoder.train()
        segmentation_model.train()
        running_loss = 0.0
        num_samples = 0
        train_loop = tqdm(train_loader, leave=True)
        for batch in train_loop:
            post_fire_imgs, pre_fire_imgs, masks = batch
            post_fire_imgs = torch.tensor(post_fire_imgs).float().to(device)
            pre_fire_imgs = torch.tensor(pre_fire_imgs).float().to(device)
            masks = torch.tensor(masks).float().to(device)

            optimizer.zero_grad()
            latent, reconstructed = autoencoder(pre_fire_imgs, post_fire_imgs)
            reconstruction_loss = reconstruction_criterion(reconstructed, post_fire_imgs)
            
            # Train segmentation model
            segmentation_output = segmentation_model(latent)
            # print(masks.size())
            
            resized_masks = resize_mask(masks, segmentation_output.shape[2:])  # Resize masks to match the output
            # print(resized_masks.size())
            segmentation_loss = segmentation_criterion(segmentation_output, resized_masks)
            
            loss = reconstruction_loss + segmentation_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * post_fire_imgs.size(0)
            num_samples += post_fire_imgs.size(0)

            train_loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            train_loop.set_postfix(loss=running_loss / num_samples)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {epoch_loss:.4f}")
        # writer.add_scalar('Loss/Train', epoch_loss, epoch)

        # Validation step
        autoencoder.eval()
        segmentation_model.eval()
        val_loss = 0.0
        num_val_samples = 0
        val_loop = tqdm(val_loader, leave=True, desc="Validation")
        with torch.no_grad():
            for batch in val_loop:
                post_fire_imgs, pre_fire_imgs, masks = batch
                post_fire_imgs = torch.tensor(post_fire_imgs).float().to(device)
                pre_fire_imgs = torch.tensor(pre_fire_imgs).float().to(device)
                masks = torch.tensor(masks).float().to(device)

                latent, reconstructed = autoencoder(pre_fire_imgs, post_fire_imgs)
                reconstruction_loss = reconstruction_criterion(reconstructed, post_fire_imgs)
                
                segmentation_output = segmentation_model(latent)
                resized_masks = resize_mask(masks, segmentation_output.shape[2:])  # Resize masks to match the output
                segmentation_loss = segmentation_criterion(segmentation_output, resized_masks)
                
                loss = reconstruction_loss + segmentation_loss
                val_loss += loss.item() * post_fire_imgs.size(0)

                num_val_samples += post_fire_imgs.size(0)
                val_loop.set_postfix(loss=val_loss / num_val_samples)



        val_loss /= len(val_loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss:.4f}")
        # writer.add_scalar('Loss/Validation', val_loss, epoch)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join("model_weights", "best_joint_model.pth")
            torch.save({
                'autoencoder': autoencoder.state_dict(),
                'segmentation_model': segmentation_model.state_dict()
            }, save_path)

    # writer.close()
    return autoencoder, segmentation_model

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Define transforms
    transforms = DoubleCompose([
        DoubleToTensor(),
        # DoubleElasticTransform(alpha=250, sigma=10),
        DoubleHorizontalFlip(),
        DoubleVerticalFlip(),
        DoubleAffine(degrees=(-15, 15), translate=(0.15, 0.15), scale=(0.8, 1))
    ])

    # Load dataset
    dataset = WildfireDataset(PRE_POST_FIRE_DIR, transforms=None, include_pre_fire=True)
    combined_dataset = CombinedDataset(dataset, transforms=transforms)
    train_size = int(0.8 * len(combined_dataset))
    val_size = len(combined_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(combined_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize models
    autoencoder = ModifiedAutoencoder().to(device)
    segmentation_model = SimpleSegmentationNetwork(input_channels=256, output_channels=1).to(device)

    # Train models
    trained_autoencoder, trained_segmentation_model = train_joint_model(autoencoder, segmentation_model, train_loader, val_loader, device, epochs=50, lr=1e-3)

    # Save the final models
    save_path = os.path.join("model_weights", "final_joint_model.pth")
    torch.save({
        'autoencoder': trained_autoencoder.state_dict(),
        'segmentation_model': trained_segmentation_model.state_dict()
    }, save_path)
    print("Final joint model saved successfully.")