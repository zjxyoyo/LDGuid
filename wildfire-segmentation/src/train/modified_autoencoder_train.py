import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data_loader.dataset import WildfireDataset, CombinedDataset
from src.model.modified_autoencoder2 import ModifiedAutoencoder

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

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Define transforms
    transforms = DoubleCompose([
        DoubleToTensor(),
        DoubleElasticTransform(alpha=250, sigma=10),
        DoubleHorizontalFlip(),
        DoubleVerticalFlip(),
        DoubleAffine(degrees=(-15, 15), translate=(0.15, 0.15), scale=(0.8, 1))
    ])

    # Load dataset
    dataset = WildfireDataset(PRE_POST_FIRE_DIR, transforms=None)
    combined_dataset = CombinedDataset(dataset, transforms=None)
    dataloader = DataLoader(combined_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)

    # Initialize autoencoder
    autoencoder = ModifiedAutoencoder().to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Training loop
    num_epochs = 30
    for epoch in range(num_epochs):
        autoencoder.train()
        running_loss = 0.0
        reconstruction_loss = 0.0
        for batch in tqdm(dataloader):
            post_fire_imgs, pre_fire_imgs = batch["post_fire_image"], batch["pre_fire_image"]
            post_fire_imgs = torch.tensor(post_fire_imgs).float().to(device).permute(0,3,1,2)
            pre_fire_imgs = torch.tensor(pre_fire_imgs).float().to(device).permute(0,3,1,2)

            optimizer.zero_grad()

            latent = autoencoder.encoder(torch.cat((pre_fire_imgs, post_fire_imgs), dim=1))
            pre_fire_processed = autoencoder.pre_fire_processor(pre_fire_imgs)
            pre_fire_resized = nn.functional.interpolate(pre_fire_processed, size=latent.shape[2:], mode='bilinear', align_corners=False)
            reconstructed = autoencoder.decoder(torch.cat((latent, pre_fire_resized), dim=1))
            reconstruct_loss = criterion(reconstructed, post_fire_imgs)

            zero_mask = torch.zeros_like(pre_fire_imgs).to(device)
            zero_mask_processed = autoencoder.pre_fire_processor(zero_mask)
            zero_mask_resized = nn.functional.interpolate(zero_mask_processed, size=latent.shape[2:], mode='bilinear', align_corners=False)
            reconstructed_with_zero_mask = autoencoder.decoder(torch.cat((latent, zero_mask_resized), dim=1))
            zero_mask_loss = criterion(reconstructed_with_zero_mask, post_fire_imgs)

            loss = reconstruct_loss - 0.5 * zero_mask_loss
            # Forward pass
            loss.backward()
            optimizer.step()

            reconstruction_loss += reconstruct_loss.item() * post_fire_imgs.size(0)

            running_loss += loss.item() * post_fire_imgs.size(0)

        epoch_loss = running_loss / len(combined_dataset)
        r_loss = reconstruction_loss / len(combined_dataset)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        print(f"Epoch [{epoch+1}/{num_epochs}], Reconstruction Loss: {r_loss:.4f}")

    # Save the autoencoder
    save_path = os.path.join("model_weights", "modified_autoencoder.pth")
    torch.save(autoencoder.state_dict(), save_path)
    print("Modified autoencoder saved successfully.")

