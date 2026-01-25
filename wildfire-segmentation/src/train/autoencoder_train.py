import os
import sys

sys.path.append(os.getcwd())

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from src.data_loader.dataset import WildfireDataset, CombinedDataset
from src.model.autoencoder import Autoencoder

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
    combined_dataset = CombinedDataset(dataset, transforms=transforms)
    dataloader = DataLoader(combined_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)

    # Initialize autoencoder
    autoencoder = Autoencoder().to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        autoencoder.train()
        running_loss = 0.0
        for batch in tqdm(dataloader):
            post_fire_imgs, pre_fire_imgs = batch
            post_fire_imgs = torch.tensor(post_fire_imgs).float().to(device)
            pre_fire_imgs = torch.tensor(pre_fire_imgs).float().to(device)

            optimizer.zero_grad()

            # Train on post-fire images
            _, reconstructed_post_fire = autoencoder(post_fire_imgs)
            loss_post_fire = criterion(reconstructed_post_fire, post_fire_imgs)

            # Train on pre-fire images
            _, reconstructed_pre_fire = autoencoder(pre_fire_imgs)
            loss_pre_fire = criterion(reconstructed_pre_fire, pre_fire_imgs)

            loss = loss_post_fire + loss_pre_fire
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * post_fire_imgs.size(0)

        epoch_loss = running_loss / (2 * len(combined_dataset))
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Save the autoencoder
    save_path = os.path.join("model_weights", "autoencoder.pth")
    torch.save(autoencoder.state_dict(), save_path)
    print("Autoencoder saved successfully.")
