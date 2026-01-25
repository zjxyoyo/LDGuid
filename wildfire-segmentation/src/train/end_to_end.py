import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, jaccard_score

from src.data_loader.dataset import WildfireDataset, CombinedDataset
from src.model.modified_autoencoder2 import ModifiedAutoencoder
from src.model.unet import UNet
from src.data_loader.augmentation2 import DoubleCompose, DoubleToTensor, DoubleHorizontalFlip, DoubleVerticalFlip, DoubleAffine, DoubleElasticTransform
from data.load_data import POST_FIRE_DIR, PRE_POST_FIRE_DIR

# Helper functions to calculate metrics
def calculate_metrics(output, target, threshold=0.5):
    preds_binary = (torch.sigmoid(output) > threshold).float()
    true_masks_binary = target.float()
    preds_binary_np = preds_binary.view(-1).detach().cpu().numpy()
    true_masks_binary_np = true_masks_binary.view(-1).detach().cpu().numpy()
    # Ensure no NaNs or Infs
    np.nan_to_num(preds_binary_np, copy=False)
    np.nan_to_num(true_masks_binary_np, copy=False)

    f1 = f1_score(true_masks_binary_np, preds_binary_np, average="macro")
    iou = jaccard_score(true_masks_binary_np, preds_binary_np, average="macro")
    return f1, iou

# Combined training and evaluation function
def train_and_evaluate_combined(autoencoder, segmentation_model, dataloaders, device, epochs=50, lr=1e-3, alpha=0.5):
    optimizer = optim.Adam(list(autoencoder.parameters()) + list(segmentation_model.parameters()), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()

    best_val_loss = float('inf')
    best_f1 = 0.0
    best_iou = 0.0
    conv1x1 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1).to(device)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Training Phase
        autoencoder.train()
        segmentation_model.train()
        train_loss = 0.0
        train_f1 = 0.0
        train_iou = 0.0
        num_train_batches = len(dataloaders['train'])

        for batch in tqdm(dataloaders['train']):
            post_fire_imgs, pre_fire_imgs, masks = batch["post_fire_image"], batch["pre_fire_image"], batch["mask"]
            post_fire_imgs = post_fire_imgs.float().to(device)
            pre_fire_imgs = pre_fire_imgs.float().to(device)
            masks = masks.float().to(device)



            optimizer.zero_grad()

            # Autoencoder Forward Pass
            latent, reconstructed = autoencoder(pre_fire_imgs, post_fire_imgs)
            reconstruct_loss = mse_loss(reconstructed, post_fire_imgs)

            # Zero-mask Forward Pass
            zero_mask = torch.zeros_like(pre_fire_imgs).to(device)
            zero_mask_processed = autoencoder.pre_fire_processor(zero_mask)
            zero_mask_resized = nn.functional.interpolate(zero_mask_processed, size=latent.shape[2:], mode='bilinear', align_corners=False)
            reconstructed_with_zero_mask = autoencoder.decoder(torch.cat((latent, zero_mask_resized), dim=1))
            zero_mask_loss = mse_loss(reconstructed_with_zero_mask, post_fire_imgs)
            
          
            with torch.no_grad():
              latent_resized = nn.functional.interpolate(latent, size=(512, 512), mode='bilinear', align_corners=False)
              latent_new = conv1x1(latent_resized)
              combined_input = torch.cat((post_fire_imgs, latent_new), dim=1)
            segmentation_output = segmentation_model(combined_input)
            segmentation_loss = criterion(segmentation_output, masks)


            # Combined Loss
            loss = reconstruct_loss - 0.5 * zero_mask_loss + alpha * segmentation_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            f1, iou = calculate_metrics(segmentation_output, masks)
            train_f1 += f1
            train_iou += iou

        train_loss /= num_train_batches
        train_f1 /= num_train_batches
        train_iou /= num_train_batches

        print(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, Train IoU: {train_iou:.4f}")

        # Validation Phase
        segmentation_model.eval()
        autoencoder.eval()
        val_loss = 0.0
        val_f1 = 0.0
        val_iou = 0.0
        num_val_batches = len(dataloaders['val'])
        thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        f1_scores = {t: [] for t in thresholds}
        iou_scores = {t: [] for t in thresholds}

        with torch.no_grad():
            for batch in tqdm(dataloaders['val']):
                post_fire_imgs, pre_fire_imgs, masks = batch["post_fire_image"], batch["pre_fire_image"], batch["mask"]
                post_fire_imgs = post_fire_imgs.float().to(device)
                pre_fire_imgs = pre_fire_imgs.float().to(device)
                masks = masks.float().to(device)

                # Autoencoder Forward Pass
                latent, reconstructed = autoencoder(pre_fire_imgs, post_fire_imgs)
                reconstruct_loss = mse_loss(reconstructed, post_fire_imgs)

                # Zero-mask Forward Pass
                zero_mask = torch.zeros_like(pre_fire_imgs).to(device)
                zero_mask_processed = autoencoder.pre_fire_processor(zero_mask)
                zero_mask_resized = nn.functional.interpolate(zero_mask_processed, size=latent.shape[2:], mode='bilinear', align_corners=False)
                reconstructed_with_zero_mask = autoencoder.decoder(torch.cat((latent, zero_mask_resized), dim=1))
                zero_mask_loss = mse_loss(reconstructed_with_zero_mask, post_fire_imgs)
                
                latent_resized = nn.functional.interpolate(latent, size=(512, 512), mode='bilinear', align_corners=False)
                latent_new = conv1x1(latent_resized)
                combined_input = torch.cat((post_fire_imgs, latent_new), dim=1)

                outputs = segmentation_model(combined_input)
                segmentation_loss = criterion(outputs, masks)

                loss = reconstruct_loss - 0.5 * zero_mask_loss + alpha * segmentation_loss

                val_loss += loss.item()
                for t in thresholds:
                    f1, iou = calculate_metrics(outputs, masks, threshold=t)
                    f1_scores[t].append(f1)
                    iou_scores[t].append(iou)

        val_loss /= num_val_batches
        avg_f1_scores = {t: sum(scores) / len(scores) for t, scores in f1_scores.items()}
        avg_iou_scores = {t: sum(scores) / len(scores) for t, scores in iou_scores.items()}

        print(f"Val Loss: {val_loss:.4f}")
        for t in thresholds:
            print(f"Threshold: {t} - F1 Score: {avg_f1_scores[t]:.4f}, IOU: {avg_iou_scores[t]:.4f}")

        best_f1 = max(best_f1, max(avg_f1_scores.values()))
        best_iou = max(best_iou, max(avg_iou_scores.values()))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(segmentation_model.state_dict(), "best_segmentation_model.pth")

        # Save training results to a CSV file
        results = {
            'Epoch': epoch + 1,
            'Train Loss': train_loss,
            'Train F1': train_f1,
            'Train IoU': train_iou,
            'Val Loss': val_loss,
            'Val F1': best_f1,
            'Val IoU': best_iou,
        }

        results_df = pd.DataFrame([results])
        results_df.to_csv('training_results.csv', mode='a', header=not os.path.exists('training_results.csv'), index=False)

    return segmentation_model

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Define transforms
    train_transforms = DoubleCompose([
        DoubleToTensor(),
        DoubleElasticTransform(alpha=250, sigma=10, p=0.5),
        DoubleHorizontalFlip(p=0.5),
        DoubleVerticalFlip(p=0.5),
        DoubleAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1))
    ])

    val_transforms = DoubleCompose([
        DoubleToTensor()
    ])

    # Load dataset
    dataset = WildfireDataset(PRE_POST_FIRE_DIR, transforms=None)
    combined_dataset = CombinedDataset(dataset, transforms=None)

    # Split dataset into training, validation, and test sets
    train_size = int(0.8 * len(combined_dataset))
    val_size = int(0.2 * len(combined_dataset))
    test_size = len(combined_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(combined_dataset, [train_size, val_size, test_size])

    # Apply transforms to the split datasets
    train_dataset.dataset.transforms = train_transforms
    val_dataset.dataset.transforms = val_transforms
    test_dataset.dataset.transforms = val_transforms

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=False)

    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

    # Initialize autoencoder and segmentation model
    autoencoder = ModifiedAutoencoder().to(device)
    segmentation_model = UNet(n_channels=16, n_classes=1, bilinear=False).to(device)

    # Train and evaluate the model
    trained_segmentation_model = train_and_evaluate_combined(autoencoder, segmentation_model, dataloaders, device, epochs=50, lr=1e-3, alpha=0.5)

    # Save the trained segmentation model
    torch.save(trained_segmentation_model.state_dict(), "trained_segmentation_model.pth")
    print("Trained segmentation model saved successfully.")
