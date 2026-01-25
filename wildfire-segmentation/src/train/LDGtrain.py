import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import pandas as pd
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
    
    np.nan_to_num(preds_binary_np, copy=False)
    np.nan_to_num(true_masks_binary_np, copy=False)

    f1 = f1_score(true_masks_binary_np, preds_binary_np, average="macro")
    iou = jaccard_score(true_masks_binary_np, preds_binary_np, average="macro")
    return f1, iou

def validate_model(model, conv1x1, autoencoder, dataloader, device):
    """
    Update: 现在需要传入训练好的 conv1x1
    """
    model.eval()
    conv1x1.eval() # 别忘了把这个也设为 eval 模式
    
    val_loss = 0.0
    thresholds = [0.5]
    f1_scores = {t: [] for t in thresholds}
    iou_scores = {t: [] for t in thresholds}
    criterion = nn.BCEWithLogitsLoss()
    num_val_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            post_fire_imgs, pre_fire_imgs, masks = batch["post_fire_image"], batch["pre_fire_image"], batch["mask"]
            post_fire_imgs = post_fire_imgs.float().to(device)
            pre_fire_imgs = pre_fire_imgs.float().to(device)
            masks = masks.float().to(device)
            
            # Latent Extraction
            latent, _ = autoencoder(pre_fire_imgs, post_fire_imgs)
            latent_resized = nn.functional.interpolate(latent, size=(512, 512), mode='bilinear', align_corners=False)
            
            # Feature Compression (Training this part now!)
            latent_new = conv1x1(latent_resized)
            
            combined_input = torch.cat((post_fire_imgs, latent_new), dim=1)

            outputs = model(combined_input)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * post_fire_imgs.size(0)
            num_val_samples += post_fire_imgs.size(0)
            
            for t in thresholds:
                f1, iou = calculate_metrics(outputs, masks, threshold=t)
                f1_scores[t].append(f1)
                iou_scores[t].append(iou)
    
    avg_val_loss = val_loss / num_val_samples
    avg_f1_scores = {t: sum(scores) / len(scores) for t, scores in f1_scores.items()}
    avg_iou_scores = {t: sum(scores) / len(scores) for t, scores in iou_scores.items()}
    
    return avg_val_loss, avg_f1_scores, avg_iou_scores


def train_and_evaluate(autoencoder, segmentation_model, conv1x1, dataloaders, device, epochs=50, lr=1e-3):
    """
    Update: 传入 conv1x1，并将其加入优化器
    """
    # Key Fix: Optimize both UNet and Conv1x1 parameters
    optimizer = optim.Adam(
        list(segmentation_model.parameters()) + list(conv1x1.parameters()), 
        lr=lr
    )
    
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    best_f1 = 0.0
    best_iou = 0.0

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Training Phase
        segmentation_model.train()
        conv1x1.train() # Set conv1x1 to training mode
        
        train_loss = 0.0
        train_f1 = 0.0
        train_iou = 0.0
        num_train_batches = len(dataloaders['train'])

        for batch in tqdm(dataloaders['train'], desc="Training"):
            post_fire_imgs, pre_fire_imgs, masks = batch["post_fire_image"], batch["pre_fire_image"], batch["mask"]
            post_fire_imgs = post_fire_imgs.float().to(device)
            pre_fire_imgs = pre_fire_imgs.float().to(device)
            masks = masks.float().to(device)

            optimizer.zero_grad()

            # Autoencoder is frozen, no gradients needed here
            with torch.no_grad():
                latent, _ = autoencoder(pre_fire_imgs, post_fire_imgs)
                latent_resized = nn.functional.interpolate(latent, size=(512, 512), mode='bilinear', align_corners=False)
            
            # Conv1x1 IS trainable, so we need gradients here
            latent_new = conv1x1(latent_resized)
            
            combined_input = torch.cat((post_fire_imgs, latent_new), dim=1)

            outputs = segmentation_model(combined_input)
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            f1, iou = calculate_metrics(outputs, masks)
            train_f1 += f1
            train_iou += iou

        train_loss /= num_train_batches
        train_f1 /= num_train_batches
        train_iou /= num_train_batches

        print(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, Train IoU: {train_iou:.4f}")

        # Validation Phase (Pass conv1x1 to validator)
        val_loss, avg_f1_scores, avg_iou_scores = validate_model(segmentation_model, conv1x1, autoencoder, dataloaders['val'], device)
        
        # Select metrics for threshold 0.5 for logging (or pick best)
        current_f1 = avg_f1_scores[0.5]
        current_iou = avg_iou_scores[0.5]
        
        print(f"Val Loss: {val_loss:.4f} | F1 (0.5): {current_f1:.4f} | IoU (0.5): {current_iou:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Key Fix: Save BOTH models
            torch.save({
                'segmentation_model': segmentation_model.state_dict(),
                'conv1x1': conv1x1.state_dict()
            }, "best_ldg_model_bundle.pth")
            print("Best model bundle saved.")

        # Save training results
        results = {
            'Epoch': epoch + 1,
            'Train Loss': train_loss,
            'Train F1': train_f1,
            'Train IoU': train_iou,
            'Val Loss': val_loss,
            'Val F1 (0.5)': current_f1,
            'Val IoU (0.5)': current_iou,
        }
        results_df = pd.DataFrame([results])
        results_df.to_csv('training_results.csv', mode='a', header=not os.path.exists('training_results.csv'), index=False)

    return segmentation_model, conv1x1

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    # Split dataset
    train_size = int(0.8 * len(combined_dataset))
    val_size = int(0.2 * len(combined_dataset))
    test_size = len(combined_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(combined_dataset, [train_size, val_size, test_size])

    # Apply transforms
    train_dataset.dataset.transforms = train_transforms
    val_dataset.dataset.transforms = val_transforms
    test_dataset.dataset.transforms = val_transforms

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

    # Initialize models
    autoencoder = ModifiedAutoencoder().to(device)
    # Load AE weights (Make sure path is correct)
    if os.path.exists("model_weights/modified_autoencoder.pth"):
        autoencoder.load_state_dict(torch.load("model_weights/modified_autoencoder.pth"))
        print("Autoencoder weights loaded.")
    else:
        print("Warning: Autoencoder weights not found! Using random init.")
    
    autoencoder.eval() 

    # Key Fix: Initialize conv1x1 HERE and pass it
    conv1x1 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1).to(device)
    
    # UNet input is 12 (post) + 4 (latent) = 16
    segmentation_model = UNet(n_channels=16, n_classes=1, bilinear=False).to(device)

    # Train
    train_and_evaluate(autoencoder, segmentation_model, conv1x1, dataloaders, device, epochs=50, lr=1e-3)