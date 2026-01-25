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
# from src.model.simple_segmentation_net import SimpleSegmentationNet  # simple segmentation net
from src.data_loader.augmentation2 import DoubleCompose, DoubleToTensor, DoubleHorizontalFlip, DoubleVerticalFlip, DoubleAffine, DoubleElasticTransform
# from src.data_loader.augmentation import DoubleCompose, DoubleElasticTransform, DoubleHorizontalFlip, DoubleVerticalFlip, DoubleAffine, DoubleToTensor
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

    # output = (output > threshold).float()
    # target = target.float()
    # output_np = output.cpu().numpy().flatten()
    # target_np = target.cpu().numpy().flatten()

    # f1 = f1_score(target_np, output_np, average='binary')
    # iou = jaccard_score(target_np, output_np, average='binary')
    return f1, iou

def validate_model(model, dataloader, device):
    model.eval()
    val_loss = 0.0
    thresholds = [0.5]
    f1_scores = {t: [] for t in thresholds}
    iou_scores = {t: [] for t in thresholds}
    criterion = nn.BCEWithLogitsLoss()
    num_val_samples = 0
    conv1x1 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1).to(device)
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            post_fire_imgs, pre_fire_imgs, masks= batch["post_fire_image"], batch["pre_fire_image"], batch["mask"]
            post_fire_imgs = post_fire_imgs.float().to(device).permute(0,3,1,2)
            pre_fire_imgs = pre_fire_imgs.float().to(device).permute(0,3,1,2)
            masks = masks.float().to(device).permute(0,3,1,2)
            latent, _ = autoencoder(pre_fire_imgs, post_fire_imgs)
            latent_resized = nn.functional.interpolate(latent, size=(512, 512), mode='bilinear', align_corners=False)
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
    
    print(f"Validation Loss: {avg_val_loss:.4f}")
    for t in thresholds:
        print(f"Threshold: {t} - F1 Score: {avg_f1_scores[t]:.4f}, IOU: {avg_iou_scores[t]:.4f}")
    
    return avg_val_loss, avg_f1_scores, avg_iou_scores


def train_and_evaluate(autoencoder, segmentation_model, dataloaders, device, epochs=50, lr=1e-3):
    optimizer = optim.Adam(segmentation_model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    best_f1 = 0.0
    best_iou = 0.0

    conv1x1 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1).to(device)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Training Phase
        segmentation_model.train()
        train_loss = 0.0
        train_f1 = 0.0
        train_iou = 0.0
        num_train_batches = len(dataloaders['train'])


        for batch in tqdm(dataloaders['train']):
            # print(f"Before empty_cache: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            # torch.cuda.empty_cache()  # Clear cache before each batch
            # print(f"After empty_cache: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            post_fire_imgs, pre_fire_imgs, masks= batch["post_fire_image"], batch["pre_fire_image"], batch["mask"]
            post_fire_imgs = post_fire_imgs.float().to(device)#.permute(0,3,1,2)
            pre_fire_imgs = pre_fire_imgs.float().to(device)#.permute(0,3,1,2)
            masks = masks.float().to(device)#.permute(0,3,1,2)
            # print(f"Size: post fire, pre fire, mask: {post_fire_imgs.size()}, {pre_fire_imgs.size()}, {masks.size()}")

            optimizer.zero_grad()

            with torch.no_grad():
                latent, _ = autoencoder(pre_fire_imgs, post_fire_imgs)
                latent_resized = nn.functional.interpolate(latent, size=(512, 512), mode='bilinear', align_corners=False)
                latent_new = conv1x1(latent_resized)
                combined_input = torch.cat((post_fire_imgs, latent_new), dim=1)
                # print(combined_input.size())

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

        # # Validate the model
        # val_loss, avg_f1_scores, avg_iou_scores = validate_model(segmentation_model, dataloaders['val'], device)
        # best_f1 = max(best_f1, max(avg_f1_scores.values()))
        # best_iou = max(best_iou, max(avg_iou_scores.values()))

        # Validation Phase
        segmentation_model.eval()
        val_loss = 0.0
        val_f1 = 0.0
        val_iou = 0.0
        num_val_batches = len(dataloaders['val'])
        thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        f1_scores = {t: [] for t in thresholds}
        iou_scores = {t: [] for t in thresholds}

        with torch.no_grad():
            for batch in tqdm(dataloaders['val']):
                post_fire_imgs, pre_fire_imgs, masks= batch["post_fire_image"], batch["pre_fire_image"], batch["mask"]
                post_fire_imgs = post_fire_imgs.float().to(device)#.permute(0,3,1,2)
                pre_fire_imgs = pre_fire_imgs.float().to(device)#.permute(0,3,1,2)
                masks = masks.float().to(device)#.permute(0,3,1,2)

                latent, _ = autoencoder(pre_fire_imgs, post_fire_imgs)
                latent_resized = nn.functional.interpolate(latent, size=(512, 512), mode='bilinear', align_corners=False)
                latent_new = conv1x1(latent_resized)
                combined_input = torch.cat((post_fire_imgs, latent_new), dim=1)

                outputs = segmentation_model(combined_input)
                loss = criterion(outputs, masks)

                val_loss += loss.item()
                for t in thresholds:
                    f1, iou = calculate_metrics(outputs, masks, threshold=t)
                    f1_scores[t].append(f1)
                    iou_scores[t].append(iou)
                # f1, iou = calculate_metrics(outputs, masks)
                # val_f1 += f1
                # val_iou += iou

        val_loss /= num_val_batches
        # val_f1 /= num_val_batches
        # val_iou /= num_val_batches
        avg_f1_scores = {t: sum(scores) / len(scores) for t, scores in f1_scores.items()}
        avg_iou_scores = {t: sum(scores) / len(scores) for t, scores in iou_scores.items()}
        for t in thresholds:
            print(f"Threshold: {t} - F1 Score: {avg_f1_scores[t]:.4f}, IOU: {avg_iou_scores[t]:.4f}")

        best_f1 = max(best_f1, max(avg_f1_scores.values()))
        best_iou = max(best_iou, max(avg_iou_scores.values()))
        # print(f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, Val IoU: {val_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f}")  

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
    test_dataset.dataset.transforms = val_transforms  # Apply val transforms to the test set for consistency
    # train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    # val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    # test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=False)

    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

    # Initialize autoencoder and segmentation model
    autoencoder = ModifiedAutoencoder().to(device)
    autoencoder.load_state_dict(torch.load("model_weights/modified_autoencoder.pth"))
    autoencoder.eval()  # Set autoencoder to evaluation mode

    segmentation_model = UNet(n_channels=16, n_classes=1, bilinear=False).to(device)

    # Train and evaluate the model
    trained_segmentation_model = train_and_evaluate(autoencoder, segmentation_model, dataloaders, device, epochs=50, lr=1e-3)

    # Save the trained segmentation model
    torch.save(trained_segmentation_model.state_dict(), "trained_segmentation_model.pth")
    print("Trained segmentation model saved successfully.")
