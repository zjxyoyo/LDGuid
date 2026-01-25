import os
import sys

sys.path.append(os.getcwd())

import csv

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, jaccard_score
from torch import optim
from tqdm import tqdm

from data.load_data import POST_FIRE_DIR
from src.data_loader.dataloader import get_loader
from src.data_loader.dataset import WildfireDataset
from src.model.unet import UNet


# Function to load states
def load_checkpoint(checkpoint_name, model, scaler, optimizer):
    epoch = None
    if os.path.exists(checkpoint_name):

        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_name)
        else:
            checkpoint = torch.load(checkpoint_name, map_location=torch.device('cpu'))

        model.load_state_dict(checkpoint["model_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint.get("epoch", 0)
        print("Checkpoint loaded successfully.")
    else:
        print(f"Checkpoint '{checkpoint_name}' does not exist.")

    return epoch


# Function to update CSV file
def update_csv(csv_file, epoch, train_loss, train_iou, train_f1, test_loss, test_iou, test_f1):
    headers = ["epoch", "train_loss", "train_iou", "train_f1", "test_loss", "test_iou", "test_f1"]
    if not os.path.exists(csv_file):
        with open(csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(headers)
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([epoch, train_loss, train_iou, train_f1, test_loss, test_iou, test_f1])


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:0")
    print(device)
    dt = WildfireDataset(POST_FIRE_DIR, transforms=None)

    # Training Configs
    # Set the number of epochs
    NUM_EPOCHS = 80
    # Set the batch size number
    BATCH_SIZE = 8
    # Learning Rate
    LR = 3e-5
    # L2 Regularization
    WEIGHT_DECAY = 1e-5

    # Other configs
    # Directory to save model weights
    SAVE_PATH = "model_weights"
    # File to save metrics
    CSV_FILE = "training_metrics.csv"
    # Temp CSV file in case run gets killed
    CSV_TEMP_FILE = "training_metrics_temp.csv"
    # load checkpoint
    LOAD_CHECKPOINT = False
    # Checkpoint file name
    CHECKPOINT_NAME = "unet_epoch_2.chkpt"
    # Checkpoint Epoch - number of epochs between chkpt
    CHECKPOINT_EPOCH = 5

    # Create directory to save model weights
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # Initialize metrics dataframe
    metrics_df = pd.DataFrame(columns=["epoch", "train_loss", "val_loss", "train_f1", "val_f1", "train_iou", "val_iou"])

    dict(batch_size=BATCH_SIZE, num_workers=4)
    train_loader = get_loader(dt.train, is_train=True, loader_args=dict(batch_size=BATCH_SIZE, shuffle=True))
    val_loader = get_loader(dt.val, is_train=False, loader_args=dict(batch_size=BATCH_SIZE, shuffle=False))

    model = UNet(n_channels=12, n_classes=1, bilinear=False)
    model.to(device=device)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # Added a schedular
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
    criterion = torch.nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()

    CUR_EPOCH = 0
    # Load checkpoint
    if LOAD_CHECKPOINT:
        CUR_EPOCH = load_checkpoint(os.path.join(SAVE_PATH, CHECKPOINT_NAME), model, scaler, optimizer)

    # Define a threshold for F1 score
    THRESHOLD = 0.5
    for epoch in range(CUR_EPOCH, NUM_EPOCHS):
        model.train()
        train_loss, train_f1, train_iou = 0, 0, 0
        num_samples = 0
        train_loop = tqdm(train_loader, leave=True)
        for batch in train_loop:
            images, true_masks = batch["image"], batch["mask"]

            images = images.to(device)
            true_masks = true_masks.to(device=device, dtype=torch.long)

            # Autocast for mixed precision
            with torch.cuda.amp.autocast():
                masks_pred = model(images)
                loss = criterion(masks_pred, true_masks.float())

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Apply sigmoid to convert to probabilities and threshold to convert to binary mask
            preds_binary = (torch.sigmoid(masks_pred) > THRESHOLD).float()
            # Convert true masks to float
            true_masks_binary = true_masks.float()

            # Move tensors to CPU and then convert to NumPy for scikit-learn functions
            # Flatten the tensors and move to CPU
            preds_binary_np = preds_binary.view(-1).detach().cpu().numpy()
            true_masks_binary_np = true_masks_binary.view(-1).detach().cpu().numpy()

            # Ensure no NaNs or Infs
            np.nan_to_num(preds_binary_np, copy=False)
            np.nan_to_num(true_masks_binary_np, copy=False)

            # Calculate and update metrics
            batch_f1 = f1_score(true_masks_binary_np, preds_binary_np, average="macro")
            batch_iou = jaccard_score(true_masks_binary_np, preds_binary_np, average="macro")
            train_loss += loss.item() * images.size(0)
            train_f1 += batch_f1
            train_iou += batch_iou
            num_samples += images.size(0)

            train_loop.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
            train_loop.set_postfix(loss=train_loss / num_samples)

        # Validation step
        model.eval()
        val_loss, val_f1, val_iou, num_val_samples = 0, 0, 0, 0
        val_loop = tqdm(val_loader, leave=True, desc="Validation")
        with torch.no_grad():
            for batch in val_loop:
                images, true_masks = batch["image"], batch["mask"]

                images = images.to(device)
                true_masks = true_masks.to(device)
                with torch.cuda.amp.autocast():
                    masks_pred = model(images)
                    loss = criterion(masks_pred, true_masks.float())
                val_loss += loss.item() * images.size(0)

                # Calculate metrics
                preds_binary = (torch.sigmoid(masks_pred) > THRESHOLD).float()
                true_masks_binary = true_masks.float()

                # Move tensors to CPU and then convert to NumPy for scikit-learn functions
                # Flatten the tensors and move to CPU
                preds_binary_np = preds_binary.view(-1).detach().cpu().numpy()
                true_masks_binary_np = true_masks_binary.view(-1).detach().cpu().numpy()

                # Ensure no NaNs or Infs
                np.nan_to_num(preds_binary_np, copy=False)
                np.nan_to_num(true_masks_binary_np, copy=False)

                batch_f1 = f1_score(true_masks_binary_np, preds_binary_np, average="macro")
                batch_iou = jaccard_score(true_masks_binary_np, preds_binary_np, average="macro")

                val_f1 += batch_f1
                val_iou += batch_iou
                num_val_samples += images.size(0)
                val_loop.set_postfix(loss=val_loss / num_val_samples)

        # Step the learning rate scheduler
        scheduler.step()
        # Calculating average metrics
        avg_train_loss = train_loss / num_samples
        avg_val_loss = val_loss / num_val_samples
        avg_train_f1 = train_f1 / len(train_loader)
        avg_val_f1 = val_f1 / len(val_loader)
        avg_train_iou = train_iou / len(train_loader)
        avg_val_iou = val_iou / len(val_loader)

        # Saving metrics to dataframe
        new_row = pd.DataFrame(
            [
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "train_f1": avg_train_f1,
                    "val_f1": avg_val_f1,
                    "train_iou": avg_train_iou,
                    "val_iou": avg_val_iou,
                },
            ]
        )
        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
        update_csv(
            CSV_TEMP_FILE, epoch + 1, avg_train_loss, avg_train_iou, avg_train_f1, avg_val_loss, avg_val_iou, avg_val_f1
        )

        # Save model weights
        if (epoch + 1) % CHECKPOINT_EPOCH == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                },
                os.path.join(SAVE_PATH, f"unet_epoch_{epoch + 1}.chkpt"),
            )

    # Saving metrics to CSV file
    metrics_df.to_csv(CSV_FILE, index=False)

    print("Training completed and metrics saved.")
