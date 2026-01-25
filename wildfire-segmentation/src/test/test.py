import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import torch
from sklearn.metrics import f1_score, jaccard_score
from torch import optim
from tqdm import tqdm

from data.load_data import POST_FIRE_DIR, PRE_POST_FIRE_DIR
from src.data_loader.dataloader import get_loader
from src.data_loader.dataset import WildfireDataset
from src.model.unet import UNet

from src.train.train import load_checkpoint


if __name__ == "__main__":
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0")
    print(device)
    dt = WildfireDataset(POST_FIRE_DIR, transforms=None)

    # Directory to save model weights
    SAVE_PATH = "model_weights"

    # Checkpoint file name
    CHECKPOINT_NAME = "pre_trained.chkpt"
   

    # Create directory to save model weights
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)


    dict(batch_size=8, num_workers=4)
    test_loader = get_loader(dt.test, is_train=False, loader_args=dict(batch_size=8, shuffle=False))

    model = UNet(n_channels=12, n_classes=1, bilinear=False)
    model.to(device=device)

    optimizer = optim.Adam(model.parameters())
    criterion = torch.nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()

    # Load checkpoint 
    CUR_EPOCH = load_checkpoint(os.path.join(SAVE_PATH, CHECKPOINT_NAME), model, scaler, optimizer)

    # Define a threshold for F1 score
    THRESHOLD = 0.5
    # Validation step
    model.eval()
    test_loss, test_f1, test_iou, num_test_samples = 0, 0, 0, 0
    test_loop = tqdm(test_loader, leave=True, desc="Test")
    with torch.no_grad():
        for batch in test_loop:
            images, true_masks = batch["image"], batch["mask"]

            images = images.to(device)
            true_masks = true_masks.to(device)
            with torch.cuda.amp.autocast():
                masks_pred = model(images)
                loss = criterion(masks_pred, true_masks.float())
            test_loss += loss.item() * images.size(0)

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

            test_f1 += batch_f1
            test_iou += batch_iou
            num_test_samples += images.size(0)
            test_loop.set_postfix(loss=test_loss / num_test_samples)

        # Calculating average metrics
        avg_test_loss = test_loss / num_test_samples
        avg_test_f1 = test_f1 / len(test_loader)
        avg_test_iou = test_iou / len(test_loader)

        

        print("Test loss ", avg_test_loss, "Test f1 ", avg_test_f1, "Test IOU ", avg_test_iou)