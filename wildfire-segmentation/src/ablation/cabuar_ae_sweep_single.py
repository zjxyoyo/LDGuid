import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv

from src.data_loader.dataset import WildfireDataset, CombinedDataset
from src.model.modified_autoencoder2 import ModifiedAutoencoder
from src.data_loader.augmentation import (
    DoubleAffine,
    DoubleCompose,
    DoubleElasticTransform,
    DoubleHorizontalFlip,
    DoubleVerticalFlip,
    DoubleToTensor
)
from data.load_data import POST_FIRE_DIR, PRE_POST_FIRE_DIR

# --- 1. 获取命令行参数 (Alpha) ---
try:
    # 从命令行读取 alpha 值，例如: python script.py 1.2
    CURRENT_ADV_WEIGHT = float(sys.argv[1])
except IndexError:
    print("Error: Please provide an adversarial weight argument.")
    print("Usage: python -m src.train.cabuar_ae_sweep_single <alpha>")
    sys.exit(1)

# --- 2. 配置路径和参数 ---
BASE_SAVE_DIR = "./results/cabuar_autoencoder_sweep"

SAVE_DIR = os.path.join(BASE_SAVE_DIR, f"alpha_{CURRENT_ADV_WEIGHT}")
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, f"best_ae_cabuar_alpha_{CURRENT_ADV_WEIGHT}.pth")
LOG_CSV_PATH = os.path.join(SAVE_DIR, f"training_log_alpha_{CURRENT_ADV_WEIGHT}.csv")

os.makedirs(SAVE_DIR, exist_ok=True)

NUM_EPOCHS = 50
BATCH_SIZE = 8

def main():
    print(f"{'='*20} STARTING TRAINING WITH ALPHA = {CURRENT_ADV_WEIGHT} {'='*20}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define transforms 
    transforms = DoubleCompose([
        DoubleToTensor(),
        DoubleElasticTransform(alpha=250, sigma=10),
        DoubleHorizontalFlip(),
        DoubleVerticalFlip(),
        DoubleAffine(degrees=(-15, 15), translate=(0.15, 0.15), scale=(0.8, 1))
    ])

    # Load dataset
    print("Loading CaBuAr dataset...")
    dataset = WildfireDataset(PRE_POST_FIRE_DIR, transforms=None)
    combined_dataset = CombinedDataset(dataset, transforms=None)
    dataloader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # Initialize autoencoder
    autoencoder = ModifiedAutoencoder().to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # --- 初始化 CSV Logger ---
    with open(LOG_CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'total_loss', 'recon_loss'])

    best_loss = float('inf')

    # Training loop
    for epoch in range(NUM_EPOCHS):
        autoencoder.train()
        running_loss = 0.0
        reconstruction_loss_accum = 0.0
        
        # 使用 tqdm 显示进度
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Alpha={CURRENT_ADV_WEIGHT})")
        
        for batch in pbar:
            # 注意：这里根据你的 Dataset 实现，key 可能需要确认一下
            # 如果报错 KeyError，请检查 CombinedDataset 的 __getitem__ 返回的 keys
            post_fire_imgs = batch["post_fire_image"].float().to(device)
            pre_fire_imgs = batch["pre_fire_image"].float().to(device)
            
            # 如果 dataset 没有做 permute，这里需要做。
            # DoubleToTensor 通常已经把 HWC 变成了 CHW。
            # 假设 DoubleToTensor 输出已经是 (C, H, W)，这里就不需要 permute(0,3,1,2)
            # 如果你的原代码里有 permute，说明你的 Dataset 输出可能是 numpy array HWC。
            # 为了保险，保留你原代码的逻辑，但请注意 tensor转换
            if post_fire_imgs.ndim == 4 and post_fire_imgs.shape[-1] == 12: # (B, H, W, C) check
                 post_fire_imgs = post_fire_imgs.permute(0, 3, 1, 2)
                 pre_fire_imgs = pre_fire_imgs.permute(0, 3, 1, 2)

            optimizer.zero_grad()

            # 1. Forward pass (Reconstruction)
            latent = autoencoder.encoder(torch.cat((pre_fire_imgs, post_fire_imgs), dim=1))
            
            # Pre-fire processing logic (from your original code)
            pre_fire_processed = autoencoder.pre_fire_processor(pre_fire_imgs)
            pre_fire_resized = nn.functional.interpolate(pre_fire_processed, size=latent.shape[2:], mode='bilinear', align_corners=False)
            reconstructed = autoencoder.decoder(torch.cat((latent, pre_fire_resized), dim=1))
            
            reconstruct_loss = criterion(reconstructed, post_fire_imgs)

            # 2. Adversarial pass (Zero Mask)
            zero_mask = torch.zeros_like(pre_fire_imgs).to(device)
            zero_mask_processed = autoencoder.pre_fire_processor(zero_mask)
            zero_mask_resized = nn.functional.interpolate(zero_mask_processed, size=latent.shape[2:], mode='bilinear', align_corners=False)
            reconstructed_with_zero_mask = autoencoder.decoder(torch.cat((latent, zero_mask_resized), dim=1))
            
            zero_mask_loss = criterion(reconstructed_with_zero_mask, post_fire_imgs)

            # --- 关键修改：使用传入的 CURRENT_ADV_WEIGHT ---
            # 原代码: loss = reconstruct_loss - 0.5 * zero_mask_loss
            loss = reconstruct_loss - CURRENT_ADV_WEIGHT * zero_mask_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()

            reconstruction_loss_accum += reconstruct_loss.item()
            running_loss += loss.item()
            
            # 更新进度条显示的 loss
            pbar.set_postfix({'loss': loss.item(), 'recon': reconstruct_loss.item()})

        # Calculate epoch averages
        avg_epoch_loss = running_loss / len(dataloader)
        avg_recon_loss = reconstruction_loss_accum / len(dataloader)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Total Loss: {avg_epoch_loss:.4f} | Recon Loss: {avg_recon_loss:.4f}")

        # --- 记录到 CSV ---
        with open(LOG_CSV_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_epoch_loss, avg_recon_loss])

        # Save model (Optional: Save best based on Recon Loss)
        # 这里为了简单，我们每轮都保存，或者保存最后一个
        if avg_recon_loss < best_loss:
            best_loss = avg_recon_loss
            torch.save(autoencoder.state_dict(), MODEL_SAVE_PATH)
            # print("Saved best model.")

    print(f"Training finished for alpha={CURRENT_ADV_WEIGHT}. Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()