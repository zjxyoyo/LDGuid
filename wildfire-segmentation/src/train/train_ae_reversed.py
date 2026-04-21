import os
import sys
import csv
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data_loader.dataset import WildfireDataset, CombinedDataset
from src.model.reversed_autoencoder import ReversedAutoencoder

from src.data_loader.augmentation import (
    CustomColorJitter, DoubleAffine, DoubleCompose,
    DoubleElasticTransform, DoubleHorizontalFlip,
    DoubleToTensor, DoubleVerticalFlip, GaussianNoise,
)

from data.load_data import POST_FIRE_DIR, PRE_POST_FIRE_DIR

# --- 配置保存路径 ---
BASE_SAVE_DIR = "./results/reversed_autoencoder"
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(BASE_SAVE_DIR, "best_reversed_ae.pth")
LOG_CSV_PATH = os.path.join(BASE_SAVE_DIR, "training_log_reversed.csv")

# 初始化 CSV 文件头
with open(LOG_CSV_PATH, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'Total_Loss', 'Recon_Loss'])

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transforms = DoubleCompose([
        DoubleToTensor(),
        DoubleElasticTransform(alpha=250, sigma=10),
        DoubleHorizontalFlip(),
        DoubleVerticalFlip(),
        DoubleAffine(degrees=(-15, 15), translate=(0.15, 0.15), scale=(0.8, 1))
    ])

    dataset = WildfireDataset(PRE_POST_FIRE_DIR, transforms=None)
    combined_dataset = CombinedDataset(dataset, transforms=None)
    dataloader = DataLoader(combined_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)

    autoencoder = ReversedAutoencoder().to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    num_epochs = 50 
    best_loss = float('inf')  # 用于追踪最小的 Recon Loss
    
    # 这里的权重系数可以根据你的需要调整，目前暂定为 0.5
    ADV_WEIGHT = 0.5 

    for epoch in range(num_epochs):
        running_loss = 0.0
        reconstruction_loss_accum = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_data in pbar:
            # batch_data 现在是一个字典，我们通过 key 把对应的 tensor 提出来
            pre_fire_imgs = batch_data["pre_fire_image"].to(device).float()
            post_fire_imgs = batch_data["post_fire_image"].to(device).float()

            pre_fire_imgs = pre_fire_imgs.permute(0, 3, 1, 2)
            post_fire_imgs = post_fire_imgs.permute(0, 3, 1, 2)

            optimizer.zero_grad()

            # ---------------------------------------------------------
            # 1. 正常重建路径 (Post -> Pre)
            # ---------------------------------------------------------
            combined_input = torch.cat((pre_fire_imgs, post_fire_imgs), dim=1)
            latent = autoencoder.encoder(combined_input)
            
            post_fire_processed = autoencoder.post_fire_processor(post_fire_imgs)
            post_fire_resized = nn.functional.interpolate(post_fire_processed, size=latent.shape[2:], mode='bilinear', align_corners=False)
            
            reconstructed_pre = autoencoder.decoder(torch.cat((latent, post_fire_resized), dim=1))
            reconstruct_loss = criterion(reconstructed_pre, pre_fire_imgs)

            # ---------------------------------------------------------
            # 2. 对抗/零掩码路径
            # ---------------------------------------------------------
            zero_mask = torch.zeros_like(post_fire_imgs).to(device)
            zero_mask_processed = autoencoder.post_fire_processor(zero_mask)
            zero_mask_resized = nn.functional.interpolate(zero_mask_processed, size=latent.shape[2:], mode='bilinear', align_corners=False)
            
            reconstructed_with_zero_mask = autoencoder.decoder(torch.cat((latent, zero_mask_resized), dim=1))
            zero_mask_loss = criterion(reconstructed_with_zero_mask, pre_fire_imgs)

            # ---------------------------------------------------------
            # 3. 总损失与反向传播
            # ---------------------------------------------------------
            loss = reconstruct_loss - ADV_WEIGHT * zero_mask_loss
            
            loss.backward()
            optimizer.step()

            reconstruction_loss_accum += reconstruct_loss.item() * pre_fire_imgs.size(0)
            running_loss += loss.item() * pre_fire_imgs.size(0)
            
            # 更新进度条显示的 loss
            pbar.set_postfix({'loss': loss.item(), 'recon': reconstruct_loss.item()})

        # 计算每个 epoch 的平均 Loss
        avg_epoch_loss = running_loss / len(combined_dataset)
        avg_recon_loss = reconstruction_loss_accum / len(combined_dataset)

        print(f"Epoch [{epoch+1}/{num_epochs}] | Total Loss: {avg_epoch_loss:.4f} | Recon Loss: {avg_recon_loss:.4f}")

        # --- 记录到 CSV ---
        with open(LOG_CSV_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_epoch_loss, avg_recon_loss])

        # --- 保存 Best Model ---
        if avg_recon_loss < best_loss:
            best_loss = avg_recon_loss
            torch.save(autoencoder.state_dict(), MODEL_SAVE_PATH)
            print(f"[*] New best model saved with Recon Loss: {best_loss:.4f}")