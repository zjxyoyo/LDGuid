import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import cv2
from sklearn.metrics import jaccard_score
from argparse import ArgumentParser

# 确保能引用到项目根目录
sys.path.append(os.getcwd())

# ===================================================================
# 1. Imports (关键修改：同时导入两个版本的网络定义)
# ===================================================================
try:
    # Baseline 使用原始的 BIT 定义
    from models.networks import BASE_Transformer as Baseline_BIT_Class
    
    # Ours 使用修改后的 BIT 定义 (支持 Latent 注入，通道数可能不同)
    from models.networks_new import BASE_Transformer as Ours_BIT_Class
    
    from datasets.CD_dataset import CDDataset
    from models.ae_model import ChangeDetectionAE
except ImportError as e:
    print("Error importing project modules.")
    print(f"Details: {e}")
    sys.exit(1)

# ===================================================================
# 2. Configuration
# ===================================================================
# 路径配置
DATASET_ROOT_PATH = "/home/zjxeric/projects/def-bereyhia/zjxeric/data/LEVIR" 

# 权重路径
BASELINE_WEIGHTS = "/home/zjxeric/projects/def-bereyhia/zjxeric/BIT_CD/checkpoints/CD_BIT_LEVIR/best_ckpt.pt" 
OURS_WEIGHTS = "/home/zjxeric/projects/def-bereyhia/zjxeric/BIT_CD/checkpoints/CD_LatentBIT_LEVIR_Experiment/best_latent_bit_levir.pth" 
AE_WEIGHTS = "/home/zjxeric/projects/def-bereyhia/zjxeric/wildfire-segmentation/results/levir_autoencoder/best_ae_levir.pth"

OUTPUT_DIR = "viz_levir_comparison_final"

# 参数配置
IMG_SIZE = 256
IMG_CHANNELS = 3
AE_LATENT_CHANNELS = 128
LATENT_CHANNELS_COMPRESSED = 4

# Baseline 参数 (原始 BIT)
BACKBONE_CHANNELS = 32 
TOKENIZER_IN_CHANNELS_BASE = 32 

# Ours 参数 (注入了 Latent)
TOKENIZER_IN_CHANNELS_OURS = 36  # 32 + 4

N_CLASSES = 2
RESNET_STAGES_NUM = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================================================================
# 3. LatentBIT Class (Ours 的包装器)
# ===================================================================
class LatentBIT(nn.Module):
    def __init__(self, pretrained_ae, bit_model, compression_layer):
        super().__init__()
        self.autoencoder = pretrained_ae
        self.bit_model = bit_model
        self.compression = compression_layer
        
        self.autoencoder.eval()
        for param in self.autoencoder.parameters():
            param.requires_grad = False

    def forward(self, pre_image, post_image):
        with torch.no_grad():
            latent_change, _ = self.autoencoder(pre_image, post_image)
        
        compressed_latent = self.compression(latent_change)

        feature_A = self.bit_model.forward_single(pre_image)
        feature_B = self.bit_model.forward_single(post_image)
        
        aligned_latent = F.interpolate(compressed_latent, size=feature_A.shape[2:], mode='bilinear', align_corners=False)
        
        enriched_feature_A = torch.cat((feature_A, aligned_latent), dim=1)
        enriched_feature_B = torch.cat((feature_B, aligned_latent), dim=1)

        output = self.bit_model.forward_transformer(enriched_feature_A, enriched_feature_B)
        return output

# ===================================================================
# 4. Helper Functions
# ===================================================================
def create_fancy_error_map(pred, gt):
    pred = pred.squeeze()
    gt = gt.squeeze()
    if pred.ndim > 2: pred = pred[:, :, 0]
    if gt.ndim > 2: gt = gt[:, :, 0]

    h, w = pred.shape
    color_map = np.zeros((h, w, 3), dtype=np.uint8)
    
    # White: TP (Correct Change)
    color_map[(pred==1) & (gt==1)] = [255, 255, 255]
    # Green: FP (False Alarm)
    color_map[(pred==1) & (gt==0)] = [0, 255, 0] 
    # Red: FN (Missed Change)
    color_map[(pred==0) & (gt==1)] = [0, 0, 255]
    
    return color_map

def denormalize_image(tensor):
    img = tensor.permute(1, 2, 0).cpu().numpy()
    if img.min() < 0: # Check if normalized to [-1, 1]
        img = (img * 0.5) + 0.5
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)

# ===================================================================
# 5. Main Loop
# ===================================================================
def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    print(f"--- Visualization Script ---")
    print(f"Output Directory: {OUTPUT_DIR}")

    # --- A. Load Dataset ---
    print("Loading LEVIR Test Set...")
    test_dataset = CDDataset(
        root_dir=DATASET_ROOT_PATH, 
        split='test',  
        img_size=IMG_SIZE, 
        is_train=False, 
        label_transform='norm'
    )
    print(f"Samples: {len(test_dataset)}")

    # --- B. Load Baseline Model (Using Baseline_BIT_Class) ---
    print("Loading Baseline BIT (models.networks)...")
    base_model = Baseline_BIT_Class(
        input_nc=3, 
        output_nc=N_CLASSES, 
        with_pos='learned', 
        resnet_stages_num=RESNET_STAGES_NUM,
        enc_depth=1, 
        dec_depth=8 # 关键修正：对应 dd8
        # tokenizer_in_channels=TOKENIZER_IN_CHANNELS_BASE, # 32
        # dim=TOKENIZER_IN_CHANNELS_BASE
    ).to(device)
    
    try:
        ckpt = torch.load(BASELINE_WEIGHTS, map_location=device, weights_only=False)
        # 兼容不同的保存方式
        if 'model_G_state_dict' in ckpt:
             base_model.load_state_dict(ckpt['model_G_state_dict'])
        elif 'net_G' in ckpt:
            base_model.load_state_dict(ckpt['net_G'])
        else:
            base_model.load_state_dict(ckpt)
        print("Baseline Loaded.")
    except Exception as e:
        print(f"Error loading Baseline: {e}")

    # --- C. Load Ours Model (Using Ours_BIT_Class) ---
    print("Loading Ours LatentBIT (models.networks_new)...")
    
    ae = ChangeDetectionAE(in_channels=IMG_CHANNELS).to(device)
    ae.load_state_dict(torch.load(AE_WEIGHTS, map_location=device, weights_only=False))

    # 这里的 bit_for_ours 必须用 networks_new 里的类
    bit_for_ours = Ours_BIT_Class(
        input_nc=3, output_nc=N_CLASSES, with_pos='learned', 
        resnet_stages_num=RESNET_STAGES_NUM,
        tokenizer_in_channels=TOKENIZER_IN_CHANNELS_OURS, # 36
        dim=TOKENIZER_IN_CHANNELS_OURS
    ).to(device)

    compression = nn.Conv2d(AE_LATENT_CHANNELS, LATENT_CHANNELS_COMPRESSED, kernel_size=1).to(device)
    ours_model = LatentBIT(ae, bit_for_ours, compression).to(device)

    try:
        checkpoint = torch.load(OURS_WEIGHTS, map_location=device, weights_only=False)
        if 'bit_model_state_dict' in checkpoint:
            ours_model.bit_model.load_state_dict(checkpoint['bit_model_state_dict'])
            ours_model.compression.load_state_dict(checkpoint['compression_state_dict'])
        else:
            ours_model.load_state_dict(checkpoint)
        print("Ours Loaded.")
    except Exception as e:
        print(f"Error loading Ours: {e}")

    base_model.eval()
    ours_model.eval()

    results = []
    print("Starting Inference...")

    for i in tqdm(range(len(test_dataset))):
        try:
            batch = test_dataset[i]
            
            pre_t = batch['A'].unsqueeze(0).to(device)
            post_t = batch['B'].unsqueeze(0).to(device)
            mask_t = batch['L'].unsqueeze(0).to(device)
            mask_np = mask_t.squeeze().cpu().numpy()

            if np.sum(mask_np) < 50: continue

            # Baseline
            with torch.no_grad():
                base_out = base_model(pre_t, post_t)
                base_pred = torch.argmax(base_out, dim=1).float().cpu().numpy().squeeze()
            
            # Ours
            with torch.no_grad():
                ours_out = ours_model(pre_t, post_t)
                ours_pred = torch.argmax(ours_out, dim=1).float().cpu().numpy().squeeze()

            iou_base = jaccard_score(mask_np.flatten(), base_pred.flatten(), average='binary')
            iou_ours = jaccard_score(mask_np.flatten(), ours_pred.flatten(), average='binary')
            delta = iou_ours - iou_base
            
            # 筛选条件：Base 凑合，Ours 更好
            if iou_base > 0.25 and delta > 0.1 and iou_ours > 0.75:
                pre_img_save = denormalize_image(batch['A'])
                post_img_save = denormalize_image(batch['B'])
                
                results.append({
                    'id': i,
                    'pre': pre_img_save, 'post': post_img_save, 'mask': mask_np,
                    'base_pred': base_pred, 'ours_pred': ours_pred, 'delta': delta,
                    'score_str': f"Base={iou_base:.2f}_Ours={iou_ours:.2f}"
                })
        except Exception as e:
            continue

    print(f"Found {len(results)} candidates. Saving Top 5...")
    results.sort(key=lambda x: x['delta'], reverse=True)
    
    for idx, item in enumerate(results[:5]):
        folder = os.path.join(OUTPUT_DIR, f"Rank{idx+1}_{item['score_str']}")
        os.makedirs(folder, exist_ok=True)
        
        cv2.imwrite(f"{folder}/1_Pre.png", cv2.cvtColor(item['pre'], cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{folder}/2_Post.png", cv2.cvtColor(item['post'], cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{folder}/3_GT.png", (item['mask'] * 255).astype(np.uint8))
        
        base_err = create_fancy_error_map(item['base_pred'], item['mask'])
        ours_err = create_fancy_error_map(item['ours_pred'], item['mask'])
        
        cv2.imwrite(f"{folder}/4_BIT_Error.png", base_err) 
        cv2.imwrite(f"{folder}/5_Ours_LatentBIT_Error.png", ours_err)
        
    print(f"Done! Check {OUTPUT_DIR}")

if __name__ == "__main__":
    main()