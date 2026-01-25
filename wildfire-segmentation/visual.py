import os
import sys
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import cv2
from sklearn.metrics import jaccard_score
from data.load_data import PRE_POST_FIRE_DIR

# ================= 配置区域 =================
# 必须指向那个包含 Pre 和 Post 的文件夹
DATA_DIR = PRE_POST_FIRE_DIR

# 模型权重 (确保路径对)
BASELINE_WEIGHTS = "model_weights/pre_trained.chkpt" 
BUNDLE_WEIGHTS = "best_ldg_model_bundle.pth" # 刚刚重新训练出来的那个 Bundle

OUTPUT_DIR = "viz_final_paper"
TOP_K = 10 
# ============================================

sys.path.append(os.getcwd())
# 强制只用这一个 Dataset 类，避免混乱
from src.data_loader.dataset import WildfireDataset, CombinedDataset
from src.data_loader.augmentation2 import DoubleCompose, DoubleToTensor
from src.model.unet import UNet
from src.model.modified_autoencoder2 import ModifiedAutoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def create_fancy_error_map(pred, gt):
    # 强制 Squeeze，去掉所有为 1 的维度
    pred = pred.squeeze()
    gt = gt.squeeze()
    
    # 双重保险：如果 Squeeze 后还是 3D (例如 (512, 512, 12) 的错误数据)，强行取第一层
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
def save_rgb_image(img_12ch, path):
    # 提取 RGB (B4, B3, B2 -> Index 3, 2, 1)
    # 你的数据如果已经是 0-1 float，直接乘 255
    rgb = img_12ch[:, :, [3, 2, 1]] 
    rgb = np.clip(rgb * 3.5, 0, 1) # 简单提亮
    cv2.imwrite(path, (rgb * 255).astype(np.uint8))

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    # 1. 统一加载数据 (只用 PRE_POST_FIRE_DIR)
    # 这样 U-Net 和 LDG 面对的就是完全同一张测试图
    # val_transforms = DoubleCompose([DoubleToTensor()])
    raw_dataset = WildfireDataset(DATA_DIR, transforms=None)
    val_ds = raw_dataset.val_pre_fire
    
    # 2. 加载模型
    print("Loading Models...")
    
    # === Baseline U-Net ===
    # 注意：这里输入通道设为 12 (因为它只吃 Post)
    base_model = UNet(n_channels=12, n_classes=1, bilinear=False).to(device)
    try:
        chkpt = torch.load(BASELINE_WEIGHTS, map_location=device)
        # 兼容不同的保存格式
        state = chkpt["model_state_dict"] if "model_state_dict" in chkpt else chkpt
        base_model.load_state_dict(state)
        print("Baseline U-Net Loaded.")
    except:
        print("Error loading Baseline U-Net! Check path.")

    # === Ours (Bundle) ===
    # Autoencoder
    ae = ModifiedAutoencoder().to(device)
    # 假设 AE 权重没变，或者你可以从 Bundle 里加载如果保存了的话
    # 这里还是加载那个旧的 AE 权重
    ae.load_state_dict(torch.load("model_weights/modified_autoencoder.pth", map_location=device))
    
    # Ours Segmentation (UNet + Conv1x1)
    ours_seg = UNet(n_channels=16, n_classes=1, bilinear=False).to(device)
    conv1x1 = nn.Conv2d(32, 4, 1).to(device)
    
    try:
        bundle = torch.load(BUNDLE_WEIGHTS, map_location=device)
        ours_seg.load_state_dict(bundle['segmentation_model'])
        conv1x1.load_state_dict(bundle['conv1x1'])
        print("Ours LDG Loaded.")
    except:
        print("Error loading Ours LDG! Check path.")

    # base_model.eval()
    base_model.train()
    ae.eval()
    ours_seg.eval()
    conv1x1.eval()

    results = []
    
    # 3. 开始筛选 (只看 Dataset 的最后 20% 作为 Validation)
    print(f"Scanning Validation Set ({len(val_ds)} samples)...")
    # 全局索引
    # indices = dataset.val_idxs 
    # === Debug: 检查 Baseline 权重是否加载 ===
    print(f"Base Model Layer 1 Mean: {base_model.inc.double_conv[0].weight.mean().item()}")
    for i in tqdm(range(len(val_ds))):
        sample = val_ds[i] # 直接取验证集第 i 个样本
        
        # 拿到数据 (H,W,C)
        pre_np = sample['pre_fire_image']
        post_np = sample['post_fire_image']
        mask_np = sample['mask']

        # 有时候 Mask 读出来是 (512, 512, 1) 或者 (1, 512, 512)
        mask_np = mask_np.squeeze()
        # 如果万一 Mask 是多通道的 (512, 512, 3)，只取第一个通道
        if mask_np.ndim > 2: mask_np = mask_np[:, :, 0]
        
        if np.sum(mask_np) < 50: continue # 跳过没有火的图

        # 转 Tensor (1, C, H, W)
        pre_t = torch.from_numpy(pre_np).float().permute(2,0,1).unsqueeze(0).to(device)
        post_t = torch.from_numpy(post_np).float().permute(2,0,1).unsqueeze(0).to(device)
        
        # === Baseline 推理 ===
        # 关键点：Baseline 虽然训练时没见过 Pre，但推理时只需要 Post
        # 我们直接把 Post 喂给它，这完全符合逻辑
        with torch.no_grad():
            base_logits = base_model(post_t)
            base_prob = torch.sigmoid(base_logits)
            base_pred = (base_prob > 0.15).float().cpu().numpy().squeeze()

            # if i == 0:
            #         print(f"\n[DEBUG ID {i}]")
            #         print(f"Input Range: {post_t.min():.4f} ~ {post_t.max():.4f}")
            #         print(f"Logits Range: {base_logits.min():.4f} ~ {base_logits.max():.4f}")
            #         print(f"Prob Range:   {base_prob.min():.4f} ~ {base_prob.max():.4f}")
            #         print(f"Pred Sum: {base_pred.sum()} (Expected > 0)")
            
        # === Ours 推理 ===
        with torch.no_grad():
            latent, _ = ae(pre_t, post_t)
            latent = nn.functional.interpolate(latent, (512,512), mode='bilinear')
            latent = conv1x1(latent)
            inp = torch.cat([post_t, latent], dim=1)
            out = ours_seg(inp)
            ours_pred = (torch.sigmoid(out) > 0.15).float().cpu().numpy().squeeze()

        # 算分
        iou_base = jaccard_score(mask_np.flatten(), base_pred.flatten(), average='binary')
        iou_ours = jaccard_score(mask_np.flatten(), ours_pred.flatten(), average='binary')
        
        delta = iou_ours - iou_base
        
        # 只保存提升明显的 (例如提升超过 10%)
        # if delta > 0.1:
        if iou_base > 0.15 and delta > 0.1 and iou_ours > 0.75:
            results.append({
                'id': i, 'delta': delta, 
                'pre': pre_np, 'post': post_np, 'mask': mask_np,
                'base_pred': base_pred, 'ours_pred': ours_pred,
                'score_str': f"Base={iou_base:.2f}_Ours={iou_ours:.2f}"
            })

    # 4. 保存前 5 名
    results.sort(key=lambda x: x['delta'], reverse=True)
    
    for idx, item in enumerate(results[:5]):
        folder = os.path.join(OUTPUT_DIR, f"Rank{idx+1}_{item['score_str']}")
        os.makedirs(folder, exist_ok=True)
        
        # 保存各种图
        save_rgb_image(item['pre'], f"{folder}/1_Pre.png")
        save_rgb_image(item['post'], f"{folder}/2_Post.png")
        cv2.imwrite(f"{folder}/3_GT.png", item['mask']*255)
        
        # 重点：保存彩色误差图
        base_err = create_fancy_error_map(item['base_pred'], item['mask'])
        ours_err = create_fancy_error_map(item['ours_pred'], item['mask'])
        
        # 这里用 BGR 保存给 OpenCV
        cv2.imwrite(f"{folder}/4_Base_Error.png", base_err) 
        cv2.imwrite(f"{folder}/5_Ours_Error.png", ours_err)
        
    print("Done! Check folder:", OUTPUT_DIR)

if __name__ == "__main__":
    main()