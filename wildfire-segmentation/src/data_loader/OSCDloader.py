# OSCDloader.py

import os
import torch
from torch.utils.data import DataLoader

from oscd_dataset_class import Manual_OSCD_Dataset

def main():
    """主函数，用于测试手动数据加载器。"""
    print("--- 开始测试手动 OSCD DataLoader ---")

    # ==============================================================================
    # 唯一需要修改的地方就是这里：
    # 1. 定义数据所在的基础路径
    base_path = '~/projects/def-bereyhia/zjxeric/data'
    
    # 2. 构造数据集的完整根目录路径
    #    os.path.expanduser会把'~'正确地替换成你的家目录
    #    请确保 "Onera Satellite Change Detection" 是你解压后在 'data' 文件夹下的目录名
    DATASET_ROOT_PATH = os.path.join(os.path.expanduser(base_path), "Onera Satellite Change Detection")
    # ==============================================================================
    
    print(f"正在从以下路径加载数据: {DATASET_ROOT_PATH}")

    # 使用我们自定义的类来实例化 PyTorch Dataset
    # 这里我们加载训练集 'train'
    train_dataset = Manual_OSCD_Dataset(root_dir=DATASET_ROOT_PATH, split='train')
    
    # 创建 DataLoader
    # 注意：因为I/O现在是从 $PROJECT 目录读取，可能会比 $SCRATCH 稍慢
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    
    print(f"成功加载 {len(train_dataset)} 个训练样本。")

    # 从 DataLoader 中取出一个批次的数据进行测试
    print("\n--- 正在从 DataLoader 中获取一个样本批次... ---")
    sample_batch = next(iter(train_loader))
    print("成功获取一个批次！")
    
    # 检查数据维度是否正确
    print("\n--- 批次数据形状检查 ---")
    print(f"Pre-image batch shape: {sample_batch['pre_image'].shape}")
    print(f"Post-image batch shape: {sample_batch['post_image'].shape}")
    print(f"Mask batch shape: {sample_batch['mask'].shape}")
    print("\n--- 手动 DataLoader 测试成功！ ---")


if __name__ == "__main__":
    main()