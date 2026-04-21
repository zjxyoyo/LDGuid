import torch
from torch import nn

class ReversedAutoencoder(nn.Module):
    def __init__(self):
        super(ReversedAutoencoder, self).__init__()
        # Encoder 保持不变：依然吃进去 pre 和 post 联合提取 latent (24 channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  
            nn.ReLU()
        )

        # 【核心修改 1】：现在处理器用来处理 Post-fire 图像
        self.post_fire_processor = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),  # 假设输入是 12 通道
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # 【核心修改 2】：Decoder 现在负责重建 Pre-fire 图像
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32 latent + 32 processed post-fire
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 12, kernel_size=3, stride=2, padding=1, output_padding=1),  # 输出 12 通道 (重建的 pre-fire)
            nn.Sigmoid()
        )

    def forward(self, pre_fire, post_fire):
        # 注意这里不直接写死 forward 逻辑，因为你在 train.py 里面是分步调用的。
        # 这里仅提供模块，具体的数据流向在 train.py 控制。
        pass