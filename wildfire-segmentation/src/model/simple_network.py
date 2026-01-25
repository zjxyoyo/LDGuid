import torch
from torch import nn
import torch.nn.functional as F

class SimpleSegmentationNetwork(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SimpleSegmentationNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, output_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(output_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.upsample(F.relu(self.bn3(self.conv3(x))))
        return x
