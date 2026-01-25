import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(gating_channels, inter_channels, kernel_size=1), nn.BatchNorm2d(inter_channels)
        )

        self.W_x = nn.Sequential(nn.Conv2d(in_channels, inter_channels, kernel_size=1), nn.BatchNorm2d(inter_channels))

        self.psi = nn.Sequential(nn.Conv2d(inter_channels, 1, kernel_size=1), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(F.relu(g1 + x1))
        return x * psi


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, dilation=1, enable_shortcut=False):
        super().__init__()
        self.enable_shortcut = enable_shortcut
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, dilation=dilation, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        if self.enable_shortcut:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        if self.enable_shortcut:
            return self.double_conv(x) + self.shortcut(x)
        else:
            return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, enable_dropout=False, enable_attention_gate=False):
        super().__init__()
        self.enable_attention_gate = enable_attention_gate
        self.enable_dropout = enable_dropout

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

        if self.enable_dropout:
            self.dropout = nn.Dropout(0.5)

        if self.enable_attention_gate:
            # Adjusted channel sizes
            self.attention_gate = AttentionGate(out_channels, in_channels // 2, out_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        if self.enable_attention_gate:
            # Apply attention gate
            x2 = self.attention_gate(x2, x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)

        if self.enable_dropout:
            return self.dropout(self.conv(x))
        else:
            return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
