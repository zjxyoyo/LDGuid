import torch
from torch import nn
import torch.nn.functional as F

class ChangeDetectionAE(nn.Module):
    """
    This AE disentangles context (from pre-image) and change (from both).
    The `in_channels` parameter makes it flexible for both 12-channel (OSCD)
    and 3-channel (SVCD) data.
    """
    def __init__(self, in_channels=3): # We'll default to 3 for SVCD
        super().__init__()
        
        # This encoder learns "change". Its input will be 3+3=6 channels for SVCD.
        self.change_encoder = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        # This encoder processes "context". Its input will be 3 channels for SVCD.
        self.context_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        # The decoder reconstructs a 3-channel image for SVCD.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, in_channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    # The forward pass logic is perfectly general and does not need to change.
    def forward(self, pre_image, post_image):
        latent_change = self.change_encoder(torch.cat((pre_image, post_image), dim=1))
        context_features = self.context_encoder(pre_image)
        combined_features_for_decoder = torch.cat((latent_change, context_features), dim=1)
        reconstructed_post = self.decoder(combined_features_for_decoder)
        return latent_change, reconstructed_post