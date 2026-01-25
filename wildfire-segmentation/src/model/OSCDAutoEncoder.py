import torch
from torch import nn
import torch.nn.functional as F

class OSCDAE(nn.Module):
    """
    This AE correctly implements the user's design.
    It disentangles context (from pre-image) and change (from both).
    The decoder needs both to reconstruct the post-image.
    """
    def __init__(self, in_channels=12):
        super().__init__()
        
        # This encoder learns the "change" from the combined images
        self.change_encoder = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
            # The output here is the latent_change of shape [B, 128, H/4, W/4]
        )

        # This encoder processes the "context" from the pre-image
        self.context_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
            # The output here is the context_features of shape [B, 128, H/4, W/4]
        )

        # The decoder takes the COMBINED features to reconstruct the post-image
        # Input channels = 128 (from change) + 128 (from context) = 256
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, in_channels, kernel_size=1, stride=1, padding=0), # Use 1x1 conv for final layer
            nn.Sigmoid()
        )

    def forward(self, pre_image, post_image):
        # 1. Get the change information
        latent_change = self.change_encoder(torch.cat((pre_image, post_image), dim=1))
        
        # 2. Get the context information from the pre-image
        context_features = self.context_encoder(pre_image)
        
        # 3. Combine them to feed the decoder
        # This concatenation is the key to your design
        combined_features_for_decoder = torch.cat((latent_change, context_features), dim=1)
        
        # 4. Reconstruct the post-image
        reconstructed_post = self.decoder(combined_features_for_decoder)
        
        # We return the latent_change as that is the feature we want for the U-Net later
        return latent_change, reconstructed_post