import torch
from torch import nn

class ModifiedAutoencoder(nn.Module):
    def __init__(self):
        super(ModifiedAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(24, 64, kernel_size=3, stride=2, padding=1),  # 24 channels for concatenated pre and post images
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256 + 12, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 256 channels from L and 12 channels from pre-fire
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 12, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, pre_fire, post_fire):
        combined_input = torch.cat((pre_fire, post_fire), dim=1)
        latent = self.encoder(combined_input)
        # print(latent.size())
        # print(pre_fire.size())

        # Ensure pre_fire and latent have matching spatial dimensions
        pre_fire_resized = nn.functional.interpolate(pre_fire, size=latent.shape[2:], mode='bilinear', align_corners=False)
        #print(f"Latent shape: {latent.shape}, Pre-fire resized shape: {pre_fire_resized.shape}")
        latent_combined = torch.cat((latent, pre_fire_resized), dim=1)  # Combine latent features with pre-fire image
        reconstructed = self.decoder(latent_combined)
        return latent, reconstructed