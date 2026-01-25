import torch
from torch import nn

class ModifiedAutoencoder(nn.Module):
    def __init__(self):
        super(ModifiedAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1),  # 24 channels for concatenated pre and post images
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # Reduced to 128 channels
            nn.ReLU()
        )

        self.pre_fire_processor = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),  # Process pre-fire image to match latent feature size
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 128 latent + 128 processed pre-fire
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 12, kernel_size=3, stride=2, padding=1, output_padding=1),  # Assuming 3-channel output (RGB)
            nn.Sigmoid()
        )

    def forward(self, pre_fire, post_fire):
        combined_input = torch.cat((pre_fire, post_fire), dim=1)
        latent = self.encoder(combined_input)
        
        pre_fire_processed = self.pre_fire_processor(pre_fire)
        pre_fire_resized = nn.functional.interpolate(pre_fire_processed, size=latent.shape[2:], mode='bilinear', align_corners=False)
        
        latent_combined = torch.cat((latent, pre_fire_resized), dim=1)  # Combine latent features with processed pre-fire image
        reconstructed = self.decoder(latent_combined)
        return latent, reconstructed
