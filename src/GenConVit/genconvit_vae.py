import torch
import torch.nn as nn
from torchvision import transforms
from timm import create_model
import torch.nn.functional as F
from codeFor490.GenContVit.model_embedder import HybridEmbed


class Encoder(nn.Module):
    def __init__(self, latent_dims=4):
        super(Encoder, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU()
        )

        self.latent_dims = latent_dims
        # After 4 convolution layers with stride 2, the image size becomes 256 -> 128 -> 64 -> 32 -> 16
        self.fc1 = nn.Linear(128 * 16 * 16, 256)  # Adjusted for 256x256 input size
        self.fc2 = nn.Linear(256, 128)
        self.mu = nn.Linear(128 * 16 * 16, self.latent_dims)  # Adjusted for 256x256 input size
        self.var = nn.Linear(128 * 16 * 16, self.latent_dims)  # Adjusted for 256x256 input size

        self.kl = 0
        self.kl_weight = 0.5  # Adjust as needed
        self.relu = nn.LeakyReLU()

    def reparameterize(self, x):
        std = torch.exp(0.5 * self.mu(x))
        eps = torch.randn_like(std)
        z = eps * std + self.mu(x)
        return z, std

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)

        mu = self.mu(x)
        var = self.var(x)
        z, _ = self.reparameterize(x)
        self.kl = self.kl_weight * torch.mean(-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim=1), dim=0)

        return z

class Decoder(nn.Module):
    def __init__(self, latent_dims=4):
        super(Decoder, self).__init__()
        self.features = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2),
            nn.LeakyReLU()
        )

        self.latent_dims = latent_dims
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(256, 8, 8))

    def forward(self, x):
        x = self.unflatten(x)
        x = self.features(x)
        return x

class GenConViTVAE(nn.Module):
    def __init__(self, img_size, num_classes, pretrained=True):
        super(GenConViTVAE, self).__init__()
        self.latent_dims = 16384
        self.encoder = Encoder(self.latent_dims)
        self.decoder = Decoder(self.latent_dims)
        self.embedder = create_model('swinv2_tiny_window8_256', pretrained=pretrained)
        self.convnext_backbone = create_model('convnext_tiny', pretrained=pretrained, num_classes=1000, drop_path_rate=0, head_init_scale=1.0)
        self.convnext_backbone.patch_embed = HybridEmbed(self.embedder, img_size, embed_dim=768)
        self.num_feature = self.convnext_backbone.head.fc.out_features * 2
 
        self.fc = nn.Linear(self.num_feature, self.num_feature//4)
        self.fc3 = nn.Linear(self.num_feature//2, self.num_feature//4)
        self.fc2 = nn.Linear(self.num_feature//4, num_classes)
        self.relu = nn.ReLU()
        self.resize = transforms.Resize((256,256), antialias=True)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        x1 = self.convnext_backbone(x)
        x2 = self.convnext_backbone(x_hat)

        # Calculate the reconstruction loss (Mean Squared Error)
        reconstruction_loss = F.mse_loss(self.resize(x_hat), x)  # MSE loss between input and reconstructed image
        #kl_divergence = self.encoder.kl
        #total_loss = reconstruction_loss + kl_divergence

        x = torch.cat((x1,x2), dim=1)
        x = self.fc2(self.relu(self.fc(self.relu(x))))
        
        return x, reconstruction_loss