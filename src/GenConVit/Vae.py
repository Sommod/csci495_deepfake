import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, latent_dims=4):
        super(Encoder, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=8),
            nn.LeakyReLU(),

            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
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
            nn.LeakyReLU(),
        )

        self.latent_dims = latent_dims

        self.mu = nn.Linear(128 * 8 * 8, self.latent_dims)
        self.var = nn.Linear(128 * 8 * 8, self.latent_dims)

        # kl loss
        self.kl = 0

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

        self.kl = torch.mean(-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim=1), dim=0)

        return z, self.kl


class Decoder(nn.Module):
    def __init__(self, latent_dims=4, output_channels=3, feature_map_size=16):
        super(Decoder, self).__init__()

        self.latent_dims = latent_dims
        self.feature_map_size = feature_map_size  # this is the height/width of the feature map after upsampling
        self.output_channels = output_channels

        # Fully connected layer to map latent to initial feature map size
        self.fc = nn.Linear(latent_dims, 64 * feature_map_size * feature_map_size)

        # Transposed Convolutional layers
        self.features = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(8, 3, kernel_size=2, stride=2),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        # x is the latent vector, pass it through the fully connected layer
        x = self.fc(x)

        # Reshape it into the required feature map size for the convolution layers
        x = x.view(-1, 64, self.feature_map_size, self.feature_map_size)

        # Pass it through the transposed convolutional layers
        x = self.features(x)
        return x

class VariationalAutoEncoder(nn.Module):
    def __init__(self, latent_dims = 256):
        super(VariationalAutoEncoder, self).__init__()
        self.latent_dims = latent_dims

        self.encoder = Encoder(self.latent_dims)
        self.decoder = Decoder(self.latent_dims)

        # loss function to calculate how different the reconstruction is from the original image
        self.loss = nn.MSELoss()

    def forward(self, x):
        z, kl_loss = self.encoder(x)
        x_hat = self.decoder(z)
        #extracted feature +  loss
        reconstruction_loss = self.loss(x_hat, x)
        return x_hat, reconstruction_loss, kl_loss
