import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, latent_dims=4):
        super(Encoder, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            #nn.Dropout2d(0.1),

            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            #nn.Dropout2d(0.1),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            #nn.Dropout2d(0.1),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            #nn.Dropout2d(0.1),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            #nn.Dropout2d(0.1),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            #nn.Dropout2d(0.1),

        )

        self.latent_dims = latent_dims
        self.mu = nn.Linear(256 * 4 * 4, latent_dims)
        self.var = nn.Linear(256 * 4 * 4, latent_dims)
        self.kl = 0

    def reparameterize(self, x):
        mu = self.mu(x)
        logvar = self.var(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu
        return z, mu, logvar

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)

        z, mu, logvar = self.reparameterize(x)
        self.kl = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0)
        return z, self.kl

class Decoder(nn.Module):
    def __init__(self, latent_dims=4, output_channels=3, feature_map_size=4):
        super(Decoder, self).__init__()

        self.latent_dims = latent_dims
        self.feature_map_size = feature_map_size  # this is the height/width of the feature map after upsampling
        self.output_channels = output_channels

        # Fully connected layer to map latent to initial feature map size
        self.fc = nn.Linear(latent_dims, 256 * feature_map_size * feature_map_size)

        # Transposed Convolutional layers
        self.features = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.LeakyReLU(),
            #nn.Dropout2d(0.01),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.LeakyReLU(),
	        #nn.Dropout2d(0.01),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.LeakyReLU(),
	        #nn.Dropout2d(0.01),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.LeakyReLU(),
	        #nn.Dropout2d(0.01),
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2),
            nn.LeakyReLU(),
            #nn.Dropout2d(0.01),
            nn.ConvTranspose2d(8, 3, kernel_size=2, stride=2),
            nn.LeakyReLU()

        )

    def forward(self, x):
        # x is the latent vector, pass it through the fully connected layer
        x = self.fc(x)

        # Reshape it into the required feature map size for the convolution layers
        x = x.view(-1, 256, self.feature_map_size, self.feature_map_size)

        # Pass it through the transposed convolutional layers
        x = self.features(x)
        return x

class VariationalAutoEncoder(nn.Module):
    def __init__(self, latent_dims = 1024):
        super(VariationalAutoEncoder, self).__init__()
        self.latent_dims = latent_dims

        self.encoder = Encoder(self.latent_dims)
        self.decoder = Decoder(self.latent_dims)

        # loss function to calculate how different the reconstruction is from the original image
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, x):
        z, kl_loss = self.encoder(x)
        x_hat = self.decoder(z)
        #extracted feature +  loss
        reconstruction_loss = self.loss(x_hat, x) /  x.size(0)
        return x_hat, reconstruction_loss, kl_loss
