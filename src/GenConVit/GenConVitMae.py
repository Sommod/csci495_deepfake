import torch
import torch.nn as nn
import random
import timm
from codeFor490.GenContVit.model_embedder import HybridEmbed

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),

            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),

            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)
        )

    def forward(self, x):
        return self.features(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, 3, kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.features(x)


class GenConViTMae(nn.Module):
    def __init__(self, img_size, num_classes, pretrained=True, mask_ratio=0.25):
        super(GenConViTMae, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.backbone = timm.create_model('convnext_tiny', pretrained=pretrained)
        self.embedder = timm.create_model('swinv2_tiny_window8_256', pretrained=pretrained)
        self.backbone.patch_embed = HybridEmbed(self.embedder, img_size=img_size, embed_dim=768)
        self.num_features = self.backbone.head.fc.out_features * 2
        self.fc = nn.Linear(self.num_features, self.num_features // 4)
        self.fc2 = nn.Linear(self.num_features // 4, num_classes)
        self.relu = nn.GELU()
        self.mask_ratio = mask_ratio
        self.loss_fn = nn.MSELoss(reduction='none')  # Use MSE loss for reconstruction

    def forward(self, x):
        if self.training:
            # Apply mask during training
            masked_images, mask = self.apply_mask(x)
        else:
            # No masking during inference
            masked_images = x
            mask = torch.ones_like(x)

        encimg = self.encoder(masked_images)  # Pass the masked images through the encoder
        decimg = self.decoder(encimg)  # Decoder tries to reconstruct the image

        reconstruction_loss = self.compute_reconstruction_loss(decimg, x, mask)

        x1 = self.backbone(decimg)  # Features from reconstructed image
        x2 = self.backbone(x)  # Features from original image
        x = torch.cat((x1, x2), dim=1)
        x = self.fc2(self.relu(self.fc(self.relu(x))))  # Final classification layer
        return x, reconstruction_loss

    def apply_mask(self, images):
        """
        Applies a random mask to the input images.
        Masking a percentage of the image pixels by setting them to 0.
        """
        batch_size, _, height, width = images.size()
        mask = torch.ones_like(images)

        # Generate random masks
        mask_area = int(self.mask_ratio * height * width)
        for i in range(batch_size):
            mask_indices = random.sample(range(height * width), mask_area)
            for index in mask_indices:
                h = index // width
                w = index % width
                mask[i, :, h, w] = 0  # Set masked regions to 0
        return images * mask, mask  # Apply the mask to the input images

    def compute_reconstruction_loss(self, decimg, images, mask):
        """
        Compute the reconstruction loss (MSE) on the unmasked regions.
        """
        # Compute MSE loss, but only for unmasked regions
        loss = self.loss_fn(decimg, images)  # Apply MSE loss
        loss = loss * mask  # Mask the loss (set loss to 0 for masked pixels)
        return loss.sum() / mask.sum()  # Normalize by the number of non-masked pixels