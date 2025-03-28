import torch
import torch.nn as nn
import timm
from timm import create_model
from torch.nn.functional import mse_loss
import torch.nn.functional as F

from codeFor490.GenContVit.model_embedder import HybridEmbed


class MaskedAutoEncoderViT(nn.Module):
    def __init__(self, img_size, patch_size=(16, 16), mask_ratio=0.25, pretrained=True):
        super(MaskedAutoEncoderViT, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        self.encoder = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=0)

        # Decoder that reconstructs the image
        self.decoder = nn.Sequential(
            nn.Linear(self.encoder.num_features, 768),  # Latent dim to match encoder output size
            nn.ReLU(),
            nn.Linear(768, img_size * img_size * 3),  # Reconstruction to match image size
            nn.Sigmoid()  # To ensure the output is within [0, 1]
        )

        self.embedder = create_model('swinv2_tiny_window8_256', pretrained=pretrained)
        self.convnext_backbone = create_model('convnext_tiny', pretrained=pretrained, num_classes=1000, drop_path_rate=0, head_init_scale=1.0)
        self.convnext_backbone.patch_embed = HybridEmbed(self.embedder, img_size, embed_dim=768)

        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, x, binary_mask = None):
        # Mask the input during training
        if self.training:
            masked_images, mask = self.apply_patch_mask(x, binary_mask)
        else:
            masked_images = x
            mask = torch.ones_like(x)

        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        # Flatten the masked image into patches and pass through the encoder
        encoder_img = self.encoder(masked_images)  # Encoder processes the masked patches
        # Decoder reconstructs the image
        decoder_img = self.decoder(encoder_img)  # Reconstructed image
        img = decoder_img.view(4, 3, 256, 256)
        x1 = self.convnext_backbone(x)
        x2 = self.convnext_backbone(img)

        reconstruction_loss = self.compute_reconstruction_loss(img, x, mask)

        x = torch.cat((x1,x2), dim=1)

        return x, reconstruction_loss

    def apply_patch_mask(self, image, binary_mask):
        """
        Apply random patches to an image within the boundary defined by binary_mask.
        """
        # Get image dimensions
        batch_size, _, height, width = image.shape
        patch_height, patch_width = self.patch_size

        # Create a copy of the image to apply patches
        output_image = image.clone()
        patch_mask = torch.zeros_like(binary_mask, dtype=torch.float32)

        # Iterate through each image in the batch
        for i in range(batch_size):
            mask = binary_mask[i]
            patch_positions = (mask == 1)
            indices = torch.nonzero(patch_positions)
            indices = indices[torch.randperm(indices.size(0))]
            num_patches = int(self.mask_ratio * (indices.shape[0] // 500))
            indices = indices[:num_patches]

            for idx in indices:
                y, x = idx[0], idx[1]
                # Create a random patch from the image or use a fixed patch
                output_image[i, :, y:y + patch_height, x:x + patch_width] = 0

                patch_mask[i, y:y + patch_height, x:x + patch_width] = 0
        return output_image, patch_mask


    def compute_reconstruction_loss(self, decimg, x, mask):
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        # Compute pixel-wise MSE loss for reconstruction
        recon_loss = self.loss_fn(decimg, x)
        recon_loss = (recon_loss * mask).sum() / mask.sum()  # Normalize by unmasked pixels
        return recon_loss