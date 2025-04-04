import torch
import torch.nn as nn
import timm
from timm import create_model
import torch.nn.functional as F

from codeFor490.GenContVit.model_embedder import HybridEmbed


class MaskedAutoEncoderViT(nn.Module):
    def __init__(self, img_size, patch_size=(16, 16), mask_ratio=0.75, pretrained=True):
        super(MaskedAutoEncoderViT, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        # Encoder that utilize the vision transformer
        self.encoder = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=0)

        # Decoder that reconstructs the image using linear layers
        self.decoder = nn.Sequential(
            nn.Linear(self.encoder.num_features, 768),
            nn.ReLU(),
            nn.Linear(768, img_size * img_size * 3),
        )

        # the ConvNext and Swin backbone
        self.embedder = create_model('swinv2_tiny_window8_256', pretrained=pretrained)
        self.convnext_backbone = create_model('convnext_tiny', pretrained=pretrained, num_classes=1000, drop_path_rate=0, head_init_scale=1.0)
        self.convnext_backbone.patch_embed = HybridEmbed(self.embedder, img_size, embed_dim=768)

        # loss function to calculate how different the reconstruction is from the original image
        self.loss = nn.MSELoss(reduction='none')

    def forward(self, x, binary_mask = None):

        # Mask the input during training
        if self.training:
            masked_images, mask = self.apply_patch_mask(x, binary_mask)
        else:
            masked_images = x
            mask = torch.ones_like(x)

        # resize to fit into transformer
        masked_images = F.interpolate(masked_images, size=(224, 224), mode='bilinear', align_corners=False)

        # autoencoders
        encoder_img = self.encoder(masked_images)
        decoder_img = self.decoder(encoder_img)

        #reshape
        img = decoder_img.view(4, 3, self.img_size, self.img_size)
        x1 = self.convnext_backbone(x)
        x2 = self.convnext_backbone(img)

        #loss
        reconstruction_loss = self.compute_reconstruction_loss(img, x, mask)
        x = torch.cat((x1,x2), dim=1)

        return x, reconstruction_loss

    def apply_patch_mask(self, image, binary_mask):
        # Get image dimensions
        batch_size, channels, height, width = image.shape
        patch_height, patch_width = self.patch_size

        # Calculate the patch size
        num_patches_h = height // patch_height
        num_patches_w = width // patch_width

        # change to (batch_size, num_patch, channels, patch_height, patch_width)
        patch_image = self.image_patching(image)

        # Resizing the binary mask to match the patch size
        binary_mask_resized = F.interpolate(binary_mask.unsqueeze(1).float(),
                                            size=(num_patches_h, num_patches_w),
                                            mode='nearest').squeeze(1)
        binary_mask_resized = binary_mask_resized.view(batch_size, -1)

        masked_positions = []

        # get position to mask based on binary_mask
        for i in range(batch_size):
            available_indexes = torch.nonzero(binary_mask_resized[i] == 1, as_tuple=True)[0]
            available_position = available_indexes.size(0)
            num_patches = int(self.mask_ratio * available_position)
            random_available_indices = torch.randperm(available_position)[:num_patches]
            masked_positions.append(available_indexes[random_available_indices])


        mask = torch.ones_like(patch_image)

        # mask the selected regions
        for i in range(batch_size):
            for pos in masked_positions[i]:
                grid_idx = pos.item()
                # Set the corresponding patch to zero in patch_image and mask
                patch_image[i, grid_idx] = 0
                mask[i, grid_idx] = 0

        # Unpatch back to original image shape
        reconstructed_image = self.image_unpatching(patch_image, (height, width))
        reconstructed_mask = self.image_unpatching(mask, (height, width))
        return reconstructed_image, reconstructed_mask

    def compute_reconstruction_loss(self, decimg, x, mask):
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        # calculate reconstruction loss and normalize by visible pixels
        recon_loss = self.loss(decimg, x)
        recon_loss = (recon_loss * mask).sum() / mask.sum()
        return recon_loss

    def image_patching(self, image):
        #shapes
        batch_size, channels, height, width = image.shape
        patch_height, patch_width = self.patch_size
        # Unfold the image to get extract patches
        patches = image.unfold(2, patch_height, 16).unfold(3, patch_width, 16)
        # Reshape the patches tensor to have shape (batch_size, num_patches, channels, patch_height, patch_width)
        patches = patches.contiguous().view(batch_size, channels, -1, patch_height, patch_width)
        patches = patches.permute(0, 2, 1, 3, 4)
        return patches

    def image_unpatching(self, patches, image_size):
        # retrieve shapes
        batch_size, num_patches, channels, patch_height, patch_width = patches.shape
        height, width = image_size
        patch_height, patch_width = self.patch_size
        reconstructed_image = torch.zeros(batch_size, channels, height, width)

        # Unfold the patches
        index = 0
        for i in range(0, height, patch_height):
            for j in range(0, width, patch_width):
                reconstructed_image[:, :, i:i + patch_height, j:j + patch_width] = patches[:, index, :, :, :]
                index += 1
        return reconstructed_image