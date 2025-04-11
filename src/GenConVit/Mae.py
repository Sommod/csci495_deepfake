import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class MaskedAutoEncoderViT(nn.Module):
    def __init__(self, img_size, patch_size=(16, 16), mask_ratio=0.75, pretrained=True):
        super(MaskedAutoEncoderViT, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        self.encoder = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=0)

        self.decoder = nn.Sequential(
            nn.Linear(self.encoder.num_features, 768),
            nn.ReLU(),
            nn.Linear(768, img_size * img_size * 3),
        )

        self.loss = nn.MSELoss()

    def forward(self, x, binary_mask=None):
        device = x.device  # Get the device from input tensor

        if self.training:
            masked_images, mask = self.apply_patch_mask(x, binary_mask)
        else:
            masked_images = x
            mask = torch.ones_like(x).to(device)

        masked_images = F.interpolate(masked_images, size=(224, 224), mode='bilinear', align_corners=False)

        encoder_img = self.encoder(masked_images)
        decoder_img = self.decoder(encoder_img)

        img = decoder_img.view(x.shape[0], 3, self.img_size, self.img_size)

        reconstruction_loss = self.compute_reconstruction_loss(img, x, mask)
        return img, reconstruction_loss

    def apply_patch_mask(self, image, binary_mask):
        device = image.device
        batch_size, channels, height, width = image.shape
        patch_height, patch_width = self.patch_size
        num_patches_h = height // patch_height
        num_patches_w = width // patch_width

        patch_image = self.image_patching(image)

        binary_mask_resized = F.interpolate(binary_mask.unsqueeze(1).float(),
                                            size=(num_patches_h, num_patches_w),
                                            mode='nearest').squeeze(1)
        binary_mask_resized = binary_mask_resized.view(batch_size, -1)

        masked_positions = []
        for i in range(batch_size):
            available_indexes = torch.nonzero(binary_mask_resized[i] == 1, as_tuple=True)[0]
            available_position = available_indexes.size(0)
            num_patches = int(self.mask_ratio * available_position)
            random_available_indices = torch.randperm(available_position, device=device)[:num_patches]
            masked_positions.append(available_indexes[random_available_indices])

        mask = torch.ones_like(patch_image).to(device)

        for i in range(batch_size):
            for pos in masked_positions[i]:
                grid_idx = pos.item()
                patch_image[i, grid_idx] = 0
                mask[i, grid_idx] = 0

        reconstructed_image = self.image_unpatching(patch_image, (height, width)).to(device)
        reconstructed_mask = self.image_unpatching(mask, (height, width)).to(device)
        return reconstructed_image, reconstructed_mask

    def compute_reconstruction_loss(self, decimg, x, mask):
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        recon_loss = self.loss(decimg, x)
        recon_loss = (recon_loss * mask).sum() / mask.sum()
        return recon_loss

    def image_patching(self, image):
        batch_size, channels, height, width = image.shape
        patch_height, patch_width = self.patch_size
        patches = image.unfold(2, patch_height, patch_height).unfold(3, patch_width, patch_width)
        patches = patches.contiguous().view(batch_size, channels, -1, patch_height, patch_width)
        patches = patches.permute(0, 2, 1, 3, 4)
        return patches

    def image_unpatching(self, patches, image_size):
        batch_size, num_patches, channels, patch_height, patch_width = patches.shape
        height, width = image_size
        device = patches.device

        reconstructed_image = torch.zeros(batch_size, channels, height, width, device=device)

        index = 0
        for i in range(0, height, patch_height):
            for j in range(0, width, patch_width):
                reconstructed_image[:, :, i:i + patch_height, j:j + patch_width] = patches[:, index, :, :, :]
                index += 1
        return reconstructed_image