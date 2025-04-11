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

        # Vision Transformer Encoder
        self.encoder = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=0)
        self.patch_embed = self.encoder.patch_embed

        # Mask Token for Transformer decoder
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.encoder.num_features))

        # Decoder (Transformer Decoder Layer)
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=self.encoder.num_features, nhead=8, dim_feedforward=2048
            ),
            num_layers=6
        )

        # Final layer to map decoder output back to image
        self.final_layer = nn.Linear(self.encoder.num_features, img_size * img_size * 3)

        # Reconstruction loss
        self.loss = nn.MSELoss(reduction='none')

    def forward(self, x, binary_mask=None):
        device = x.device  # Get the device from input tensor

        # Resize input image to 224x224
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        # Apply masking and patch embedding
        if self.training:
            x_masked, patch_mask, ids_restore = self.apply_patch_mask(x, binary_mask)
        else:
            x_masked = x
            patch_mask = torch.ones_like(x).to(device)

        encoded = self.encoder.forward_features(x_masked)
        # Generate Masked Tokens to fill the missing parts (like MAE)
        B, N, D = x_masked.shape
        mask_tokens = self.mask_token.expand(B, N - encoded.shape[1], D)  # [B, N-masked, D]
        full_sequence = torch.zeros(B, N, D, device=device)
        full_sequence.scatter_(1, ids_restore.unsqueeze(-1).expand(-1, -1, D), encoded)

        # Pass through Decoder
        decoder_output = self.decoder(full_sequence, encoded)  # Decoder with embedded tokens

        # Final mapping to image space
        recon_img = self.final_layer(decoder_output)  # [B, N, patch_dim]
        recon_img = recon_img.view(B, 3, self.img_size, self.img_size)  # Reconstruct image

        # Compute Reconstruction Loss
        recon_loss = self.compute_reconstruction_loss(recon_img, x, patch_mask)
        return recon_img, recon_loss

    def apply_patch_mask(self, image, binary_mask):
        """
        Apply the masking process to patches from the image.
        image: [B, 3, H, W]
        binary_mask: [B, H, W] — 1 = allowed to be masked, 0 = not maskable
        """
        device = image.device
        B, _, H, W = image.shape
        patch_H, patch_W = self.patch_size
        N_h, N_w = H // patch_H, W // patch_W
        N = N_h * N_w

        # 1. Patchify the image using ViT's patch_embed
        x = self.patch_embed(image)  # [B, N, D]

        # 2. Resize binary mask to patch level
        binary_mask_resized = F.interpolate(
            binary_mask.unsqueeze(1).float(), size=(N_h, N_w), mode='nearest'
        ).squeeze(1)  # [B, N_h, N_w]
        binary_mask_flat = binary_mask_resized.view(B, -1)  # [B, N]

        # 3. Determine which positions are allowed to be masked (where mask == 1)
        masked_positions = []
        for i in range(B):
            allowed = torch.nonzero(binary_mask_flat[i] == 1, as_tuple=True)[0]
            num_to_mask = int(self.mask_ratio * allowed.size(0))
            rand_indices = torch.randperm(allowed.size(0), device=device)[:num_to_mask]
            masked_positions.append(allowed[rand_indices])

        # 4. Create a full patch-level mask (1 = keep, 0 = masked)
        patch_mask = torch.ones((B, N), device=device)
        for i in range(B):
            patch_mask[i, masked_positions[i]] = 0  # 0 where masked

        # 5. Apply mask (zero out masked patch embeddings)
        x_masked = x.clone()
        x_masked[patch_mask == 0] = 0  # Masking the patches
        ids_restore = torch.argsort(patch_mask, dim=1)  # Restore order of patches
        return x_masked, patch_mask, ids_restore

    def compute_reconstruction_loss(self, decimg, x, mask):
        """
        Compute the reconstruction loss by only considering the masked positions
        """
        recon_loss = self.loss(decimg, x)  # Pixel-wise loss
        recon_loss = (recon_loss * mask).sum() / mask.sum()  # Masked loss only
        return recon_loss