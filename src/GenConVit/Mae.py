import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from einops import rearrange
from torchvision.transforms import transforms


class MaskedAutoEncoderViT(nn.Module):
    def __init__(self, img_size, patch_size=(16, 16), mask_ratio=0.75, pretrained=True):
        super(MaskedAutoEncoderViT, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        self.encoder = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=0)

        self.patch_embed = self.encoder.patch_embed
        self.num_patches = self.patch_embed.num_patches
        self.embed_dim = self.encoder.embed_dim

        self.encoder.patch_embed = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.decoder = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dim * 2, self.patch_size[0] * self.patch_size[1] * 3)
        )

        self.trainingloss = nn.MSELoss(reduction='none')
        self.valloss = nn.MSELoss(reduction='sum')

        self.useL1 = False
        self.traininglossl1 = nn.L1Loss(reduction='none')

    def forward(self, x, binary_mask=None):
        batch_size = x.shape[0]

        # convert image into patches
        x_resized = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x_patches = self.patch_embed(x_resized)

        if self.training and binary_mask is not None:
            visible_index, masked_index = self.apply_patch_mask(x_patches, binary_mask)

            # Create list of visible patch embeddings per sample
            x_visible = [x_patches[i, visible_index[i]] for i in range(batch_size)]
            x_visible = torch.stack([
                F.pad(patch, (0, 0, 0, self.num_patches - patch.shape[0]))
                for patch in x_visible
            ])

            # Encode only visible patches
            x_visible = self.encoder.forward_features(x_visible)
            x_visible = x_visible[:, 1:, :]

            # Prepare decoder input
            decoder_tokens = x_patches.new_zeros(batch_size, self.num_patches, self.embed_dim)
            for i in range(batch_size):
                decoder_tokens[i, masked_index[i]] = self.mask_token
                decoder_tokens[i, visible_index[i]] = x_visible[i, :len(visible_index[i])]
        else:
            decoder_tokens = self.encoder.forward_features(x_patches)
            decoder_tokens = decoder_tokens[:, 1:, :]

        decoded_patches = self.decoder(decoder_tokens)

        # Reconstruct full image
        batch_size, N, _ = decoded_patches.shape
        patch_h, patch_w = self.patch_size
        H = W = int(N ** 0.5)
        img = rearrange(decoded_patches, 'b (h w) (c ph pw) -> b c (h ph) (w pw)',
                        h=H, w=W, c=3, ph=patch_h, pw=patch_w)
        img = F.interpolate(img, size=(self.img_size , self.img_size), mode='bilinear', align_corners=False)

        # compute loss
        reconstruction_loss = self.compute_reconstruction_loss(img, x, masked_index if self.training else None)

        return img, reconstruction_loss

    def apply_patch_mask(self, patch_embeddings, binary_mask):
        B, N, D = patch_embeddings.shape
        device = patch_embeddings.device

        if binary_mask is not None:
            # Resize binary mask to match patch embeddings
            h = w = int(N ** 0.5)
            binary_mask_resized = F.interpolate(binary_mask.unsqueeze(1).float(),
                                                size=(h, w),
                                                mode='nearest').squeeze(1)
            binary_mask_resized = binary_mask_resized.view(B, -1)  # [B, N]
        else:
            binary_mask_resized = torch.ones(B, N, device=patch_embeddings.device)

        # Determine masked positions
        visible_index = []
        masked_index = []

        for i in range(B):
            patchable = binary_mask_resized [i].nonzero(as_tuple=True)[0]
            perm = torch.randperm(patchable.numel(), device=device)
            num_mask = int(self.mask_ratio * patchable.numel())
            masked = patchable[perm[:num_mask]]
            visible = patchable[perm[num_mask:]]
            visible_index.append(visible)
            masked_index.append(masked)

        return visible_index, masked_index

    def compute_reconstruction_loss(self, decimg, x, masked_index=None):
        if not self.training:
            return self.valloss(decimg, x) / x.size(0)

        # Choose loss function based on flag
        loss_fn = self.traininglossl1 if self.useL1 else self.trainingloss
        recon_loss = loss_fn(decimg, x)

        if masked_index is None:
            return recon_loss.mean()

        # Create mask for selected patches
        B, _, H, W = x.shape
        num_patches_H = H // self.patch_size[0]
        num_patches_W = W // self.patch_size[1]
        patch_H = H // (self.img_size // self.patch_size[0])
        patch_W = W // (self.img_size // self.patch_size[1])
        mask = torch.zeros(B, 1, H, W, device=x.device)
        for i, indices in enumerate(masked_index):
            for idx in indices:
                row = (idx // num_patches_H) * patch_H
                col = (idx % num_patches_W) * patch_W
                mask[i, :, row:row + patch_H, col:col + patch_W] = 1

        mask = mask.expand_as(x)
        return (recon_loss * mask).sum() / mask.sum()
