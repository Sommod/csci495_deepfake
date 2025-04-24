import torch.nn as nn
from timm import create_model
from GenContVit.model_embedder import HybridEmbed
import torch
import torch.nn.functional as F


class GenConViTMae(nn.Module):
    def __init__(self, num_classes, mae):
        super(GenConViTMae, self).__init__()
        self.mae = mae.encoder
        self.patch_embed = mae.patch_embed
        self.spatial_fusion = SpatialFusion(768, 768, 768)
        self.embedder = create_model('swinv2_tiny_window8_256', pretrained=True)
        self.convnext_backbone = create_model('convnext_tiny', pretrained=True, num_classes=256, drop_path_rate=0, head_init_scale=1.0)
        self.convnext_backbone.patch_embed = HybridEmbed(self.embedder, mae.img_size, embed_dim=768)

        self.num_features = self.convnext_backbone.head.fc.out_features
        self.fc = nn.Linear(768 * 2, 768)
        self.fc2 = nn.Linear(768, num_classes)
        self.relu = nn.GELU()

    def forward(self, x):
        # extract tokens from mae
        x_resized = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x_patches = self.patch_embed(x_resized)
        mae_tokens = self.mae.forward_features(x_patches)
        mae_tokens = mae_tokens[:, 1:, :]

        #backbone
        conv_feat = self.convnext_backbone.forward_features(x)  # [B, C, H, W]

        # fuse with the tokens
        spatial_fused = self.spatial_fusion(mae_tokens, conv_feat)
        fused_pooled = torch.mean(spatial_fused, dim=[2, 3])  # [B, F]
        global_mae = mae_tokens.mean(dim=1)
        x = torch.cat([fused_pooled, global_mae], dim=1)

        # Final classification layer
        x = self.fc2(self.relu(self.fc(self.relu(x))))
        return x

class SpatialFusion(nn.Module):
    def __init__(self, mae_dim, conv_dim, out_dim):
        super().__init__()
        self.mae_proj = nn.Conv2d(mae_dim, out_dim, kernel_size=1)
        self.conv_proj = nn.Conv2d(conv_dim, out_dim, kernel_size=1)

        # attention mechanism
        self.attn = nn.Sequential(
            nn.Conv2d(2 * out_dim, out_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_dim, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )

        self.output_proj = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )

    def resize_mae_tokens_to_conv(self, mae_spatial, conv_feat):
        B, N, D = mae_spatial.shape
        H_mae = W_mae = int(N ** 0.5)
        H_conv, W_conv = conv_feat.shape[2], conv_feat.shape[3]
        mae_spatial = mae_spatial.permute(0, 2, 1).reshape(B, D, H_mae, W_mae)  # [B, D, 14, 14]
        mae_resized = F.interpolate(mae_spatial, size=(H_conv, W_conv), mode='bilinear', align_corners=False)  # [B, D, Hc, Wc]
        return mae_resized

    def forward(self, mae_spatial, conv_feat):
        B, _, H, W = conv_feat.shape

        # resize mae to be conv_feat
        mae_spatial = self.resize_mae_tokens_to_conv(mae_spatial, conv_feat)

        # Project both streams to common space
        mae_proj = self.mae_proj(mae_spatial)
        conv_proj = self.conv_proj(conv_feat)

        # create attentions
        joint = torch.cat([mae_proj, conv_proj], dim=1)
        weights = self.attn(joint)  # [B, 2, H, W]

        # split attentions
        mae_weight, conv_weight = weights[:, 0:1], weights[:, 1:2]

        # fuse the projections based on weights
        fused = mae_weight * mae_proj + conv_weight * conv_proj

        return self.output_proj(fused)
