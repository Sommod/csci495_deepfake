import torch
import torch.nn as nn
from torchvision import transforms
from timm import create_model
import torch.nn.functional as F
from GenContVit.model_embedder import HybridEmbed
from GenContVit.Vae import VariationalAutoEncoder


class GenConViTVAE(nn.Module):
    def __init__(self, img_size, num_classes, vae, pretrained=True):
        super(GenConViTVAE, self).__init__()

        self.vae = vae.encoder

        # backbones
        self.embedder = create_model('swinv2_tiny_window8_256', pretrained=pretrained)
        self.convnext_backbone = create_model('convnext_tiny', pretrained=pretrained, num_classes=1024, drop_path_rate=0, head_init_scale=1.0)
        self.convnext_backbone.patch_embed = HybridEmbed(self.embedder, img_size, embed_dim=768)
        self.num_feature = self.convnext_backbone.head.fc.out_features * 2

        # classifiers
        self.fc = nn.Linear(self.num_feature, self.num_feature//4)
        self.fc3 = nn.Linear(self.num_feature//2, self.num_feature//4)
        self.fc2 = nn.Linear(self.num_feature//4, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        z, _ = self.vae(x)

        #backbone
        x1 = self.convnext_backbone(x)

        x = torch.cat((x1, z), dim=1)

        # classifier
        x = self.fc2(self.relu(self.fc(self.relu(x))))
        
        return x
