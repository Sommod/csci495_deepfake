import torch.nn as nn
from timm import create_model
from GenContVit.model_embedder import HybridEmbed


class GenConViTMae(nn.Module):
    def __init__(self, num_classes, mae):
        super(GenConViTMae, self).__init__()
        self.mae = mae
        self.mae.eval()

        self.embedder = create_model('swinv2_tiny_window8_256', pretrained=True)
        self.convnext_backbone = create_model('convnext_tiny', pretrained=True, num_classes=1000, drop_path_rate=0, head_init_scale=1.0)
        self.convnext_backbone.patch_embed = HybridEmbed(self.embedder, mae.img_size, embed_dim=768)

        self.num_features = self.convnext_backbone.head.fc.out_features
        self.fc = nn.Linear(self.num_features, self.num_features // 4)
        self.fc2 = nn.Linear(self.num_features // 4, num_classes)
        self.relu = nn.GELU()

    def train(self, mode=True):
        # Keep mae in eval mode regardless of the mode of GenConViTMae
        super(GenConViTMae, self).train(mode)
        self.mae.eval()


    def forward(self, x):
        img, _ = self.mae(x)
        #backbone
        x = self.convnext_backbone(img)
        # Final classification layer
        x = self.fc2(self.relu(self.fc(self.relu(x))))
        return x
