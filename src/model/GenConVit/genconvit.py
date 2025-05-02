import torch
import torch.nn as nn
from GenContVit.GenConVitMae import GenConViTMae
from GenContVit.genconvit_vae import GenConViTVAE

class GenConViT(nn.Module):
    def __init__(self, img_size, num_classes, mae, vae):
        super(GenConViT, self).__init__()
        self.model_ed = GenConViTMae(num_classes, mae)
        self.model_vae = GenConViTVAE(img_size, num_classes, vae)
    def forward(self, x):
        x2 = self.model_vae(x)
        x1 = self.model_ed(x)
        return x1, x2
