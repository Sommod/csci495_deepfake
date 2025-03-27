import torch
import torch.nn as nn
from codeFor490.GenContVit.genconvit_ed import GenConViTED
from codeFor490.GenContVit.genconvit_vae import GenConViTVAE

class GenConViT(nn.Module):
    def __init__(self, img_size, num_classes):
        super(GenConViT, self).__init__()
        self.model_ed = GenConViTED(img_size, num_classes)
        self.model_vae = GenConViTVAE(img_size, num_classes)
    def forward(self, x):
        x1 = self.model_ed(x)
        x2,_ = self.model_vae(x)
        x = (x1+x2)/2
        return x
