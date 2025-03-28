import torch.nn as nn

class GenConViTMae(nn.Module):
    def __init__(self, num_classes, mae):
        super(GenConViTMae, self).__init__()
        self.mae = mae
        self.mae.eval()
        self.num_features = self.mae.convnext_backbone.head.fc.out_features * 2
        self.fc = nn.Linear(self.num_features, self.num_features // 4)
        self.fc2 = nn.Linear(self.num_features // 4, num_classes)
        self.relu = nn.GELU()

    def train(self, mode=True):
        # Keep mae in eval mode regardless of the mode of GenConViTMae
        super(GenConViTMae, self).train(mode)
        self.mae.eval()  # Ensure mae stays in eval mode


    def forward(self, x):
        x, _ = self.mae(x)
        x = self.fc2(self.relu(self.fc(self.relu(x))))  # Final classification layer
        return x