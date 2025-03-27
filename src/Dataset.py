import os
import cv2
from torch.utils.data import Dataset
import pandas as pd
import torch
from torchvision.transforms import transforms

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # GaussianNoise(std=.005)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CustomImageDataset(Dataset):
    def __init__(self, csv_file: pd.DataFrame, img_dir: str, num_classes: int, transform=None):
        self.csv = csv_file
        self.img_dir = img_dir
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        row = self.csv.iloc[idx]
        img_name = row['path']
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)
        if self.transform:
            image = self.transform(image)
        class_label = row["label"]
        class_label = torch.tensor(class_label, dtype=torch.long)
        return image, class_label