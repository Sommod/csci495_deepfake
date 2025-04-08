import os
import cv2
from torch.utils.data import Dataset
import pandas as pd
import torch
from torchvision.transforms import transforms
from MaskingProcess import load_mask_processor
from MaskingProcess import segmenter

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# for the main model
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

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        class_label = row["label"]
        class_label = torch.tensor(class_label, dtype=torch.long)
        return image, class_label

# for the masked autoencoder model
class MaeDataset(Dataset):
    def __init__(self, csv_file: pd.DataFrame, img_dir: str, transform=None):
        self.csv = csv_file
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.csv)
    def __getitem__(self, idx):
        row = self.csv.iloc[idx]
        img_name = row['path']
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)
        # Apply transformations
        if self.transform:
            image = self.transform(image)
            image_np = image.permute(1, 2, 0).numpy()  # Convert to HWC format for Mediapipe
            binary_mask = load_mask_processor().process(image_np)
            binary_mask = torch.from_numpy(binary_mask).float()

        return image, binary_mask
