import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import pandas as pd
import torch
from torchvision.transforms import transforms
import mediapipe as mp

class MediapipeSegmenter:
    _instance = None  # Singleton instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MediapipeSegmenter, cls).__new__(cls)
            cls._instance.segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        return cls._instance

    def process(self, image):
        # Read the image
        height, width, _ = image.shape
        image = (image * 255).astype(np.uint8)
        # Convert the image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Perform segmentation
        result = self._instance.segmenter.process(image_rgb)
        # Extract the segmentation mask (background vs person)
        mask = result.segmentation_mask
        # Threshold the mask to create a binary mask for the person
        _, binary_mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)
        return binary_mask

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5)
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

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        class_label = row["label"]
        class_label = torch.tensor(class_label, dtype=torch.long)
        return image, class_label

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
            binary_mask = MediapipeSegmenter().process(image_np)
            binary_mask = torch.from_numpy(binary_mask).float()

        return image, binary_mask