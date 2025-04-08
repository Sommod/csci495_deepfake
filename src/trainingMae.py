import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import mediapipe as mp

from codeFor490.Dataset import train_transform
from codeFor490.Dataset import MaeDataset, test_transform
from codeFor490.EarlyStopping import EarlyStopping
from codeFor490.GenContVit.Mae import MaskedAutoEncoderViT
from codeFor490.MaskingProcess import worker_init_fn

def trainingMae(training, validating, device, directory):

    maeTraining = training[training["label"] == 1]
    maeValidating = validating[validating["label"] == 1]

    # adjust this to change the amount of images to train through
    maeTraining = maeTraining[ : len(training)//1000]
    maeValidating = maeValidating[ : len(validating)//1000]

    model_mae = MaskedAutoEncoderViT(256)

    # ensure num_workers = 0 unless you want warnings from mediapipe screaming at you
    train_set = MaeDataset(maeTraining, directory, transform=train_transform)
    val_set = MaeDataset(maeValidating, directory, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    train_validate_mae(model_mae, device, train_loader, val_loader)
    return model_mae

# Function for training and validation
def train_validate_mae(model, device, train_loader, val_loader, epochs=100):
    early_stopping = EarlyStopping(patience=5, verbose=True, path ='MaeCheckPoint.pth')
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        # Training loop
        for inputs, mask in train_loader:
            inputs, mask = inputs.to(device), mask.to(device)
            optimizer.zero_grad()
            _, reconstruction_loss = model(inputs, mask)
            reconstruction_loss.backward()
            optimizer.step()
            running_loss += reconstruction_loss.item()

        train_loss = running_loss / len(train_loader)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, mask in val_loader:
                inputs, mask = inputs.to(device), mask.to(device)
                _, reconstruction_loss = model(inputs, mask)
                val_loss += reconstruction_loss.item()

        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            early_stopping.load_best_model(model)
            print("Early stopping triggered.")
            break
    print("Training complete!")