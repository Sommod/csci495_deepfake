import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import mediapipe as mp
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Util.Dataset import MaeDataset, train_transform, test_transform
from Util.EarlyStopping import EarlyStopping
from model.Mae import MaskedAutoEncoderViT
from Util.MaskingProcess import worker_init_fn

def trainingMae(training, validating, device, directory):

    maeTraining = training[training["label"] == 1]
    maeValidating = validating[validating["label"] == 1]

    maeTraining = maeTraining[:10000]
    maeValidating = maeValidating[:2000]

    model_mae = MaskedAutoEncoderViT(256)

    # ensure num_workers = 0 unless you want warnings from mediapipe screaming at you
    train_set = MaeDataset(maeTraining, directory, transform=train_transform)
    val_set = MaeDataset(maeValidating, directory, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=1, pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=True, num_workers=1, pin_memory=True, worker_init_fn=worker_init_fn)

    train_validate_mae(model_mae, device, train_loader, val_loader)
    return model_mae

# Function for training and validation
def train_validate_mae(model, device, train_loader, val_loader, epochs=1000):
    early_stopping = EarlyStopping(patience=60, verbose=True, path ='Output/MaeCheckPoint.pth', save_all = False)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total = 0

        # Training loop
        for inputs, mask in train_loader:
            inputs, mask = inputs.to(device), mask.to(device)
            optimizer.zero_grad()
            _, reconstruction_loss = model(inputs, mask)
            reconstruction_loss.backward()
            optimizer.step()
            running_loss += reconstruction_loss.item()
            total += inputs.size(0)

        train_loss = running_loss / total

        # Validation loop
        model.eval()
        val_loss = 0.0
        total = 0

        with torch.no_grad():
            for inputs, mask in val_loader:
                inputs, mask = inputs.to(device), mask.to(device)
                _, reconstruction_loss = model(inputs, mask)
                val_loss += reconstruction_loss.item()
                total += inputs.size(0)

        val_loss /= total

        if val_loss < 250.0:
            model.useL1 = True

        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        scheduler.step(val_loss)

        early_stopping(val_loss, epoch, model)
        if early_stopping.early_stop:
            early_stopping.load_best_model(model)
            print("Early stopping triggered.")
            break
    print("Training complete!")
