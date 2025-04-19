'''
    Training the VAE for the model
'''

# Imports
import warnings
warnings.filterwarnings('ignore')

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from GenConVit.genconvit_vae import GenConViTVAE
from Dataset import VaeDataset, train_transform, test_transform
from MaskingProcess import worker_init_fn
from EarlyStopping import EarlyStopping

def train_vae(training, validating, device, directory):

    vaeTraining = vaeTraining[ : len(vaeTraining) // 1000]
    vaeValidating = vaeValidating[ : len(vaeValidating) // 1000]

    model_vae = GenConViTVAE(256, 2)

    train_set = VaeDataset(vaeTraining, directory, transform = train_transform)
    val_set = VaeDataset(vaeValidating, directory, transform = test_transform)
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    train_val_vae(model_vae, device, train_loader, val_loader)
    return model_vae

def train_val_vae(model, device, train_loader, val_loader, epochs = 10):
    early_stopping = EarlyStopping(patience=5, verbose=True, path ='VaeCheckPoint.pth')
    optimizer = optim.Adam(model.parameters(), lr = 0.0001, weight_decay = 0.0001)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, mask, in train_loader:
            inputs, mask = inputs.to(device), mask.to(device)
            optimizer.zero_grad

            _, reconstruction_loss = model(inputs, mask)
            reconstruction_loss.backward()
            optimizer.step()
            running_loss += reconstruction_loss.item()

        train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, mask in val_loader:
                inputs, mask = inputs.to(device), mask.to(device)
                _, reconstruction_loss = model(inputs, mask)
                val_loss += reconstruction_loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            early_stopping.load_best_model(model)
            print("Early stopping triggered.")
            break
    print("Training complete!")