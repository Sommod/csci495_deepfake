import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import mediapipe as mp

from Dataset import CustomImageDataset, test_transform, train_transform
from EarlyStopping import EarlyStopping
from GenContVit.Vae import VariationalAutoEncoder


def trainingVae(training, validating, device, directory):

    model_vae = VariationalAutoEncoder()

    # remove this when doing full testing
    training = training[:len(training)]
    validating = validating[:len(validating)]
    print(len(training))
    print(len(validating))

    # ensure num_workers = 0 unless you want warnings from mediapipe screaming at you
    train_set = CustomImageDataset(training, directory, 2, transform=train_transform)
    val_set = CustomImageDataset(validating, directory, 2, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=True, num_workers=1, pin_memory=True)

    train_validate_vae(model_vae, device, train_loader, val_loader)
    return model_vae

# Function for training and validation
def train_validate_vae(model, device, train_loader, val_loader, epochs=100):
    early_stopping = EarlyStopping(patience=10, verbose=True, path ='VaeCheckPoint.pth', save_all = False)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)
    model.to(device)

    warmup_epochs = 30

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_reconstruction_loss = 0.0
        running_kl_loss = 0.0
        total = 0

        kl_weight = min(1.0, (epoch + 1) / warmup_epochs)

        # Training loop
        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            _, reconstruction_loss, kl_loss = model(inputs)

            kl_loss *= kl_weight

            total_loss = reconstruction_loss + kl_loss
            total_loss.backward()
            optimizer.step()
            total += inputs.size(0)
            running_loss += total_loss.item()
            running_reconstruction_loss += reconstruction_loss.item()
            running_kl_loss += kl_loss.item()

            print(f"Train Loss: {running_loss:.4f} | Train Reconstruction Loss: {running_reconstruction_loss:.4f} | "
                f"Train KL Loss: {running_kl_loss:.4f}")

        train_loss = running_loss / total
        running_reconstruction_loss /= total
        running_kl_loss /= total

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_reconstruction_loss = 0.0
        val_kl_loss = 0.0
        total = 0
        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(device)
                _, reconstruction_loss, kl_loss = model(inputs)

                kl_loss *= kl_weight

                total_loss = reconstruction_loss + kl_loss

                total += inputs.size(0)
                val_loss += total_loss.item()
                val_reconstruction_loss += reconstruction_loss.item()
                val_kl_loss += kl_loss.item()

        val_loss /= total
        val_reconstruction_loss /= total
        val_kl_loss /= total

        print(f"Epoch {epoch + 1}/{epochs} | ")
        print(f"Train Loss: {train_loss:.4f} | Train Reconstruction Loss: {running_reconstruction_loss:.4f} | "
              f"Train KL Loss: {running_kl_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Reconstruction Loss: {val_reconstruction_loss:.4f} | "
              f"Val KL Loss: {val_kl_loss:.4f}")

        early_stopping(val_loss, epoch, model)
        if early_stopping.early_stop:
            early_stopping.load_best_model(model)
            print("Early stopping triggered.")
            break
    print("Training complete!")
