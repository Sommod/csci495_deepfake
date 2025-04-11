import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from EarlyStopping import EarlyStopping
from Dataset import CustomImageDataset, train_transform, test_transform
from GenConVit.Mae import MaskedAutoEncoderViT
from GenConVit.genconvit import GenConViT
from GenConVit.trainingMae import trainingMae


def main():
    # read from csv
    training = pd.read_csv(
        "train.csv")
    validating = pd.read_csv(
        "valid.csv")
    # get training images path
    image_directory = r"real-vs-fake/"

    # use the first one for gpu if it does not run out of memory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    mae_pretrained = True
    if os.path.exists('MaeCheckPoint.pth') and mae_pretrained:
        model_mae = MaskedAutoEncoderViT(256)
        model_mae.load_state_dict(torch.load('MaeCheckPoint.pth', weights_only=True, map_location=device))
    else:
        model_mae = trainingMae(training, validating, device, image_directory)
    print("Mae loaded")

    training = training.sample(frac=1, random_state=42)
    validating = validating.sample(frac=1, random_state=42)
    training = training[ : len(training)//100]
    validating = validating[ : len(validating)//100]
    print(len(training))
    print(len(validating))
    num_class = training["label"].nunique()

    # create dataset
    train_set = CustomImageDataset(training, image_directory, num_class, transform=train_transform)
    val_set = CustomImageDataset(validating, image_directory, num_class, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=True, num_workers=6, pin_memory=True)

    # Send model to GPU
    model = GenConViT(256, num_class, model_mae).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    #training and validating
    train_validate_model(model, device, train_loader, val_loader, epochs=100, checkpoint_path="checkpoint.zip")

    # testing
    testing = pd.read_csv(
        "test.csv")
    test_set = CustomImageDataset(testing, image_directory, num_class, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=1, pin_memory=True)
    print(test_model(model, test_loader, device))

# Function for training and validation
def train_validate_model(model, device, train_loader, val_loader, epochs=100, checkpoint_path=None):
    if checkpoint_path is not None:
        early_stopping = EarlyStopping(patience=6, verbose=True, zip_file=checkpoint_path)
    else:
        early_stopping = EarlyStopping(patience=6, verbose=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    model.to(device)
    start_epoch = 0


    # Resume from checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path) and os.path.isfile(checkpoint_path):
        model, optimizer, scheduler, start_epoch = early_stopping.load_checkpoint(model, optimizer, scheduler, device)

    warmup_epochs = 30

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        running_vae_loss = 0.0
        running_mae_loss = 0.0
        running_reconstruction_loss = 0.0
        running_kl_loss = 0.0
        correct, total = 0, 0

        all_preds, all_labels = [], []

        kl_weight = min(1.0, (epoch + 1) / warmup_epochs)

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            mae, vae, reconstruction_loss, kl_loss = model(inputs)
            reconstruction_loss *= 5

            kl_loss *= kl_weight

            output = (mae + vae)/2
            vae_loss = criterion(vae, labels)
            mae_loss = criterion(mae, labels)
            loss = (vae_loss + mae_loss)/2
            total_loss_for_batch = loss + reconstruction_loss + kl_loss

            # Accumulate the losses
            running_loss += total_loss_for_batch.item()
            running_vae_loss += vae_loss.item()
            running_mae_loss += mae_loss.item()
            running_reconstruction_loss += reconstruction_loss.item()
            running_kl_loss += kl_loss.item()

            _, predicted = torch.max(output, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

            print(f"Train Loss: {running_loss:.4f} | Train VAE Loss: {running_vae_loss:.4f} | "
                  f"Train MAE Loss: {running_mae_loss:.4f} | Train Reconstruction Loss: {running_reconstruction_loss:.4f} | "
                  f"Train KL Loss: {running_kl_loss:.4f}")

            total_loss_for_batch.backward()  # Backpropagate only during training
            optimizer.step()

        train_loss = running_loss / total
        train_acc = 100 * correct / total

        running_vae_loss /= total
        running_mae_loss /= total
        running_reconstruction_loss /= total
        running_kl_loss /= total

        precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)

        model.eval()
        val_loss = 0.0
        val_vae_loss = 0.0
        val_mae_loss = 0.0
        val_reconstruction_loss = 0.0
        val_kl_loss = 0.0
        correct, total = 0, 0

        all_val_preds, all_val_labels = [], []


        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                mae, vae, reconstruction_loss, kl_loss = model(inputs)
                reconstruction_loss *= 5
                kl_loss *= kl_weight

                output = (mae + vae) / 2
                vae_loss = criterion(vae, labels)
                mae_loss = criterion(mae, labels)
                loss = (vae_loss + mae_loss)/2
                total_loss_for_batch = loss + reconstruction_loss + kl_loss
                val_loss += total_loss_for_batch.item()

                # Accumulate the validation losses
                val_vae_loss += vae_loss.item()
                val_mae_loss += mae_loss.item()
                val_reconstruction_loss += reconstruction_loss.item()
                val_kl_loss += kl_loss.item()

                all_val_preds.extend(predicted.detach().cpu().numpy())
                all_val_labels.extend(labels.detach().cpu().numpy())

                _, predicted = torch.max(output, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_loss /= total
        val_acc = 100 * correct / total

        val_vae_loss /= total
        val_mae_loss /= total
        val_reconstruction_loss /= total
        val_kl_loss /= total


        val_precision = precision_score(all_val_labels, all_val_preds, average='binary', zero_division=0)
        val_recall = recall_score(all_val_labels, all_val_preds, average='binary', zero_division=0)
        val_f1 = f1_score(all_val_labels, all_val_preds, average='binary', zero_division=0)
        val_auc = roc_auc_score(all_val_labels, all_val_preds)

        # Print average losses for train and validation
        print(f"Epoch {epoch + 1}/{epochs} | ")
        print(f"Train Loss: {train_loss:.4f} | Train VAE Loss: {running_vae_loss:.4f} | "
              f"Train MAE Loss: {running_mae_loss:.4f} | Train Reconstruction Loss: {running_reconstruction_loss:.4f} | "
              f"Train KL Loss: {running_kl_loss:.4f} | Train Acc: {train_acc:.2f}% | ")
        print(f"Val Loss: {val_loss:.4f} | Val VAE Loss: {val_vae_loss:.4f} | "
              f"Val MAE Loss: {val_mae_loss:.4f} | Val Reconstruction Loss: {val_reconstruction_loss:.4f} | "
              f"Val KL Loss: {val_kl_loss:.4f} | Val Acc: {val_acc:.2f}%")

        print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f} | ")
        print(f"Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f} | Val F1 Score: {val_f1:.4f} | Val AUC: {val_auc:.4f}")



        scheduler.step(val_loss)

        early_stopping(val_loss, epoch, model, optimizer, scheduler)

        if early_stopping.early_stop:
            early_stopping.load_best_model(model)
            print("Early stopping triggered.")
            break

    print("Training complete!")


def test_model(model, test_loader, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_vae_loss = 0.0
    total_mae_loss = 0.0
    total_reconstruction_loss = 0.0
    total_kl_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []

    with torch.no_grad():  # Disable gradient calculations
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            mae, vae, reconstruction_loss, kl_loss = model(inputs)
            output = (mae + vae)/2
            vae_loss = criterion(vae, labels)
            mae_loss = criterion(mae, labels)
            loss = (vae_loss + mae_loss)/2
            total_loss_for_batch = loss + reconstruction_loss + kl_loss

            # Accumulate losses
            total_loss += total_loss_for_batch.item()
            total_vae_loss += vae_loss.item()
            total_mae_loss += mae_loss.item()
            total_reconstruction_loss += reconstruction_loss.item()
            total_kl_loss += kl_loss.item()

            # Compute accuracy
            _, predicted = torch.max(output, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate average losses
    avg_loss = total_loss / total
    avg_vae_loss = total_vae_loss / total
    avg_mae_loss = total_mae_loss / total
    avg_reconstruction_loss = total_reconstruction_loss / total
    avg_kl_loss = total_kl_loss / total
    accuracy = 100 * correct / total

    print(f"Average VAE Loss: {avg_vae_loss:.4f}")
    print(f"Average MAE Loss: {avg_mae_loss:.4f}")
    print(f"Average Reconstruction Loss: {avg_reconstruction_loss:.4f}")
    print(f"Average KL Loss: {avg_kl_loss:.4f}")
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")

    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    auc = roc_auc_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Test AUC: {auc:.4f}")

    return avg_loss, accuracy

if __name__ == "__main__":
    main()
