import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from Util.EarlyStopping import EarlyStopping
from Util.Dataset import CustomImageDataset, train_transform, test_transform
from model.Mae import MaskedAutoEncoderViT
from model.GenConVit.genconvit import GenConViT
from Training.trainingMae import trainingMae
from Training.trainingVae import trainingVae
from model.Vae import VariationalAutoEncoder


def main():
    # read from csv
    trainings = pd.read_csv("Util/Labels/train.csv")
    validatings = pd.read_csv("Util/Labels/valid.csv")

    # get training images path
    image_directory = r"data/real-vs-fake/"


    trainings = trainings.sample(frac=1, random_state=42)
    validatings = validatings.sample(frac=1, random_state=42)
    training = trainings[:len(trainings)//10]
    validating = validatings[:len(validatings)//10]
    num_class = trainings["label"].nunique()
    print(len(training))
    print(len(validating))


    # use the first one for gpu if it does not run out of memory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mae_pretrained = True
    if os.path.exists('Output/VaeCheckPoint.pth') and mae_pretrained:
        model_vae = VariationalAutoEncoder()
        model_vae.load_state_dict(torch.load('Output/VaeCheckPoint.pth', weights_only=True, map_location=device))
    else:
        model_vae = trainingVae(training, validating, device, image_directory)
    print("Vae loaded")

    mae_pretrained = True
    if os.path.exists('Output/MaeCheckPoint.pth') and mae_pretrained:
        model_mae = MaskedAutoEncoderViT(256)
        model_mae.load_state_dict(torch.load('Output/MaeCheckPoint.pth', weights_only=True, map_location=device))
    else:
        model_mae = trainingMae(trainings, validatings, device, image_directory)
    print("Mae loaded")


    # create dataset
    train_set = CustomImageDataset(training, image_directory, num_class, transform=train_transform)
    val_set = CustomImageDataset(validating, image_directory, num_class, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=True, num_workers=1, pin_memory=True)

    # Send model to GPU
    model = GenConViT(256, num_class, model_mae, model_vae).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    del model_vae.decoder  # Delete the decoder from memory
    del model_mae.decoder
    torch.cuda.empty_cache()

    #training and validating
    train_validate_model(model, device, train_loader, val_loader, epochs=1000, checkpoint_path='Output/checkpoint.zip')

    # testing
    testing = pd.read_csv(os.path.join(os.getcwd(), "Util\\Labels\\test.csv"))
    test_set = CustomImageDataset(testing, image_directory, num_class, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=1, pin_memory=True)
    print(test_model(model, test_loader, device))

# Function for training and validation
def train_validate_model(model, device, train_loader, val_loader, epochs=1000, checkpoint_path=None):
    if checkpoint_path is not None:
        early_stopping = EarlyStopping(patience=21, verbose=True, zip_file=checkpoint_path)
    else:
        early_stopping = EarlyStopping(patience=21, verbose=True)

    criterion = nn.CrossEntropyLoss()
    optimizer_mae = optim.Adam(model.model_ed.parameters(), lr=0.00015, weight_decay=0.0001)
    optimizer_vae = optim.Adam(model.model_vae.parameters(), lr=0.00075, weight_decay=0.0001)
    scheduler_mae = ReduceLROnPlateau(optimizer_mae, mode='min', factor=0.1, patience=3)
    scheduler_vae = ReduceLROnPlateau(optimizer_vae, mode='min', factor=0.1, patience=3)
    model.to(device)
    start_epoch = 0

    # Freeze most layers in the mae but the last couple
    for param in model.model_ed.mae.parameters():
        param.requires_grad = False

    for block in model.model_ed.mae.blocks[-3:]:
        for param in block.parameters():
            param.requires_grad = True

    for param in model.model_ed.mae.norm.parameters():
        param.requires_grad = True
    model.model_ed.mae.pos_embed.requires_grad = True



    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        running_vae_loss = 0.0
        running_mae_loss = 0.0
        correct, total = 0, 0

        all_preds, all_labels = [], []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer_mae.zero_grad()
            optimizer_vae.zero_grad()
            mae, vae = model(inputs)



            output = (mae + vae) / 2
            vae_loss = criterion(vae, labels)
            mae_loss = criterion(mae, labels)
            total_loss_for_batch = (vae_loss + mae_loss) / 2

            # Accumulate the losses
            running_loss += total_loss_for_batch.item()
            running_vae_loss += vae_loss.item()
            running_mae_loss += mae_loss.item()

            _, predicted = torch.max(output, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

            mae_loss.backward()
            optimizer_mae.step()

            # Backpropagate for vae model
            vae_loss.backward()
            optimizer_vae.step()

        train_loss = running_loss / total
        train_acc = 100 * correct / total

        running_vae_loss /= total
        running_mae_loss /= total

        precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)

        model.eval()
        val_loss = 0.0
        val_vae_loss = 0.0
        val_mae_loss = 0.0
        correct, total = 0, 0

        all_val_preds, all_val_labels = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                mae, vae = model(inputs)

                output = (mae + vae) / 2
                vae_loss = criterion(vae, labels)
                mae_loss = criterion(mae, labels)
                total_loss_for_batch = (vae_loss + mae_loss) / 2
                val_loss += total_loss_for_batch.item()

                # Accumulate the validation losses
                val_vae_loss += vae_loss.item()
                val_mae_loss += mae_loss.item()

                _, predicted = torch.max(output, 1)
                all_val_preds.extend(predicted.detach().cpu().numpy())
                all_val_labels.extend(labels.detach().cpu().numpy())

                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_loss /= total
        val_acc = 100 * correct / total

        val_vae_loss /= total
        val_mae_loss /= total

        val_precision = precision_score(all_val_labels, all_val_preds, average='binary', zero_division=0)
        val_recall = recall_score(all_val_labels, all_val_preds, average='binary', zero_division=0)
        val_f1 = f1_score(all_val_labels, all_val_preds, average='binary', zero_division=0)
        val_auc = roc_auc_score(all_val_labels, all_val_preds)

        print(f"Epoch {epoch + 1}/{epochs} | ")
        print(f"Train Loss: {train_loss:.4f} | Train VAE Loss: {running_vae_loss:.4f} | "
            f"Train MAE Loss: {running_mae_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | ")
        print(f"Val Loss: {val_loss:.4f} | Val VAE Loss: {val_vae_loss:.4f} | "
            f"Val MAE Loss: {val_mae_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%")


        print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f} | ")
        print(f"Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f} | Val F1 Score: {val_f1:.4f} | Val AUC: {val_auc:.4f}")

        scheduler_mae.step(val_mae_loss)
        scheduler_vae.step(val_vae_loss)

        early_stopping(val_loss, epoch, model, optimizer_vae, scheduler_vae)

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
    correct = 0
    total = 0
    all_preds, all_labels = [], []

    with torch.no_grad():  # Disable gradient calculations
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            #mae, vae, reconstruction_loss, kl_loss = model(inputs)
            mae, vae = model(inputs)
            output = (mae + vae) / 2
            vae_loss = criterion(vae, labels)
            mae_loss = criterion(mae, labels)
 
            total_loss_for_batch = (vae_loss + mae_loss) / 2

            # Accumulate losses
            total_loss += total_loss_for_batch.item()
            total_vae_loss += vae_loss.item()
            total_mae_loss += mae_loss.item()

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
    accuracy = 100 * correct / total

    print(f"Average VAE Loss: {avg_vae_loss:.4f}")
    print(f"Average MAE Loss: {avg_mae_loss:.4f}")
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
