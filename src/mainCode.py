import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import EarlyStopping
from EarlyStopping import EarlyStopping
from codeFor490.Dataset import CustomImageDataset, train_transform, test_transform
from codeFor490.GenContVit.genconvit import GenConViT


def main():
    # read from csv
    training = pd.read_csv(
        "train.csv")

    validating = pd.read_csv(
        "valid.csv")

    training = training[ : len(training)//10000]
    validating = training[ : len(validating)//10000]

    # encoder for converting class to string later
    label_encoder = LabelEncoder()
    label_encoder.fit(training["label"])

    # get training images path
    image_directory = r"real-vs-fake/"
    num_class = len(training["label"])

    # create dataset
    train_set = CustomImageDataset(training, image_directory, num_class, transform=train_transform)
    val_set = CustomImageDataset(validating, image_directory, num_class, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)

    # Send model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GenConViT(256, num_class).to(device)
    model = nn.DataParallel(model)
    model = model.cuda()

    #training and validating
    train_validate_model(model, device, train_loader, val_loader, epochs=200)

    # testing
    testing = pd.read_csv(
        "test.csv")
    testing = testing[ : len(testing)//10000]
    test_set = CustomImageDataset(testing, image_directory, num_class, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    print(test_model(model, test_loader, device))

# Function for training and validation
def train_validate_model(model, device, train_loader, val_loader, epochs=100):
    early_stopping = EarlyStopping(patience=10, verbose=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        # Training loop
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = 100 * correct / total

        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            early_stopping.load_best_model(model)
            print("Early stopping triggered.")
            break
    print("Training complete!")


def test_model(model, test_loader, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculations
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")

    return avg_loss, accuracy


if __name__ == "__main__":
    main()