import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, recall_score, precision_score
from torch import nn
from torch.utils.data import DataLoader
import seaborn as sns

from Dataset import CustomImageDataset, test_transform
from GenContVit.Mae import MaskedAutoEncoderViT
from GenContVit.Vae import VariationalAutoEncoder
from GenContVit.genconvit import GenConViT


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # test dataset
    image_directory = r"real-vs-fake/"
    testing = pd.read_csv(
        "test.csv")
    num_class = testing["label"].nunique()
    testing = testing.sample(frac=1, random_state=42)
    testing = testing[:len(testing) // 10]
    test_set = CustomImageDataset(testing, image_directory, num_class, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=1, pin_memory=True)
    print(len(testing))

    # load model
    model_mae = MaskedAutoEncoderViT(256)
    model_vae = VariationalAutoEncoder()
    model = GenConViT(256, num_class, model_mae, model_vae).to(device)
    model.load_state_dict(torch.load('checkPoint.pth', weights_only=True, map_location=device))

    print(test_model(model, test_loader, device))

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

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    main()