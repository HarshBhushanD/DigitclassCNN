
# import random
# from pathlib import Path

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, random_split
# from torchvision import datasets, transforms

# from model import MNISTCNN


# # Reproducibility
# torch.manual_seed(42)
# random.seed(42)

# # Paths
# ROOT = Path(__file__).resolve().parent.parent
# DATA_DIR = ROOT / "data"
# MODEL_DIR = ROOT / "models"
# MODEL_DIR.mkdir(parents=True, exist_ok=True)

# # Device
# device = torch.device(
#     "cuda" if torch.cuda.is_available() else "cpu"
# )
# print("Device:", device)

# # Preprocessing
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))
# ])

# # Load training dataset
# full_train_dataset = datasets.MNIST(
#     root=str(DATA_DIR),
#     train=True,
#     download=True,
#     transform=transform
# )

# # Keep a separate validation set
# train_dataset, val_dataset = random_split(
#     full_train_dataset,
#     [55000, 5000],
#     generator=torch.Generator().manual_seed(42)
# )

# # Load the official test set separately
# test_dataset = datasets.MNIST(
#     root=str(DATA_DIR),
#     train=False,
#     download=True,
#     transform=transform
# )

# # DataLoaders
# train_loader = DataLoader(
#     train_dataset,
#     batch_size=64,
#     shuffle=True,
#     num_workers=0
# )

# val_loader = DataLoader(
#     val_dataset,
#     batch_size=64,
#     shuffle=False,
#     num_workers=0
# )

# test_loader = DataLoader(
#     test_dataset,
#     batch_size=64,
#     shuffle=False,
#     num_workers=0
# )

# print("Training samples:", len(train_dataset))
# print("Validation samples:", len(val_dataset))
# print("Test samples:", len(test_dataset))


import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model import MNISTCNN


# Reproducibility
torch.manual_seed(42)
random.seed(42)

# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "best_model.pth"

# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Device:", device)

# Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST training data
full_train_dataset = datasets.MNIST(
    root=str(DATA_DIR),
    train=True,
    download=True,
    transform=transform
)

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [55000, 5000],
    generator=torch.Generator().manual_seed(42)
)

# Load official test data
test_dataset = datasets.MNIST(
    root=str(DATA_DIR),
    train=False,
    download=True,
    transform=transform
)

# Data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Test samples:", len(test_dataset))

# Initialize model, loss function, and optimizer
model = MNISTCNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# Training configuration
EPOCHS = 5
best_val_accuracy = 0.0


def evaluate(model, loader):
    model.eval()

    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    average_loss = total_loss / total

    return average_loss, accuracy


# Training loop
for epoch in range(EPOCHS):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_accuracy = 100.0 * correct / total

    val_loss, val_accuracy = evaluate(model, val_loader)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.2f}% "
        f"Val Loss: {val_loss:.4f} "
        f"Val Accuracy: {val_accuracy:.2f}%"
    )

    # Save the best model based on validation accuracy
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(f"Best model saved: {MODEL_PATH}")

# Load the best checkpoint before final testing
model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

test_loss, test_accuracy = evaluate(model, test_loader)

print("\nTraining complete!")
print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.2f}%")
print(f"Model saved at: {MODEL_PATH}")



from sklearn.metrics import confusion_matrix, classification_report

model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        logits = model(images)
        predictions = logits.argmax(dim=1).cpu()

        all_predictions.extend(predictions.tolist())
        all_labels.extend(labels.tolist())

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_predictions))

print("\nClassification Report:")
print(
    classification_report(
        all_labels,
        all_predictions,
        digits=4
    )
)

