
from pathlib import Path

import torch
from torchvision import datasets, transforms

from model import MNISTCNN


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "best_model.pth"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load test data
test_dataset = datasets.MNIST(
    root=str(DATA_DIR),
    train=False,
    download=True,
    transform=transform
)

# Create and load the trained model
model = MNISTCNN().to(device)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)
model.eval()

# Select one test image
image, actual_label = test_dataset[0]

with torch.no_grad():
    image = image.unsqueeze(0).to(device)

    logits = model(image)
    probabilities = torch.softmax(logits, dim=1)

    predicted_label = logits.argmax(dim=1).item()
    confidence = probabilities[0, predicted_label].item()

print("Actual digit:", actual_label)
print("Predicted digit:", predicted_label)
print(f"Confidence: {confidence:.2%}")
