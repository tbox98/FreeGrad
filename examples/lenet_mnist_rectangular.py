"""
Experiment 3 (Paper): LeNet-5 on MNIST
Settings from Table 4 (p. 13): BS=256, LR=0.01, Momentum=0.9, Epochs=30
"""

import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

import freegrad as fg
from freegrad.wrappers import Activation

# --- Hyperparameters ---
BATCH_SIZE = 256
LEARNING_RATE = 0.01
MOMENTUM = 0.9
EPOCHS = 30
RECT_A = -0.5
RECT_B = 0.5

torch.manual_seed(0)
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using device: {device}")


class LeNet5(nn.Module):
    def __init__(self, forward="Logistic"):
        super().__init__()
        # Input: 1 x 28 x 28
        # C1: 1 -> 6 channels, 5x5 kernel, padding=2 -> 6 x 28 x 28
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.act1 = Activation(forward)

        # S2: AvgPool 2x2 -> 6 x 14 x 14
        self.pool = nn.AvgPool2d(2)

        # C3: 6 -> 16 channels, 5x5 kernel -> 16 x 10 x 10
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.act2 = Activation(forward)
        # S4: AvgPool 2x2 -> 16 x 5 x 5

        # C5 (Dense): 16*5*5 -> 120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.act3 = Activation(forward)

        # F6: 120 -> 84
        self.fc2 = nn.Linear(120, 84)
        self.act4 = Activation(forward)

        # Output: 84 -> 10
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Layer 1: Conv -> Act -> Pool
        y = self.conv1(x)
        y = self.act1(y)
        y = self.pool(y)

        # Layer 2: Conv -> Act -> Pool
        y = self.conv2(y)
        y = self.act2(y)
        y = self.pool(y)

        # Flatten
        y = y.view(y.size(0), -1)

        # Layer 3: FC -> Act
        y = self.fc1(y)
        y = self.act3(y)

        # Layer 4: FC -> Act
        y = self.fc2(y)
        y = self.act4(y)

        # Output Layer
        y = self.fc3(y)
        return y


# --- Data Loading ---
transform = T.Compose([T.ToTensor()])
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False
)

# --- Model Setup ---
model = LeNet5(forward="Logistic").to(device)
print("Model Architecture:", model)
print("Total Params:", sum(p.numel() for p in model.parameters()))

opt = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
criterion = nn.CrossEntropyLoss()


# --- Evaluation Helper ---
def evaluate(model, loader):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total_samples += yb.size(0)
    return total_correct / total_samples


# --- Training Loop ---
print(f"\n[Training] Logistic / Rectangular(a={RECT_A},b={RECT_B}) on activations")

# Use the freegrad context manager to override gradients for all Activations
with fg.use("rectangular", params={"a": RECT_A, "b": RECT_B}, scope="activations"):
    for epoch in range(EPOCHS):
        model.train()
        start_time = time.time()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

            total_loss += loss.item() * yb.size(0)
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total_samples += yb.size(0)

        epoch_time = time.time() - start_time
        avg_loss = total_loss / total_samples
        train_acc = total_correct / total_samples
        test_acc = evaluate(model, test_loader)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Time: {epoch_time:.2f}s | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Train Acc: {train_acc:.3f} | "
            f"Test Acc: {test_acc:.3f}"
        )

print("\nTraining finished.")
