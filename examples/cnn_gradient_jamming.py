"""
Experiment 5 (Paper): CNN with Gradient Jamming
Settings from Table 6, 7, 8 & Section 4.8 (pp. 13-15):
    - Architecture: LeNet-5 (Canonical)
    - Dataset: MNIST (Confirms Table 12, despite typo in text)
    - Batch Size: 64 (Selected from Table 6 for stability)
    - Learning Rate: 0.01
    - Jamming Rules:
        1. Full-Jamming: Replace magnitude with U[0,1], keep sign.
        2. Positive-Jamming: U[0,1] if grad > 0, else 0.
        3. Rectangular-Jamming: U[0,1] if grad in [-5, 5], else 0.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import time

import freegrad as fg
from freegrad.wrappers import Activation

torch.manual_seed(0)
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using device: {device}")

# --- Hyperparameters ---
BATCH_SIZE = 64
LEARNING_RATE = 0.01
EPOCHS = 20


# --- 1. Architecture: Canonical LeNet-5 ---
# (Matches Section 4.1 and Experiment 3/4/5 usage)
class LeNet5(nn.Module):
    def __init__(self, forward="ReLU"):
        super().__init__()
        # Input: 1 x 28 x 28
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.act1 = Activation(forward)
        self.pool = nn.AvgPool2d(2)

        self.conv2 = nn.Conv2d(6, 16, 5)
        self.act2 = Activation(forward)
        # Pool 2 creates 16 x 5 x 5

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.act3 = Activation(forward)

        self.fc2 = nn.Linear(120, 84)
        self.act4 = Activation(forward)

        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        y = self.pool(self.act1(self.conv1(x)))
        y = self.pool(self.act2(self.conv2(y)))
        y = y.view(y.size(0), -1)
        y = self.act3(self.fc1(y))
        y = self.act4(self.fc2(y))
        return self.fc3(y)


# --- 2. Data Loading (MNIST) ---
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


# --- 3. Training Routine ---
def train_experiment(rule_name, rule_params=None):
    print("\n========================================")
    print(f"[Experiment] LeNet-5 with {rule_name} Gradient")
    print(f"Params: {rule_params or 'None'} | BS={BATCH_SIZE} | LR={LEARNING_RATE}")
    print("========================================")

    model = LeNet5(forward="ReLU").to(device)
    opt = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Apply the jamming rule to all activations in the model
    with fg.use(rule_name, params=rule_params or {}, scope="activations"):
        for epoch in range(EPOCHS):
            model.train()
            total_loss, correct, total = 0.0, 0, 0
            start_t = time.time()

            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()

                total_loss += loss.item() * yb.size(0)
                correct += (logits.argmax(dim=1) == yb).sum().item()
                total += yb.size(0)

            # Validation
            model.eval()
            test_correct, test_total = 0, 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    test_correct += (logits.argmax(dim=1) == yb).sum().item()
                    test_total += yb.size(0)

            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Train Acc: {correct / total:.3f} | "
                f"Test Acc: {test_correct / test_total:.3f} | "
                f"Time: {time.time() - start_t:.1f}s"
            )


# --- 4. Run Experiments (Table 6, 7, 8) ---

# A. Full-Jamming (Table 6)
# "Backward gradient is replaced by a uniformly random value from [0, 1]... sign is preserved."
# Assumes freegrad rule 'full_jam' implements: grad = sign(grad) * Uniform(0, 1)
train_experiment("full_jam")

# B. Positive-Jamming (Table 7)
# "Replaced by [0, 1] exclusively when feedback gradient is non-negative. Negative set to 0."
# Assumes freegrad rule 'positive_jam' implements: grad = Uniform(0, 1) if grad >= 0 else 0
train_experiment("positive_jam")

# C. Rectangular-Jamming (Table 8)
# "Replaced by [0, 1] within interval [-5, 5], set to 0 outside."
# Assumes freegrad rule 'rectangular_jam' implements this logic.
train_experiment("rectangular_jam", {"a": -5.0, "b": 5.0})
