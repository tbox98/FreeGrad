"""
Experiment 3 (Paper): LeNet-style CNN on MNIST
Settings from Table 4 (p. 13): BS=256, LR=0.01, Momentum=0.9, Epochs=20
(Rectangular gradient is robust)
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import freegrad as fg
from freegrad.wrappers import Activation
import time

# --- Hyperparameters from the paper ---
BATCH_SIZE = 256
LEARNING_RATE = 0.01
MOMENTUM = 0.9
EPOCHS = 30
RECT_A = -0.5
RECT_B = 0.5
# ------------------------------------

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


class LeNetMini(nn.Module):
    def __init__(self, forward="Logistic"):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.act1 = Activation(forward)
        self.pool = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.act2 = Activation(forward)
        self.fc = nn.Linear(16 * 5 * 5, 10)

    def forward(self, x):
        z = self.pool(self.act1(self.conv1(x)))
        z = self.pool(self.act2(self.conv2(z)))
        return self.fc(z.view(z.size(0), -1))


# --- Data Loading ---
transform = T.Compose([T.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- Model, Optimizer, and Loss ---
model = LeNetMini(forward="Logistic").to(device)
print("Model:", model)
print("Params:", sum(p.numel() for p in model.parameters()))

opt = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
criterion = nn.CrossEntropyLoss()


# --- Evaluation Function ---
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
print(f"\n[Training] Logistic / Rectangular(a={RECT_A},b={RECT_B}) on activations for {EPOCHS} epochs")

# Use the freegrad context manager for the entire training process
with fg.use("rectangular", params={"a": RECT_A, "b": RECT_B}, scope="activations"):
    for epoch in range(EPOCHS):
        model.train()
        start_time = time.time()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        for bi, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

            # Accumulate stats
            total_loss += loss.item() * yb.size(0)
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total_samples += yb.size(0)

        # Epoch stats
        epoch_time = time.time() - start_time
        avg_loss = total_loss / total_samples
        train_acc = total_correct / total_samples
        test_acc = evaluate(model, test_loader)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | Time: {epoch_time:.2f}s | Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.3f} | Test Acc: {test_acc:.3f}")

print("\nTraining finished.")

# --- Inference Demo ---
x0, y0 = next(iter(test_loader))
x0, y0 = x0[:1].to(device), y0[:1].to(device)
model.eval()  # Set model to evaluation mode for inference
with torch.no_grad():
    pred = model(x0).argmax(dim=1).item()
print(f"\n[Inference] MNIST sample => pred={pred} | label={int(y0.item())}")
