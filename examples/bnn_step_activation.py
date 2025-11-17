"""
BNN application (Paper Section 5.1): Single Unit Classifier (SUC) with Heaviside Activation
Settings from Table 13 (p. 17):
    - Dataset: Ones (Uniform Hamming Weight)
    - Dimensions: 64 (matches a column in Table 13)
    - Activation: Heaviside Step Function
    - Backward: Constant (Step/1) - equivalent to rule='d(Linear)'
    - BS=64, LR=0.1
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import freegrad as fg
from freegrad.wrappers import Activation

torch.manual_seed(0)
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using device: {device}")


# --- 1. Dataset Generation (Paper Section 4.2) ---
def make_binary_uhw(n=2048, d=64, tau=0.5):
    """
    Uniform over Hamming weights (UHW) / 'Ones' Dataset.
    Generates binary vectors x in {0,1}^d.
    Label y = 1 if mean(x) > tau, else 0.
    See 'Ones Dataset (Synthetic)'.
    """
    X = torch.zeros(n, d)
    # Pick Hamming weight K uniformly from {0, ..., d}
    Ks = torch.randint(low=0, high=d + 1, size=(n,))
    for i, k in enumerate(Ks.tolist()):
        if k > 0:
            # Randomly choose k positions to be 1
            idx = torch.randperm(d)[:k]
            X[i, idx] = 1.0

    # Label based on threshold of active bits
    y = (X.mean(dim=1) > tau).float()
    return X, y


# Configuration matches Table 13 columns
N_SAMPLES = 4096
DIM = 64
BATCH_SIZE = 64
LR = 0.1
EPOCHS = 20

# Generate and Split Data (80/20)
X, y = make_binary_uhw(n=N_SAMPLES, d=DIM)
split_idx = int(0.8 * N_SAMPLES)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Dataset 'Ones' (d={DIM}) => Train: {len(X_train)}, Test: {len(X_test)}")
print(f"Pos-ratio Train: {y_train.mean():.2f}, Test: {y_test.mean():.2f}")

train_loader = DataLoader(
    TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True
)
test_loader = DataLoader(
    TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False
)


# --- 2. Model Definition (SUC) ---
class BinarySUC(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1, bias=True)
        self.act = Activation("Heaviside")

    def forward(self, x):
        # SUC: Dense -> Activation. See Fig 3
        return self.act(self.fc(x).squeeze(-1))


model = BinarySUC(DIM)
print("Model:", model)

# Standard SGD
opt = torch.optim.SGD(model.parameters(), lr=LR)

# Using MSELoss for binary 0/1 targets is compatible with Heaviside output
loss_fn = nn.MSELoss()


def run_epoch(loader, is_train=True):
    model.train(is_train)
    total_loss, correct, total = 0.0, 0, 0

    for xb, yb in loader:
        if is_train:
            opt.zero_grad()

        # Forward pass (Heaviside function -> outputs 0.0 or 1.0)
        preds = model(xb)

        loss = loss_fn(preds, yb)

        if is_train:
            loss.backward()
            opt.step()

        with torch.no_grad():
            # Accuracy: exact match of 0/1 predictions
            correct += (preds == yb).sum().item()
            total_loss += loss.item() * yb.size(0)
            total += yb.size(0)

    return total_loss / total, correct / total


# --- 3. Training with FreeGrad ---
# "Step/1" configuration from Table 13
# Forward = Heaviside (Step)
# Backward = 1 (Constant) -> rule="d(Linear)"
print("\n[Training] BNN SUC: Heaviside Forward / Constant Backward (Table 13 'Step/1')")
print(f"Settings: BS={BATCH_SIZE}, LR={LR}")

with fg.use(rule="d(Linear)", scope="activations"):
    for ep in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(train_loader, is_train=True)
        te_loss, te_acc = run_epoch(test_loader, is_train=False)

        print(
            f"Epoch {ep:02d} | "
            f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.3f} | "
            f"Test Loss: {te_loss:.4f} Acc: {te_acc:.3f}"
        )

# --- 4. Inference Demo ---
x0, y0 = X_test[0], y_test[0]
with torch.no_grad():
    pred = model(x0.unsqueeze(0)).item()
print(f"\n[Inference] Test Sample #0: Label={int(y0)} => Pred={int(pred)}")
