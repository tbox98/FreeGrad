"""
FreeGrad – SUC example (fixed)
---------------------------------------------------
- Forward: Logistic (sigmoid)
- Loss:    BCELoss  (matches probabilistic output)
- Paper-aligned settings (Table 2): BS=32, LR=0.1
- Trains both TIED (Logistic/d(Logistic)) and UNTIED (Logistic/1)
"""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import freegrad as fg
from freegrad.wrappers import Activation

torch.manual_seed(0)

class SUC(nn.Module):
    def __init__(self, forward="Logistic", d=10):
        super().__init__()
        self.lin = nn.Linear(d, 1)
        self.act = Activation(forward)

    def forward(self, x):
        # Probabilities in [0,1] (sigmoid), suitable for BCELoss
        return self.act(self.lin(x)).squeeze(-1)

def make_ones_dataset(n=2000, d=10, thresh=0.7):
    X = (torch.rand(n, d) > 0.5).float()
    y = (X.mean(dim=1) > thresh).float()
    return X, y

def run_epoch(model, loader, opt):
    model.train()
    running_loss, total, correct = 0.0, 0, 0
    criterion = nn.BCELoss()
    for xb, yb in loader:
        opt.zero_grad()
        probs = model(xb)                      # probs in [0,1]
        loss  = criterion(probs, yb)           # BCELoss expects probs
        loss.backward()
        opt.step()

        with torch.no_grad():
            pred = (probs >= 0.5).float()
            correct += (pred == yb).sum().item()
            running_loss += float(loss.item()) * yb.size(0)
            total += yb.size(0)
    return running_loss/total, correct/total

# Data
X, y = make_ones_dataset(n=2000, d=10, thresh=0.7)
print(f"Dataset => samples: {X.shape[0]}, features: {X.shape[1]}, pos-ratio: {y.mean().item():.2f}")
bs, lr, epochs = 1, 0.01, 20
loader = DataLoader(TensorDataset(X, y), batch_size=bs, shuffle=True)

# TIED: Logistic / d(Logistic)
tied = SUC("Logistic")
opt  = torch.optim.SGD(tied.parameters(), lr=lr)
print("\n[Training] TIED (Logistic / d(Logistic)) with BS=32, LR=0.1")
for ep in range(1, epochs+1):
    loss, acc = run_epoch(tied, loader, opt)
    print(f"epoch {ep:02d} | loss={loss:.4f} | acc={acc:.3f}")

# UNTIED: Logistic / 1 (constant)
untied = SUC("Logistic")
opt2   = torch.optim.SGD(untied.parameters(), lr=lr)
print("\n[Training] UNTIED (Logistic / 1) with BS=32, LR=0.1")
with fg.use(rule="d(Linear)", scope="activations"):
    for ep in range(1, epochs+1):
        loss, acc = run_epoch(untied, loader, opt2)
        print(f"epoch {ep:02d} | loss={loss:.4f} | acc={acc:.3f}")

# Inference demo
sample = X[0].unsqueeze(0)
with torch.no_grad():
    prob = untied(sample).item()
print(f"\n[Inference] sample#0 => pred_prob={prob:.3f} | label={int(y[0].item())}")
