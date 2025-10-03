"""
Experiment 2 (Paper): MLP on DIGITS
Settings from Table 3 (p. 12): BS=64, LR=0.05
"""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import freegrad as fg
from freegrad.wrappers import Activation

torch.manual_seed(0)

digits = load_digits()
X = torch.tensor(digits.data, dtype=torch.float32) / 16.0
y = torch.tensor(digits.target)
n_classes = 10
y_oh = torch.nn.functional.one_hot(y, n_classes).float()
Xtr, Xte, ytr, yte = train_test_split(X, y_oh, test_size=0.2, random_state=0)
print(f"DIGITS => train: {Xtr.shape[0]}, test: {Xte.shape[0]}, input-dim: {X.shape[1]}")

bs, lr, epochs = 64, 0.05, 10
train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=bs, shuffle=True)

class MLP(nn.Module):
    def __init__(self, h=64, forward="Logistic"):
        super().__init__()
        self.fc1 = nn.Linear(64, h)
        self.act1 = Activation(forward)
        self.fc2 = nn.Linear(h, n_classes)
    def forward(self, x):
        return self.fc2(self.act1(self.fc1(x)))

def accuracy(model, X, Y):
    with torch.no_grad():
        logits = model(X)
        return float((logits.argmax(dim=1) == Y.argmax(dim=1)).float().mean())

def train_epoch(model, opt, loader):
    model.train()
    running_loss = 0.0; total = 0
    for xb, yb in loader:
        opt.zero_grad()
        logits = model(xb)
        loss = nn.CrossEntropyLoss()(logits, yb.argmax(dim=1))
        loss.backward()
        opt.step()
        running_loss += float(loss.item()) * yb.size(0)
        total += yb.size(0)
    return running_loss/total

# TIED
mlp = MLP(h=64, forward="Logistic")
opt = torch.optim.SGD(mlp.parameters(), lr=lr)
print("\n[Training] TIED (Logistic / d(Logistic)) with BS=64, LR=0.05")
for ep in range(1, epochs+1):
    loss = train_epoch(mlp, opt, train_loader)
    acc_tr = accuracy(mlp, Xtr, ytr)
    acc_te = accuracy(mlp, Xte, yte)
    print(f"epoch {ep:02d} | loss={loss:.4f} | acc_tr={acc_tr:.3f} | acc_te={acc_te:.3f}")

# UNTIED
mlp2 = MLP(h=64, forward="Logistic")
opt2 = torch.optim.SGD(mlp2.parameters(), lr=lr)
print("\n[Training] UNTIED (Logistic / 1) with BS=64, LR=0.05")
with fg.use(rule="d(Linear)", scope="activations"):
    for ep in range(1, epochs+1):
        loss = train_epoch(mlp2, opt2, train_loader)
        acc_tr = accuracy(mlp2, Xtr, ytr)
        acc_te = accuracy(mlp2, Xte, yte)
        print(f"epoch {ep:02d} | loss={loss:.4f} | acc_tr={acc_tr:.3f} | acc_te={acc_te:.3f}")

# Inference demo
x0 = Xte[0].unsqueeze(0)
y0 = int(yte.argmax(dim=1)[0].item())
with torch.no_grad():
    pred = mlp2(x0).argmax(dim=1).item()
print(f"\n[Inference] test sample#0 => pred={pred} | label={y0}")
