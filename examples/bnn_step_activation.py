"""
BNN application (Paper Section 5): Step forward + Rectangular backward
Settings from Tables 13–14 (pp. 17–18): use LR=0.1; small batches (e.g., BS=32)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import freegrad as fg
from freegrad.wrappers import _FWD_MAP, Activation

torch.manual_seed(0)

# Add Step to forward map (Heaviside)
_FWD_MAP["Step"] = lambda z: (z >= 0).to(z.dtype)


# Toy 2D dataset: label = 1 if x0 + x1 > 0, else 0
def make_linear_2d(n=2048):
    x = torch.randn(n, 2)
    y = (x.sum(dim=1) > 0).float()
    return x, y


class TinyBNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1, bias=True)
        self.act = Activation("Step")

    def forward(self, x):
        return self.act(self.fc(x).squeeze(-1))


X, y = make_linear_2d()
print(f"Toy dataset => {X.shape[0]} samples, pos-ratio={y.mean().item():.2f}")
bs, lr, epochs = 32, 0.1, 10
loader = DataLoader(TensorDataset(X, y), batch_size=bs, shuffle=True)

model = TinyBNN()
print("Model:", model)
print("Params:", sum(p.numel() for p in model.parameters()))
opt = torch.optim.SGD(model.parameters(), lr=lr)


def run_epoch():
    model.train()
    running_loss, total, correct = 0.0, 0, 0
    for xb, yb in loader:
        opt.zero_grad()
        logits = model(xb)  # forward with Step (non-differentiable)
        loss = nn.BCEWithLogitsLoss()(logits, yb)
        loss.backward()
        opt.step()
        with torch.no_grad():
            pred = (logits > 0).float()
            correct += (pred == yb).sum().item()
            running_loss += float(loss.item()) * yb.size(0)
            total += yb.size(0)
    return running_loss / total, correct / total


print(
    "\n[Training] Step forward + Rectangular backward (a=-0.5,b=0.5) with BS=32, LR=0.1"
)
with fg.use("rectangular", params={"a": -0.5, "b": 0.5}, scope="activations"):
    for ep in range(1, epochs + 1):
        loss, acc = run_epoch()
        print(f"epoch {ep:02d} | loss={loss:.4f} | acc={acc:.3f}")

# Inference demo
x0 = torch.tensor([[0.7, -0.2]])
with torch.no_grad():
    prob = torch.sigmoid(model(x0)).item()
print(f"\n[Inference] x={x0.tolist()} => pred_prob={prob:.3f}")
