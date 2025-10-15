"""
Experiment 5 (Paper): CNN with Gradient Jamming
Settings from Tables 6–8 (pp. 14–15): BS=64, LR=0.01
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

import freegrad as fg
from freegrad.wrappers import Activation

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"


class SmallCNN(nn.Module):
    def __init__(self, forward="ReLU"):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, 3, padding=1)
        self.act = Activation(forward)
        self.pool = nn.AvgPool2d(2)
        self.fc = nn.Linear(8 * 14 * 14, 10)

    def forward(self, x):
        z = self.pool(self.act(self.conv(x)))
        return self.fc(z.view(z.size(0), -1))


ds = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=T.ToTensor()
)
loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)

model = SmallCNN().to(device)
opt = torch.optim.SGD(model.parameters(), lr=0.01)


def train_one_rule(rule: str, params=None, max_batches=4):
    print(f"\n[Training] Jamming rule: {rule} params={params or {}} (few batches)")
    total, correct, total_loss = 0, 0, 0.0
    with fg.use(rule, params=params or {}, scope="activations"):
        for bi, (xb, yb) in enumerate(loader, start=1):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = nn.CrossEntropyLoss()(logits, yb)
            loss.backward()
            opt.step()
            total += yb.size(0)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total_loss += float(loss.item()) * yb.size(0)
            print(f"batch {bi:02d} | loss={loss.item():.4f}")
            if bi >= max_batches:
                break
    print(f"avg_loss={total_loss/total:.4f} | acc={correct/total:.3f}")


train_one_rule("full_jam")
train_one_rule("positive_jam")
train_one_rule("rectangular_jam", params={"a": -5.0, "b": 5.0})

# Inference demo
x0, y0 = next(iter(loader))
x0, y0 = x0[:1].to(device), y0[:1].to(device)
with torch.no_grad():
    pred = model(x0).argmax(dim=1).item()
print(f"\n[Inference] sample => pred={pred} | label={int(y0.item())}")
