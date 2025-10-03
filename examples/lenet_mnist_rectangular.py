"""
Experiment 3 (Paper): LeNet-style CNN on MNIST
Settings from Table 4 (p. 13): BS=256, LR=0.01 (Rectangular gradient is robust)
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import freegrad as fg
from freegrad.wrappers import Activation

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

class LeNetMini(nn.Module):
    def __init__(self, forward="Logistic"):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.act1 = Activation(forward)
        self.pool = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.act2 = Activation(forward)
        self.fc = nn.Linear(16*7*7, 10)
    def forward(self, x):
        z = self.pool(self.act1(self.conv1(x)))
        z = self.pool(self.act2(self.conv2(z)))
        return self.fc(z.view(z.size(0), -1))

transform = T.Compose([T.ToTensor()])
train = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
loader = torch.utils.data.DataLoader(train, batch_size=256, shuffle=True)

model = LeNetMini(forward="Logistic").to(device)
print("Model:", model)
print("Params:", sum(p.numel() for p in model.parameters()))

opt = torch.optim.SGD(model.parameters(), lr=0.01)

def run_batches(max_batches=5):
    model.train()
    total, correct, total_loss = 0, 0, 0.0
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
    return total_loss/total, correct/total

print("\n[Training] Logistic / Rectangular(a=-0.5,b=0.5) on activations (few batches)")
with fg.use("rectangular", params={"a": -0.5, "b": 0.5}, scope="activations"):
    avg_loss, acc = run_batches(max_batches=5)
    print(f"avg_loss={avg_loss:.4f} | acc={acc:.3f}")

# Inference demo
x0, y0 = next(iter(loader))
x0, y0 = x0[:1].to(device), y0[:1].to(device)
with torch.no_grad():
    pred = model(x0).argmax(dim=1).item()
print(f"\n[Inference] MNIST sample => pred={pred} | label={int(y0.item())}")
