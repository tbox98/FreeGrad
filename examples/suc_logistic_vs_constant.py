"""
FreeGrad – SUC with Uniform-over-Hamming-Weight (UHW)
-----------------------------------------------------
- Data: binary vectors x in {0,1}^d with *uniform* distribution over Hamming weights K
        (i.e., P(K=k) = 1/(d+1), then choose k ones uniformly among positions).
- Label: y = 1{ mean(x) > τ }  (linear threshold, linearly separable)
- Forward: Logistic (sigmoid) → outputs probabilities
- Loss:    BCELoss (on probabilities)
- Compare:
    * TIED   = Logistic / d(Logistic)    (native autograd)
    * UNTIED = Logistic / 1              (fg.use(rule="d(Linear)", scope="activations"))
- Train/Test split: 80/20
- Defaults: BS=32, LR=0.1, EPOCHS=30, d=10, τ=0.5
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from contextlib import nullcontext

import freegrad as fg
from freegrad.wrappers import Activation

# -----------------------
# Repro / device setup
# -----------------------
SEED = 0
BS = 32
LR = 0.1
EPOCHS = 30
N = 2000  # total samples
D = 10  # features
TAU = 0.7  # threshold on mean(x)

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------
# Model
# -----------------------
class SUC(nn.Module):
    def __init__(self, forward="Logistic", d=D):
        super().__init__()
        self.lin = nn.Linear(d, 1)
        self.act = Activation(forward)

    def forward(self, x):
        return self.act(self.lin(x)).squeeze(-1)  # probabilities


# -----------------------
# Data: uniform-over-hamming-weight (UHW)
# -----------------------
@torch.no_grad()
def make_binary_uhw(n: int, d: int, tau: float = 0.5):
    """Uniform over Hamming weights: pick K ~ Uniform{0,...,d}, then choose K ones uniformly.
    Returns X in {0,1}^{nxd}, y in {0,1}^n with y = 1{mean(x) > tau}.
    """
    X = torch.zeros(n, d)
    Ks = torch.randint(low=0, high=d + 1, size=(n,))
    for i, k in enumerate(Ks.tolist()):
        if k > 0:
            idx = torch.randperm(d)[:k]
            X[i, idx] = 1.0
    y = (X.mean(dim=1) > tau).float()
    return X, y


X, y = make_binary_uhw(N, D, TAU)

# 80/20 split with shuffling
perm = torch.randperm(X.size(0))
cut = int(0.8 * X.size(0))
idx_train = perm[:cut]
idx_test = perm[cut:]

X_train, y_train = X[idx_train], y[idx_train]
X_test, y_test = X[idx_test], y[idx_test]

print(
    f"Dataset => total: {N} | train: {len(idx_train)} | test: {len(idx_test)} | D: {D} | "
    f"τ={TAU} | pos-ratio train: {y_train.mean().item():.2f} | test: {y_test.mean().item():.2f}"
)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BS, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BS, shuffle=False)

# -----------------------
# Train / Eval loops
# -----------------------
criterion = nn.BCELoss()


def run_epoch(model, loader, opt=None, rule_ctx=nullcontext()):
    is_train = opt is not None
    model.train(is_train)
    total_loss, total_correct, total = 0.0, 0, 0

    with rule_ctx:
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            if is_train:
                opt.zero_grad(set_to_none=True)

            probs = model(xb)  # probs in [0,1]
            loss = criterion(probs, yb)

            if is_train:
                loss.backward()
                opt.step()

            with torch.no_grad():
                pred = (probs >= 0.5).float()
                total_correct += (pred == yb).sum().item()
                total_loss += float(loss.item()) * yb.size(0)
                total += yb.size(0)

    return total_loss / total, total_correct / total


def train_model(model, train_loader, test_loader, lr=LR, epochs=EPOCHS, untied=False):
    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    # Untied = apply rule "d(Linear)" to activations (Logistic / 1)
    ctx = fg.use(rule="d(Linear)", scope="activations") if untied else nullcontext()

    print(
        f"\n[Training] {'UNTIED (Logistic / 1)' if untied else 'TIED (Logistic / d(Logistic))'} "
        f"BS={BS}, LR={lr}, EPOCHS={epochs}"
    )

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, opt=opt, rule_ctx=ctx)
        te_loss, te_acc = run_epoch(
            model, test_loader, opt=None, rule_ctx=nullcontext()
        )
        print(
            f"epoch {ep:02d} | train loss={tr_loss:.4f} acc={tr_acc:.3f} | test loss={te_loss:.4f} acc={te_acc:.3f}"
        )

    return model


# -----------------------
# Train both variants
# -----------------------
tied_model = train_model(SUC("Logistic"), train_loader, test_loader, untied=False)
untied_model = train_model(SUC("Logistic"), train_loader, test_loader, untied=True)

# -----------------------
# Inference demo
# -----------------------
sample = X_test[0].unsqueeze(0).to(device)
label = int(y_test[0].item())
with torch.no_grad():
    p_tied = tied_model(sample).item()
    p_untied = untied_model(sample).item()

print(
    f"\n[Inference] sample#0 | label={label} | tied_prob={p_tied:.3f} | untied_prob={p_untied:.3f}"
)
