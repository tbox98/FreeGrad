# Getting Started

Welcome to **freegrad** — an extension that lets you experiment with **alternative backward rules** alongside PyTorch **autograd**.

---

## 🔧 Installation

Clone the repository and install in editable mode with development dependencies:

```bash
git clone https://github.com/tbox98/BackProp.git
cd BackProp
pip install -e .[dev]
```

---

## 🚀 Quick Example

```python
import torch
import freegrad as xg
from freegrad.wrappers import Activation

# Input tensor
x = torch.randn(5, requires_grad=True)

# Standard ReLU forward
act = Activation(forward="ReLU")

# Apply custom backward rule only on activations
with xg.use(rule="rectangular_jam", params={"a": -1.0, "b": 1.0}, scope="activations"):
    y = act(x).sum()
    y.backward()

print("Input:", x)
print("Gradients with rectangular_jam:", x.grad)
```

---

## 📚 Key Concepts

- **Rules** → functions that transform gradients during backpropagation (e.g. noise, clipping, jamming).  
- **Scopes** → choose where to apply the rule: `"activations"`, `"params"`, or `"all"`.  
- **Context manager** → wrap training code in `with xg.use(...):` to activate a rule.  
- **Activation wrapper** → drop-in replacement for ReLU, Tanh, etc., with flexible backward rules.

---

## Next Steps

- [Selective gradient rules](guides/selective-rules.md)  
- [Define your own custom rules](guides/custom-rules.md)  
