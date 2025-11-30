# Getting Started

Welcome to **freegrad** — an extension that lets you experiment with **alternative backward rules** alongside PyTorch **autograd**.

---

## 🔧 Installation

Clone the repository and install in editable mode with development dependencies:

```bash
git clone https://github.com/tbox98/freegrad.git
cd freegrad
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

  - **Rules** → Functions that transform gradients during backpropagation.
  - **Scopes** → Choose where to apply the rule: `"activations"`, `"params"`, or `"all"`.
  - **Context Manager** → Wrap training code in `with xg.use(...):` to activate a rule.
  - **Activation Wrapper** → A drop-in replacement `xg.Activation` that can intercept the backward pass.
  - **Composition** → Combine multiple rules in sequence using `xg.compose`.

---

## 🧰 Built-in Rules

`freegrad` includes a variety of pre-defined rules.

### Standard Derivatives

  * `"d(ReLU)"`: The Heaviside step function (standard ReLU derivative).
  * `"d(Linear)"`: The identity function (passes gradient through).

### Surrogate Gradients

  * `"rectangular"`: Passes gradient only if the input was within a range `[a, b]`.
  * `"triangular"`: Scales gradient by a triangular function, peaking at 0.

### Gradient Transforms

  * `"scale"`: Multiplies the gradient by a scalar `s`.
  * `"clip_norm"`: Clips the L2 norm of the *entire* gradient tensor to `max_norm`.
  * `"noise"`: Adds Gaussian noise `N(0, sigma^2)` to the gradient.
  * `"centralize"`: Subtracts the mean of the gradient (per-dimension).

### Jamming Rules

  * `"full_jam"`: Multiplies gradient by uniform `U(0, 1)` noise.
  * `"positive_jam"`: Applies `U(0, 1)` noise where `tin >= 0`, zeros otherwise.
  * `"rectangular_jam"`: Applies `U(0, 1)` noise where `a <= tin <= b`, zeros otherwise.

---

## ⚡ Supported Activations

The `xg.Activation` wrapper supports the following forward functions:

  * `"Linear"` (Identity)
  * `"ReLU"`
  * `"ReLU6"`
  * `"LeakyReLU"`
  * `"ELU"`
  * `"Tanh"`
  * `"Logistic"` (Sigmoid)
  * `"SiLU"`
  * `"GELU"`
  * `"Softplus"`
  * `"Heaviside"`

---

## Next Steps

- [Selective gradient rules](guides/selective-rules.md)
- [Define your own custom rules](guides/custom-rules.md)
