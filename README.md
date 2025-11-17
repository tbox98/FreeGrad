# FreeGrad

Alternative backward rules and gradient transforms alongside PyTorch **autograd**.

[![CI](https://github.com/tbox98/FreeGrad/actions/workflows/ci.yml/badge.svg)](https://github.com/tbox98/FreeGrad/actions/workflows/ci.yml)
[![Tests](https://github.com/tbox98/FreeGrad/actions/workflows/tests.yml/badge.svg)](https://github.com/tbox98/FreeGrad/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/freegrad.svg)](https://pypi.org/project/freegrad/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-gh--pages-blue)](https://tbox98.github.io/FreeGrad/)

---

## 🚀 Features
- Register and compose custom **gradient rules** (backward transforms)
- Apply rules via a **context manager** to activations and/or params
- Lightweight wrappers for **activation layers**
- Works *alongside* standard **autograd** without patching PyTorch

---

## 📦 Installation

```bash
# Core package only (from PyPI)
pip install freegrad

# Development install (with testing, linting, docs, examples, etc.)
pip install -e '.[dev]'
```

> 💡 Note: If you’re using **zsh** (default on macOS), don’t forget the quotes around `.[dev]`.

---

## 🧪 Running Tests

After installing in development mode:

```bash
pip install -e '.[dev]'
```

Run the full test suite with:

```bash
pytest
```

Run with coverage reporting:

```bash
pytest --cov=freegrad --cov-report=term-missing
```

Run a specific test file or test:

```bash
pytest tests/test_wrappers.py -v
pytest tests/test_wrappers.py::test_activation_forward_relu -v
```

---

## 🎓 Running Examples

The repository includes runnable scripts under [`examples/`](examples/) that replicate experiments from the paper.

Install dev dependencies:

```bash
pip install -e '.[dev]'
```

Run an example:

```bash
python examples/suc_logistic_vs_constant.py
python examples/mlp_digits_constant_vs_tied.py
python examples/lenet_mnist_rectangular.py
python examples/cnn_gradient_jamming.py
python examples/bnn_step_activation.py
```

> 💡 Some examples require datasets (e.g. MNIST via `torchvision`, DIGITS via `scikit-learn`). They will be downloaded automatically the first time you run them.

---

## ⚡ Quickstart

```python
import torch
import freegrad as fg
from freegrad.wrappers import Activation

x = torch.randn(8, requires_grad=True)
act = Activation(forward="ReLU")

with fg.use(rule="rectangular_jam", params={"a": -1.0, "b": 1.0}, scope="activations"):
    y = act(x).sum()
    y.backward()

print(x.grad)
```

---

## 🛠️ Makefile Shortcuts

This project includes a `Makefile` with useful commands:

```bash
# Run everything (install deps, build paper, tests, and examples)
make

# Build the JOSS-style paper PDF only
# Requires pandoc >= 2.11 and xelatex installed on your system
make pdf

# Run the test suite with coverage
make test

# Run all examples sequentially
make examples

# Run a specific example
make suc     # Single-Unit Classifier (SUC)
make mlp     # MLP on DIGITS
make lenet   # LeNet on MNIST with Rectangular gradient
make cnn     # CNN with Gradient Jamming
make bnn     # BNN with Step activation
```

> 💡 The `install` step (`pip install -e '.[dev]'`) is included automatically when running `make`, `make test`, or `make examples`.

---

## 📖 Documentation

👉 Full docs available here: [https://tbox98.github.io/FreeGrad/](https://tbox98.github.io/FreeGrad/)

---

## 🤝 Contributing

Contributions are welcome!
Please read [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 📄 License

Distributed under the [MIT License](LICENSE).
