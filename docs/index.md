# freegrad

> Alternative backward rules and gradient transforms alongside PyTorch **autograd**.

---

## 🔥 Why freegrad?
`freegrad` is a lightweight research framework that lets you **decouple forward and backward** in PyTorch.  
It provides:
- Custom **gradient rules** (e.g. noise, clipping, jamming)  
- A clean **context manager** API for applying rules selectively  
- Drop-in **activation wrappers** with alternative backward behavior  
- Compatibility with vanilla **autograd** — nothing is patched  

---

## 🚀 Quickstart

```bash
pip install -e .[dev]
```

```python
import torch
import freegrad as xg
from freegrad.wrappers import Activation

x = torch.randn(8, requires_grad=True)
act = Activation(forward="ReLU")

with xg.use(rule="rectangular_jam", params={"a": -1.0, "b": 1.0}, scope="activations"):
    y = act(x).sum()
    y.backward()

print(x.grad)
```

---

## 📖 Documentation

- [Getting started](getting-started.md)  
- [Selective gradient rules](guides/selective-rules.md)  
- [Defining custom rules](guides/custom-rules.md)  

---

## 🤝 Contributing
Contributions are very welcome! Please see [CONTRIBUTING.md](https://github.com/tbox98/BackProp/blob/main/CONTRIBUTING.md)  
and our [Code of Conduct](https://github.com/tbox98/BackProp/blob/main/CODE_OF_CONDUCT.md).  

---

## 📄 License
Released under the [MIT License](https://github.com/tbox98/BackProp/blob/main/LICENSE).
