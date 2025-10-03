# Custom Gradient Rules

You can easily define your own gradient transformation.

---

## 🔹 Register a custom rule

```python
import torch
import freegrad as xg

@xg.register("noisy_threshold")
def noisy_threshold(ctx, grad_out, input, t: float = 0.0, sigma: float = 0.1):
    mask = (input >= t).float()
    noise = sigma * torch.randn_like(grad_out)
    return grad_out * mask + noise
```

Use it:

```python
act = Activation("ReLU")
with xg.use("noisy_threshold", params={"t":0.5, "sigma":0.05}):
    y = act(x).sum()
    y.backward()
```

---

## 🔹 Define a new autograd Function

```python
import torch

class SquareNoiseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, sigma=0.1):
        ctx.save_for_backward(x)
        ctx.sigma = sigma
        return x**2

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        noise = ctx.sigma * torch.randn_like(x)
        return grad_output * (2*x) + noise, None

def square_noise(x, sigma=0.1):
    return SquareNoiseFn.apply(x, sigma)
```

---

This lets you create entirely new ops with custom forward and backward behavior.
