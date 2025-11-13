# Custom Gradient Rules

You can easily define your own gradient transformation or compose existing ones.

---

## 🔹 Registering a Rule Function

The easiest way to create a custom rule is to define a function and register it with `xg.register`.

A rule function must have the following signature:
`fn(ctx, grad_out, input, **params)`

* `ctx`: The `torch.autograd.Context` (rarely used, can be `None`).
* `grad_out`: The incoming gradient tensor (i.e., $dL/dy$).
* `input`: The input tensor from the forward pass (i.e., $x$).
    * **Note:** This will be `None` if the rule is running on a parameter gradient (from `scope="params"`). Your rule should handle this case.
* `**params`: A dictionary that collects any parameters passed to `xg.use(..., params={...})`.

The function must return the new gradient, $dL/dx$.

### Example:

Here is a rule that applies a threshold and adds noise.

```python
import torch
import freegrad as xg

@xg.register("noisy_threshold")
def noisy_threshold(ctx, grad_out, input, t: float = 0.0, sigma: float = 0.1):
    # Handle parameter hooks where input is None
    if input is None:
        mask = 1.0
    else:
        mask = (input >= t).to(grad_out.dtype)

    noise = sigma * torch.randn_like(grad_out)
    return grad_out * mask + noise
````

Use it:

```python
act = Activation("ReLU")
with xg.use("noisy_threshold", params={"t":0.5, "sigma":0.05}):
    y = act(x).sum()
    y.backward()
```

---

## 🔹 Composing Rules

You can create a new rule by composing existing rules in series using `xg.compose`. The rules are applied in the order they are listed.

```python
import freegrad as xg

# Create a new rule that first clips, then adds noise
clip_and_noise = xg.compose("clip_norm", "noise")

# You can register it (optional)
xg.register("clip_then_noise")(clip_and_noise)

# Use it directly
with xg.use(
    clip_and_noise,
    params={"max_norm": 1.0, "sigma": 0.01},
    scope="all"
):
    loss.backward()
```

---

## 🔹 Advanced: Custom `torch.autograd.Function`

For complete control over *both* forward and backward, or to create a new operation that is not just an activation, you can define a `torch.autograd.Function`.

**Note:** This is a standard PyTorch feature. A `Function` defined this way **will not** interact with the `xg.use` context manager. It has its own, hard-coded backward pass.

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
        # Custom backward: 2*x*grad_output + noise
        noise = ctx.sigma * torch.randn_like(x)
        grad_in = grad_output * (2*x) + noise

        # Return grads for (x, sigma)
        return grad_in, None

def square_noise(x, sigma=0.1):
    return SquareNoiseFn.apply(x, sigma)

# ---
# Use it like a regular function
# ---
x = torch.randn(5, requires_grad=True)
y = square_noise(x, sigma=0.1).sum()
y.backward()
print(x.grad) # Will have noise
```
