# Selective Gradient Rules

Sometimes you only want to apply a gradient transformation to **some layers**.

---

## 🔹 Activation-only rules
Wrap just one activation using `freegrad.wrappers.Activation`.

```python
from freegrad.wrappers import Activation
import freegrad as xg
import torch

# An example 'x' tensor that needs a gradient
x = torch.tensor([-1.0, 2.0, 3.0], requires_grad=True)
act = Activation("ReLU")

with xg.use("noise", params={"sigma":0.05}, scope="activations"):
    y = act(x).sum()
    y.backward()
```

---

## 🔹 Parameter-only rules
To apply rules directly to `nn.Parameter` gradients (e.g., `model.weight.grad`), you must first attach the global parameter hook.

```python
from freegrad.hook import attach_param_hooks
import freegrad as xg
import torch
import torch.nn as nn

# Example model, loss, and data
model = nn.Linear(5, 2)
x = torch.randn(1, 5)
y = torch.randn(1, 2)
criterion = nn.MSELoss()

# This iterates model.parameters() and registers a hook for each.
attach_param_hooks(model)

# This rule will now run on parameter gradients
with xg.use("clip_norm", params={"max_norm":0.5}, scope="params"):
    loss = criterion(model(x), y)
    loss.backward()
```

**Note:** When a rule runs on a parameter, the `tin` tensor argument will be `None`, as there is no corresponding forward-pass activation.

---

## 🔹 Mixed scope
Apply everywhere (`scope="all"`):

```python
import torch
import torch.nn as nn
import freegrad as xg
from freegrad.wrappers import Activation
from freegrad.hook import attach_param_hooks

# Example model and data
model = nn.Linear(5, 5)
x = torch.randn(4, 5)
target = torch.randn(4, 5)

attach_param_hooks(model)
criterion = nn.MSELoss()
act = Activation("ReLU")

print("Running backward pass with 'centralize' rule...")
with xg.use("centralize", scope="all"):
    linear_out = model(x)
    y = act(linear_out)
    loss = criterion(y, target)
    loss.backward()
```
