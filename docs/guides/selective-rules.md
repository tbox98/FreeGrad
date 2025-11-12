# Selective Gradient Rules

Sometimes you only want to apply a gradient transformation to **some layers**.

---

## 🔹 Activation-only rules
Wrap just one activation using `freegrad.wrappers.Activation`.

```python
from freegrad.wrappers import Activation
import freegrad as xg

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

# This iterates model.parameters() and registers a hook for each.
attach_param_hooks(model)

# This rule will now run on parameter gradients
with xg.use("clip_norm", params={"max_norm":0.5}, scope="params"):
    loss = criterion(model(x), y)
    loss.backward()
```

**Note:** When a rule runs on a parameter, the `input` tensor argument will be `None`, as there is no corresponding forward-pass activation.

---

## 🔹 Mixed scope
Apply everywhere (`scope="all"`):

```python
with xg.use("centralize", scope="all"):
    loss.backward()
```
