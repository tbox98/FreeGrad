# Selective Gradient Rules

Sometimes you only want to apply a gradient transformation to **some layers**.

---

## 🔹 Activation-only rules
Wrap just one activation:

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
Hook into parameter grads:

```python
from freegrad.hooks import attach_param_hooks
import freegrad as xg

attach_param_hooks(model)

with xg.use("clip_norm", params={"max_norm":0.5}, scope="params"):
    loss = criterion(model(x), y)
    loss.backward()
```

---

## 🔹 Mixed scope
Apply everywhere (`scope="all"`):

```python
with xg.use("centralize", scope="all"):
    loss.backward()
```
