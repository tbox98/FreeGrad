## `freegrad/wrappers.py`

from typing import Callable, Dict
import torch
import torch.nn as nn
from .context import _ctx_get

# Mapping of supported forward activation functions
_FWD_MAP: Dict[str, Callable] = {
    "ReLU": torch.relu,
    "ReLU6": lambda z: torch.clamp(z, 0.0, 6.0),
    "Tanh": torch.tanh,
    "Logistic": torch.sigmoid,
    "Linear": lambda z: z,
}


class _XActivationFn(torch.autograd.Function):
    """Custom autograd Function that applies a forward activation
    and optionally overrides the backward gradient via the freegrad context.
    """

    @staticmethod
    def forward(ctx, x, fwd: Callable):
        ctx.save_for_backward(x)
        return fwd(x)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        rule, params, scope = _ctx_get()
        # If no rule is active or the scope excludes activations, return unchanged gradient
        if (rule is None) or ("activations" not in scope and "all" not in scope):
            return grad_out, None
        # Apply custom gradient rule
        grad_in = rule(ctx, grad_out, x, **params)
        return grad_in, None


class Activation(nn.Module):
    """Activation layer with standard forward and backward controlled by freegrad context."""

    def __init__(self, forward: str = "ReLU"):
        super().__init__()
        if forward not in _FWD_MAP:
            raise ValueError(f"Unsupported forward activation: {forward}")
        self._fwd = _FWD_MAP[forward]

    def forward(self, x):
        return _XActivationFn.apply(x, self._fwd)
