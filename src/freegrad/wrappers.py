# freegrad/wrappers.py
from typing import Callable, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .context import _ctx_get


def _relu6(x): return torch.clamp(x, 0.0, 6.0)


def _heaviside(x): return torch.heaviside(x, torch.tensor(0.0, dtype=x.dtype, device=x.device))


def _leaky_relu(x): return F.leaky_relu(x, negative_slope=0.01)


def _elu(x): return F.elu(x, alpha=1.0)


_FWD_MAP: Dict[str, Callable] = {
    "Linear": lambda z: z,
    "ReLU": F.relu,
    "ReLU6": _relu6,
    "LeakyReLU": _leaky_relu,
    "ELU": _elu,
    "Tanh": torch.tanh,
    "Logistic": torch.sigmoid,
    "SiLU": F.silu,
    "GELU": F.gelu,
    "Softplus": F.softplus,
    "Heaviside": _heaviside,
}


class _FreeGradActivationFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, fwd_name):
        ctx.save_for_backward(x)
        ctx.fwd_name = fwd_name
        return _FWD_MAP[fwd_name](x)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        rule, params, scope = _ctx_get()
        # Safety fallback: if no rule/scope, compute true derivative via autograd.
        if (rule is None) or ("activations" not in scope and "all" not in scope):
            with torch.enable_grad():
                x_req = x.detach().requires_grad_(True)
                y = _FWD_MAP[ctx.fwd_name](x_req)
                (grad_in,) = torch.autograd.grad(y, x_req, grad_out, retain_graph=False, create_graph=False)
            return grad_in, None
        # Untied case: apply user rule
        grad_in = rule(None, grad_out, x, **(params or {}))
        return grad_in, None


class Activation(nn.Module):
    """Activation with optional FreeGrad backward."""

    def __init__(self, forward: str = "ReLU"):
        super().__init__()
        if forward not in _FWD_MAP:
            raise ValueError(f"Unsupported forward activation: {forward}. Supported: {sorted(_FWD_MAP.keys())}")
        self._name = forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rule, params, scope = _ctx_get()
        # TIED baseline: use raw op so PyTorch computes the true derivative
        if (rule is None) or ("activations" not in scope and "all" not in scope):
            return _FWD_MAP[self._name](x)
        # UNTIED: intercept backward
        return _FreeGradActivationFn.apply(x, self._name)
