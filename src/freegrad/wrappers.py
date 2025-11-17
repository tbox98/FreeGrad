from typing import Any, Callable, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .context import _ctx_get


def _relu6(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 6.0)


def _heaviside(x: torch.Tensor) -> torch.Tensor:
    return torch.heaviside(x, torch.tensor(0.0, dtype=x.dtype, device=x.device))


def _leaky_relu(x: torch.Tensor) -> torch.Tensor:
    return F.leaky_relu(x, negative_slope=0.01)


def _elu(x: torch.Tensor) -> torch.Tensor:
    return F.elu(x, alpha=1.0)


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
    def forward(ctx: Any, x: torch.Tensor, fwd_name: str) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.fwd_name = fwd_name

        # Snapshot: capture the rule active at this exact moment
        # This runs in the main thread, so _ctx_get() is correct.
        rule, params, scope = _ctx_get()
        ctx.fg_state = (rule, params, scope)

        return _FWD_MAP[fwd_name](x)

    # noinspection PyMethodOverriding
    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> Tuple[torch.Tensor, None]:
        (x,) = ctx.saved_tensors

        # Ignore the current thread's context. Use the snapshot.
        # This is needed to support CUDA backward passes, since they are usually run
        # in a separate thread, which cannot see the original contextvars
        rule, params, scope = ctx.fg_state

        # If for some reason the snapshot is empty (shouldn't happen due to wrapper logic),
        # we can just return the standard gradient.
        if (rule is None) or ("activations" not in scope and "all" not in scope):
            # Use PyTorch autograd for standard derivative
            with torch.enable_grad():
                x_req = x.detach().requires_grad_(True)
                y = _FWD_MAP[ctx.fwd_name](x_req)
                (grad_in,) = torch.autograd.grad(y, x_req, grad_out)
            return grad_in, None

        # Apply the captured rule
        grad_in = rule(None, grad_out, x, **(params or {}))
        return grad_in, None


class Activation(nn.Module):
    """Activation with optional FreeGrad backward."""

    def __init__(self, forward: str = "ReLU"):
        super().__init__()
        if forward not in _FWD_MAP:
            raise ValueError(
                f"Unsupported forward activation: {forward}. Supported: {sorted(_FWD_MAP.keys())}"
            )
        self._name = forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Always use _FreeGradActivationFn.
        # The backward pass will correctly handle
        # both standard autograd (if rule is None)
        # and freegrad (if rule is active).
        return _FreeGradActivationFn.apply(x, self._name)
