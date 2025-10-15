# `freegrad/transforms/basic.py`

from typing import Optional

import torch

from ..registry import register


@register("d(ReLU)")
def heaviside(ctx, grad_out: torch.Tensor, input: Optional[torch.Tensor], **_):
    if input is None:
        return grad_out
    return grad_out * (input > 0).to(grad_out.dtype)


@register("d(Linear)")
def identity(ctx, grad_out: torch.Tensor, input, **_):
    return grad_out


@register("rectangular")
def rectangular(
    ctx,
    grad_out: torch.Tensor,
    input: torch.Tensor,
    a: float = -0.5,
    b: float = 0.5,
    **_,
):
    mask = (input >= a) & (input <= b)
    return grad_out * mask.to(grad_out.dtype)


@register("triangular")
def triangular(
    ctx, grad_out: torch.Tensor, input: torch.Tensor, width: float = 1.0, **_
):
    # Picco a 0, lineare a zero a ±width
    g = (1 - (input.abs() / max(width, 1e-6))).clamp(min=0.0)
    return grad_out * g


@register("scale")
def scale(ctx, grad_out: torch.Tensor, input, s: float = 1.0, **_):
    return grad_out * s


@register("clip_norm")
def clip_norm(
    ctx, grad_out: torch.Tensor, input, max_norm: float = 1.0, eps: float = 1e-12, **_
):
    n = grad_out.norm().clamp_min(eps)
    factor = (max_norm / n).clamp(max=1.0)
    return grad_out * factor


@register("noise")
def noise(ctx, grad_out: torch.Tensor, input, sigma: float = 0.1, **_):
    return grad_out + sigma * torch.randn_like(grad_out)


@register("centralize")
def centralize(
    ctx, grad_out: torch.Tensor, input, dim: int = -1, keepdim: bool = True, **_
):
    m = grad_out.mean(dim=dim, keepdim=keepdim)
    return grad_out - m
