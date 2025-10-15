# `freegrad/transforms/jamming.py`

import torch

from ..registry import register


@register("full_jam")
def full_jam(ctx, grad_out: torch.Tensor, input, **_):
    return grad_out * torch.rand_like(grad_out)


@register("positive_jam")
def positive_jam(ctx, grad_out: torch.Tensor, input: torch.Tensor, **_):
    rnd = torch.rand_like(grad_out)
    mask = (
        (input >= 0)
        if input is not None
        else torch.ones_like(grad_out, dtype=torch.bool)
    )
    return grad_out * torch.where(mask, rnd, torch.zeros_like(grad_out))


@register("rectangular_jam")
def rectangular_jam(
    ctx,
    grad_out: torch.Tensor,
    input: torch.Tensor,
    a: float = -5.0,
    b: float = 5.0,
    **_,
):
    rnd = torch.rand_like(grad_out)
    mask = (
        (input >= a) & (input <= b)
        if input is not None
        else torch.ones_like(grad_out, dtype=torch.bool)
    )
    return grad_out * torch.where(mask, rnd, torch.zeros_like(grad_out))
