from typing import Any

import torch

from ..registry import register


@register("full_jam")
def full_jam(ctx: Any, grad_out: torch.Tensor, tin: Any, **_: Any) -> torch.Tensor:
    """Applies "gradient jamming" by multiplying with uniform random noise.

    The gradient is scaled by `U(0, 1)` noise.

    Args:
        ctx: (Unused) The autograd context.
        grad_out (torch.Tensor): The incoming gradient.
        tin: (Unused) The input tensor from the forward pass.

    Returns:
        torch.Tensor: The jammed gradient.
    """
    return grad_out * torch.rand_like(grad_out)


@register("positive_jam")
def positive_jam(
    ctx: Any, grad_out: torch.Tensor, tin: torch.Tensor, **_: Any
) -> torch.Tensor:
    """Applies gradient jamming only to elements where the input was positive.

    The gradient is scaled by `U(0, 1)` where `tin >= 0`, and is
    zeroed otherwise.

    Args:
        ctx: (Unused) The autograd context.
        grad_out (torch.Tensor): The incoming gradient.
        tin (torch.Tensor): The input tensor from the forward pass.

    Returns:
        torch.Tensor: The partially jammed gradient.
    """
    rnd = torch.rand_like(grad_out)
    mask = (
        (tin >= 0) if tin is not None else torch.ones_like(grad_out, dtype=torch.bool)
    )
    return grad_out * torch.where(mask, rnd, torch.zeros_like(grad_out))


@register("rectangular_jam")
def rectangular_jam(
    ctx: Any,
    grad_out: torch.Tensor,
    tin: torch.Tensor,
    a: float = -5.0,
    b: float = 5.0,
    **_: Any,
) -> torch.Tensor:
    """Applies gradient jamming only within a rectangular window.

    The gradient is scaled by `U(0, 1)` where `a <= tin <= b`,
    and is zeroed otherwise.

    Args:
        ctx: (Unused) The autograd context.
        grad_out (torch.Tensor): The incoming gradient.
        tin (torch.Tensor): The input tensor from the forward pass.
        a (float, optional): The lower bound of the window.
            Defaults to -5.0.
        b (float, optional): The upper bound of the window.
            Defaults to 5.0.

    Returns:
        torch.Tensor: The partially jammed gradient.
    """
    rnd = torch.rand_like(grad_out)
    mask = (
        (tin >= a) & (tin <= b)
        if tin is not None
        else torch.ones_like(grad_out, dtype=torch.bool)
    )
    return grad_out * torch.where(mask, rnd, torch.zeros_like(grad_out))
