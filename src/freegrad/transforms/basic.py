from typing import Any, Optional

import torch

from ..registry import register


@register("d(ReLU)")
def heaviside(
    ctx: Any, grad_out: torch.Tensor, tin: Optional[torch.Tensor], **_: Any
) -> torch.Tensor:
    """Applies the derivative of ReLU (a Heaviside step function).

    This is the standard surrogate gradient for a ReLU activation:
    `grad_in = grad_out * (tin > 0)`.
    If tin is None (e.g., from a parameter hook), it returns the
    gradient unmodified.

    Args:
        ctx: (Unused) The autograd context.
        grad_out (torch.Tensor): The incoming gradient.
        tin (Optional[torch.Tensor]): The input tensor from the forward pass.

    Returns:
        torch.Tensor: The modified gradient.
    """
    if tin is None:
        return grad_out
    return grad_out * (tin > 0).to(grad_out.dtype)


@register("d(Linear)")
def identity(ctx: Any, grad_out: torch.Tensor, tin: Any, **_: Any) -> torch.Tensor:
    """Applies the identity transform (derivative of a linear function).

    This rule simply passes the gradient through unmodified.

    Args:
        ctx: (Unused) The autograd context.
        grad_out (torch.Tensor): The incoming gradient.
        tin: (Unused) The input tensor from the forward pass.

    Returns:
        torch.Tensor: The unmodified gradient.
    """
    return grad_out


@register("rectangular")
def rectangular(
    ctx: Any,
    grad_out: torch.Tensor,
    tin: torch.Tensor,
    a: float = -0.5,
    b: float = 0.5,
    **_: Any,
) -> torch.Tensor:
    """Applies a rectangular surrogate gradient.

    The gradient is passed through (multiplied by 1) only where the
    forward pass input `x` was in the range `[a, b]`, and is zeroed
    otherwise.

    Args:
        ctx: (Unused) The autograd context.
        grad_out (torch.Tensor): The incoming gradient.
        tin (torch.Tensor): The input tensor from the forward pass.
        a (float, optional): The lower bound of the rectangular window.
            Defaults to -0.5.
        b (float, optional): The upper bound of the rectangular window.
            Defaults to 0.5.

    Returns:
        torch.Tensor: The modified gradient.
    """
    mask = (tin >= a) & (tin <= b)
    return grad_out * mask.to(grad_out.dtype)


@register("triangular")
def triangular(
    ctx: Any, grad_out: torch.Tensor, tin: torch.Tensor, width: float = 1.0, **_: Any
) -> torch.Tensor:
    """Applies a triangular surrogate gradient.

    The gradient is scaled by a factor that peaks at 1.0 (at tin=0)
    and linearly decays to 0.0 at `tin=±width`.

    Args:
        ctx: (Unused) The autograd context.
        grad_out (torch.Tensor): The incoming gradient.
        tin (torch.Tensor): The input tensor from the forward pass.
        width (float, optional): The half-width of the triangular pulse.
            Defaults to 1.0.

    Returns:
        torch.Tensor: The modified gradient.
    """
    g = (1 - (tin.abs() / max(width, 1e-6))).clamp(min=0.0)
    return grad_out * g


@register("scale")
def scale(
    ctx: Any, grad_out: torch.Tensor, tin: Any, s: float = 1.0, **_: Any
) -> torch.Tensor:
    """Scales the gradient by a constant factor `s`.

    Args:
        ctx: (Unused) The autograd context.
        grad_out (torch.Tensor): The incoming gradient.
        tin: (Unused) The input tensor from the forward pass.
        s (float, optional): The scaling factor. Defaults to 1.0.

    Returns:
        torch.Tensor: The scaled gradient (`grad_out * s`).
    """
    return grad_out * s


@register("clip_norm")
def clip_norm(
    ctx: Any,
    grad_out: torch.Tensor,
    tin: Any,
    max_norm: float = 1.0,
    eps: float = 1e-12,
    **_: Any,
) -> torch.Tensor:
    """Clips the L2 norm of the gradient tensor.

    If the total L2 norm of `grad_out` exceeds `max_norm`, the
    gradient tensor is scaled down to have norm `max_norm`.

    Args:
        ctx: (Unused) The autograd context.
        grad_out (torch.Tensor): The incoming gradient.
        tin: (Unused) The input tensor from the forward pass.
        max_norm (float, optional): The maximum allowed norm.
            Defaults to 1.0.
        eps (float, optional): Epsilon for numerical stability when
            computing the norm. Defaults to 1e-12.

    Returns:
        torch.Tensor: The clipped gradient.
    """
    n = grad_out.norm().clamp_min(eps)
    factor = (max_norm / n).clamp(max=1.0)
    return grad_out * factor


@register("noise")
def noise(
    ctx: Any, grad_out: torch.Tensor, tin: Any, sigma: float = 0.1, **_: Any
) -> torch.Tensor:
    """Adds zero-mean Gaussian noise to the gradient.

    Noise is sampled from `N(0, sigma^2)`.

    Args:
        ctx: (Unused) The autograd context.
        grad_out (torch.Tensor): The incoming gradient.
        tin: (Unused) The input tensor from the forward pass.
        sigma (float, optional): The standard deviation of the noise.
            Defaults to 0.1.

    Returns:
        torch.Tensor: The noisy gradient (`grad_out + noise`).
    """
    return grad_out + sigma * torch.randn_like(grad_out)


@register("centralize")
def centralize(
    ctx: Any,
    grad_out: torch.Tensor,
    tin: Any,
    dim: int = -1,
    keepdim: bool = True,
    **_: Any,
) -> torch.Tensor:
    """Centralizes the gradient by subtracting its mean along a dimension.

    This makes the gradient tensor have a mean of zero along the
    specified `dim`.

    Args:
        ctx: (Unused) The autograd context.
        grad_out (torch.Tensor): The incoming gradient.
        tin: (Unused) The input tensor from the forward pass.
        dim (int, optional): The dimension along which to compute the
            mean. Defaults to -1.
        keepdim (bool, optional): Whether the output tensor has `dim`
            retained or not. Defaults to True.

    Returns:
        torch.Tensor: The centralized gradient.
    """
    m = grad_out.mean(dim=dim, keepdim=keepdim)
    return grad_out - m
