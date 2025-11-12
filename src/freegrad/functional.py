from types import TracebackType
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Type

import torch

from .context import use


@torch.no_grad()
def jvp(
    f: Callable,
    x: torch.Tensor,
    v: torch.Tensor,
    *,
    rule: Any = None,
    params: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """Computes an alternative Jacobian-vector product: J_f(x) @ v.

    This is a naive implementation using symmetric differences, intended
    for independence from the standard backward pass. For performance-critical
    use, prefer PyTorch's built-in autograd JVP when available.

    Args:
        f (Callable): The function to differentiate.
        x (torch.Tensor): The point at which to evaluate the JVP.
        v (torch.Tensor): The vector for the product.
        rule (Optional[Any], optional): A freegrad rule to apply during the
            function's forward evaluations. Defaults to None.
        params (Optional[Dict], optional): Parameters for the freegrad rule.
            Defaults to None.

    Returns:
        torch.Tensor: The resulting Jacobian-vector product.
    """
    eps = 1e-3
    with use(rule=rule, params=params or {}, scope=("all",)) if rule else nullcontext():
        y_pos = f(x + eps * v)
        y_neg = f(x - eps * v)
    return (y_pos - y_neg) / (2 * eps)


class nullcontext:
    """A no-op context manager.

    This class provides an empty `__enter__` and `__exit__` method,
    making it a valid, do-nothing context manager. It is used as a
    fallback in `jvp` and `vjp` when no `rule` is provided.
    """

    def __enter__(self) -> "nullcontext":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> Literal[False]:
        return False


def vjp(
    f: Callable,
    x: torch.Tensor,
    *,
    rule: Any = None,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
    """Computes an alternative vector-Jacobian product using the freegrad context.

    Returns a (y, vjp_fn) tuple, where y = f(x) and vjp_fn is a
    function that computes the VJP (v^T @ J_f(x)).

    Args:
        f (Callable): The function to differentiate.
        x (torch.Tensor): The point at which to evaluate the VJP.
        rule (Optional[Any], optional): A freegrad rule to apply during the
            function's forward evaluation. Defaults to None.
        params (Optional[Dict], optional): Parameters for the freegrad rule.
            Defaults to None.

    Returns:
        Tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]: A tuple
        (y, vjp_fn), where y is the forward pass result f(x) and vjp_fn
        is the backward function.
    """
    with use(rule=rule, params=params or {}, scope=("all",)) if rule else nullcontext():
        x = x.detach().requires_grad_(True)
        y = f(x)

    def vjp_fn(v: torch.Tensor) -> torch.Tensor:
        (g,) = torch.autograd.grad(y, x, v, retain_graph=True, allow_unused=False)
        return g

    return y, vjp_fn
