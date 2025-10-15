## `freegrad/functional.py`

from typing import Callable, Tuple
import torch
from .context import use


@torch.no_grad()
def jvp(
    f: Callable, x: torch.Tensor, v: torch.Tensor, *, rule=None, params=None
) -> torch.Tensor:
    """Jacobian-vector product alternativo: J_f(x) @ v.
    Implementazione naive via differenze simmetriche per indipendenza dal backward classico.
    Per casi seri, preferisci autograd JVP quando disponibile.
    """
    eps = 1e-3
    with use(rule=rule, params=params or {}, scope=("all",)) if rule else nullcontext():
        y_pos = f(x + eps * v)
        y_neg = f(x - eps * v)
    return (y_pos - y_neg) / (2 * eps)


class nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def vjp(
    f: Callable, x: torch.Tensor, *, rule=None, params=None
) -> Tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
    """Vector-Jacobian product alternativo via contesto freegrad.
    Restituisce (y, vjp_fn) dove vjp_fn(v) = v^T J_f(x).
    """
    with use(rule=rule, params=params or {}, scope=("all",)) if rule else nullcontext():
        x = x.detach().requires_grad_(True)
        y = f(x)

    def vjp_fn(v: torch.Tensor) -> torch.Tensor:
        (g,) = torch.autograd.grad(y, x, v, retain_graph=True, allow_unused=False)
        return g

    return y, vjp_fn
