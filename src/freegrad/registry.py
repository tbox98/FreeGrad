from typing import Any, Callable, Dict, Optional, Union

import torch

_RULES: Dict[str, Callable] = {}


class RuleError(KeyError):
    """Custom error for gradient rule registration or retrieval failures.

    This exception is raised if a rule is registered with a name that
    already exists, or if `get()` is called with a name that is
    not in the registry.
    """

    pass


def register(name: str):
    """Decorator to register a gradient rule.

    Expected signature of the function:
        fn(ctx, grad_out, tin, **params) -> grad_in
    """

    def deco(fn: Callable):
        if name in _RULES:
            raise RuleError(f"Rule already registered: {name}")
        _RULES[name] = fn
        return fn

    return deco


def get(rule: Union[str, Callable]) -> Callable:
    """Retrieve a gradient rule by name or return the callable directly."""
    if callable(rule):
        return rule
    try:
        return _RULES[rule]
    except KeyError as e:
        raise RuleError(f"Rule not found: {rule}") from e


def compose(*rules: Union[str, Callable]) -> Callable[..., torch.Tensor]:
    """Compose multiple gradient rules in series.

    Example:
        compose("clip_norm", "noise")
        applies clip_norm first, then noise to the gradient.
    """
    fns = [get(r) for r in rules]

    def composed(
        ctx: Any,
        grad_out: torch.Tensor,
        tin: Optional[torch.Tensor],
        **params: Any,
    ) -> torch.Tensor:
        g = grad_out
        for fn in fns:
            g = fn(ctx, g, tin, **params)
        return g

    return composed
