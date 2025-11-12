import torch

from .context import _ctx_get


class _ParamGradHook:
    def __init__(self):
        pass

    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        # Directly transform the parameter gradient if the context requires it
        rule, params, scope = _ctx_get()
        if (rule is None) or ("params" not in scope and "all" not in scope):
            return grad
        # For parameters we don't have an activation input; we use None
        return rule(None, grad, None, **params)


def attach_param_hooks(model: torch.nn.Module) -> None:
    """Attaches the gradient hook to all parameters of a model.

    This iterates over all `model.parameters()` and registers a hook
    that will apply the gradient rule active in the context, provided
    the scope includes "params" or "all".

    Warning:
        Use with caution. In most cases, modifying activation gradients
        (scope="activations") is sufficient and more common. This function
        is for rules that must run on `nn.Parameter` gradients directly.

    Args:
        model (torch.nn.Module): The model to attach hooks to.
    """
    hook = _ParamGradHook()
    for p in model.parameters():
        p.register_hook(hook)
