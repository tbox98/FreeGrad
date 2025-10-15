import torch

from .context import _ctx_get


class _ParamGradHook:
    def __init__(self):
        pass

    def __call__(self, grad: torch.Tensor):
        # Directly transform the parameter gradient if the context requires it
        rule, params, scope = _ctx_get()
        if (rule is None) or ("params" not in scope and "all" not in scope):
            return grad
        # For parameters we don't have an activation input; we use None
        return rule(None, grad, None, **params)


def attach_param_hooks(model: torch.nn.Module) -> None:
    """Attach a hook to ALL parameters of the model.
    Use with caution: in most cases scope="activations" is sufficient.
    """
    hook = _ParamGradHook()
    for p in model.parameters():
        p.register_hook(hook)
