import torch

import freegrad
from freegrad.wrappers import Activation


def test_context_applies_rule_to_activations_only():
    @freegrad.register("zero")
    def zero_rule(ctx, grad_out, tin, **_):
        return torch.zeros_like(grad_out)

    x = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
    act = Activation("Linear")

    y = act(x).sum()
    y.backward()
    assert torch.allclose(x.grad, torch.ones_like(x))
    x.grad = None

    with freegrad.use(rule="zero", scope="activations"):
        y2 = act(x).sum()
        y2.backward()
    assert torch.allclose(x.grad, torch.zeros_like(x))


def test_context_scope_excludes_activations():
    @freegrad.register("neg")
    def neg_rule(ctx, grad_out, tin, **_):
        return -grad_out

    x = torch.tensor([2.0], requires_grad=True)
    act = Activation("Linear")

    with freegrad.use(rule="neg", scope="params"):
        y = act(x).sum()
        y.backward()
    assert torch.allclose(x.grad, torch.ones_like(x))
