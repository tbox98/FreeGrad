import torch

import freegrad as xg
from freegrad.wrappers import Activation


def test_activation_forward_relu():
    act = Activation("ReLU")
    x = torch.tensor([-1.0, 0.0, 2.5], requires_grad=True)
    y = act(x)
    assert torch.allclose(y, torch.tensor([0.0, 0.0, 2.5]))


def test_activation_backward_respects_scope():
    @xg.register("scale_half")
    def scale_half(ctx, g, x, **_):
        return 0.5 * g

    x = torch.tensor([3.0], requires_grad=True)
    act = Activation("Linear")

    y = act(x).sum()
    y.backward()
    assert x.grad.item() == 1.0
    x.grad = None

    with xg.use(rule="scale_half", scope="activations"):
        y2 = act(x).sum()
        y2.backward()
    assert x.grad.item() == 0.5
