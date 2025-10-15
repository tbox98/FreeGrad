import torch
import torch.nn as nn

import freegrad
from freegrad.hook import attach_param_hooks


def test_param_hooks_apply_only_in_params_scope():
    model = nn.Linear(3, 1, bias=True)
    attach_param_hooks(model)

    x = torch.randn(4, 3)
    y = model(x).sum()

    y.backward()
    grads_no_ctx = [p.grad.clone() for p in model.parameters()]
    assert all(g is not None and torch.any(g != 0) for g in grads_no_ctx)
    model.zero_grad()

    with freegrad.use("scale", params={"s": 0.0}, scope="params"):
        (model(x).sum()).backward()
    for p in model.parameters():
        assert torch.allclose(p.grad, torch.zeros_like(p.grad))
    model.zero_grad()

    with freegrad.use("scale", params={"s": 0.0}, scope="activations"):
        (model(x).sum()).backward()
    for p in model.parameters():
        assert torch.any(p.grad != 0)
