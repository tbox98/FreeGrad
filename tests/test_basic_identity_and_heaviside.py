import torch

import freegrad


def test_identity_rule_passthrough():
    fn = freegrad.get("d(Linear)")
    g = torch.tensor([1.0, -2.0, 3.0])
    out = fn(None, g, None)
    assert torch.allclose(out, g)


def test_heaviside_masking():
    fn = freegrad.get("d(ReLU)")
    x = torch.tensor([-1.0, 0.0, 2.0])
    g = torch.ones_like(x)
    out = fn(None, g, x)
    # grad passes only where x>0
    assert out.tolist() == [0.0, 0.0, 1.0]
