import torch

import freegrad


def test_heaviside_with_none_input_passthrough():
    fn = freegrad.get("d(ReLU)")
    g = torch.tensor([1.0, 2.0, 3.0])
    # When input is None (e.g., param scope), function must passthrough
    out = fn(None, g, None)
    assert torch.allclose(out, g)


def test_centralize_reduces_mean_to_zero_last_dim():
    fn = freegrad.get("centralize")
    g = torch.tensor([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])
    out = fn(None, g, None, dim=-1, keepdim=True)
    # Mean along last dim should be ~0 for each row
    assert torch.allclose(out.mean(dim=-1, keepdim=True), torch.zeros(2, 1), atol=1e-7)
