import torch

import freegrad


def test_full_jam_with_none_input_shapes_and_range():
    fn = freegrad.get("full_jam")
    g = torch.ones(5)
    out = fn(None, g, None)
    assert out.shape == g.shape
    assert torch.all(out >= 0) and torch.all(out <= 1)


def test_positive_jam_with_none_input_behaves_like_full():
    fn = freegrad.get("positive_jam")
    g = torch.ones(8)
    out = fn(None, g, None)
    assert out.shape == g.shape
    assert torch.all(out >= 0) and torch.all(out <= 1)


def test_rectangular_jam_with_none_input_behaves_like_full():
    fn = freegrad.get("rectangular_jam")
    g = torch.ones(4)
    out = fn(None, g, None, a=-5.0, b=5.0)
    assert out.shape == g.shape
    assert torch.all(out >= 0) and torch.all(out <= 1)
