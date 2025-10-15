import torch

import freegrad


def test_rectangular_mask():
    rect = freegrad.get("rectangular")
    x = torch.tensor([-1.0, -0.25, 0.0, 0.25, 1.0], requires_grad=True)
    g = torch.ones_like(x)
    out = rect(None, g, x, a=-0.5, b=0.5)
    expected = torch.tensor([0, 1, 1, 1, 0], dtype=g.dtype)
    assert torch.allclose(out, expected)


def test_triangular_tapers_to_zero():
    tri = freegrad.get("triangular")
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    g = torch.ones_like(x)
    out = tri(None, g, x, width=1.0)
    expected = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
    assert torch.allclose(out, expected, atol=1e-6)


def test_clip_norm_scales_when_exceeding_max():
    clip = freegrad.get("clip_norm")
    g = torch.tensor([3.0, 4.0])
    out = clip(None, g, None, max_norm=1.0)
    assert torch.allclose(out, g / 5.0, atol=1e-6)


def test_noise_shape_and_dtype_are_preserved():
    torch.manual_seed(0)
    noise = freegrad.get("noise")
    g = torch.ones(5)
    out = noise(None, g, None, sigma=0.5)
    assert out.shape == g.shape and out.dtype == g.dtype
    assert not torch.allclose(out, g)


def test_rectangular_jam_zeros_outside_interval():
    rjam = freegrad.get("rectangular_jam")
    x = torch.tensor([-10.0, -6.0, -5.0, 0.0, 5.0, 6.0, 10.0])
    g = torch.ones_like(x)
    out = rjam(None, g, x, a=-5.0, b=5.0)
    assert out[0].item() == 0.0 and out[1].item() == 0.0 and out[-1].item() == 0.0
    inside = out[2:-1]
    assert torch.all(inside >= 0.0) and torch.all(inside <= 1.0)
