import torch

from freegrad.functional import vjp


def test_vjp_matches_autograd_for_square_sum():
    def f(x):
        return (x**2).sum()

    x = torch.randn(5, requires_grad=False)
    y, vjp_fn = vjp(f, x)
    v = torch.ones(())
    g = vjp_fn(v)
    assert torch.allclose(g, 2 * x, atol=1e-5)


def test_jvp_finite_difference_for_square():
    import torch

    from freegrad.functional import jvp

    torch.manual_seed(0)
    x = torch.randn(5, dtype=torch.float64)
    v = torch.randn(5, dtype=torch.float64)

    def f(z):
        return (z**2).sum()

    jvp_fd = jvp(f, x, v)  # inherits dtype from x,v
    exact = (2 * x * v).sum()
    assert torch.allclose(jvp_fd, exact, atol=1e-3)
