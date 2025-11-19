import pytest
import torch

import freegrad


def test_register_and_get_rule():
    @freegrad.register("double_grad")
    def double_rule(ctx, grad_out, tin, **_):
        return 2 * grad_out

    fn = freegrad.get("double_grad")
    g = fn(None, torch.ones(3), None)
    assert torch.allclose(g, torch.full((3,), 2.0))


def test_register_duplicate_raises():
    @freegrad.register("unique_rule")
    def r1(ctx, g, x, **_):
        return g

    with pytest.raises(KeyError):

        @freegrad.register("unique_rule")
        def r2(ctx, g, x, **_):
            return g


def test_compose_applies_in_series():
    @freegrad.register("add_one")
    def add_one(ctx, g, x, **_):
        return g + 1

    @freegrad.register("times_three")
    def times_three(ctx, g, x, **_):
        return 3 * g

    composed = freegrad.compose("add_one", "times_three")
    out = composed(None, torch.tensor([0.0, 1.0]), None)
    assert torch.allclose(out, torch.tensor([3.0, 6.0]))
