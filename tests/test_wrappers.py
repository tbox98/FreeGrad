import torch

import freegrad as xg
from freegrad.wrappers import Activation


def test_activation_forward_relu():
    act = Activation("ReLU")
    x = torch.tensor([-1.0, 0.0, 2.5], requires_grad=True)
    y = act(x)
    assert torch.allclose(y, torch.tensor([0.0, 0.0, 2.5]))


def test_activation_backward_respects_scope():
    # This rule is registered locally just for this test
    @xg.register("test_scale_half")
    def scale_half(ctx, g, x, **_):
        return 0.5 * g

    x = torch.tensor([3.0], requires_grad=True)
    act = Activation("Linear")

    # 1. Test standard autograd (no context)
    y = act(x).sum()
    y.backward()
    assert x.grad.item() == 1.0
    x.grad = None

    # 2. Test freegrad autograd (inside context)
    with xg.use(rule="test_scale_half", scope="activations"):
        y2 = act(x).sum()
        y2.backward()
    # The gradient should be scaled by the rule
    assert x.grad.item() == 0.5


def test_helper_activations():
    # 1. Cover _relu6
    act = Activation("ReLU6")
    x = torch.tensor([-2.0, 4.0, 8.0])
    y = act(x)
    # ReLU6: clamp(x, 0, 6) -> [0, 4, 6]
    assert torch.allclose(y, torch.tensor([0.0, 4.0, 6.0]))

    # 2. Cover _heaviside
    act = Activation("Heaviside")
    x = torch.tensor([-1.0, 0.0, 1.0])
    y = act(x)
    # Heaviside(x, 0.0) -> 0 if x < 0, 1 if x > 0, 0 if x == 0
    assert torch.allclose(y, torch.tensor([0.0, 0.0, 1.0]))

    # 3. Cover _leaky_relu
    act = Activation("LeakyReLU")
    x = torch.tensor([-10.0, 10.0])
    y = act(x)
    # LeakyReLU(x, slope=0.01) -> [-0.1, 10.0]
    assert torch.allclose(y, torch.tensor([-0.1, 10.0]))

    # 4. Cover _elu
    act = Activation("ELU")
    x = torch.tensor([-1.0, 1.0])
    y = act(x)
    # ELU(x, alpha=1.0) -> [exp(x)-1, x] for x<0
    expected_neg = torch.exp(torch.tensor(-1.0)) - 1.0
    assert torch.allclose(y, torch.tensor([expected_neg, 1.0]))


def test_backward_rule_persists_outside_context():
    """
    Tests that the gradient rule "snapshotted" during the .forward() pass
    is correctly applied during the .backward() pass, even if .backward()
    is called *outside* the original context.

    This confirms the fix for the CUDA/threading bug, where the context
    would be lost in separate autograd threads.
    """

    # Define a mock rule to confirm that if the rule *were* applied,
    # the gradient would be different (e.g., scaled by 100).
    @xg.register("mock_persistence_rule")
    def mock_rule(ctx, grad_out, x, **kwargs):
        return grad_out * 100.0

    x = torch.tensor([2.0], requires_grad=True)
    act = Activation("Linear")

    # 1. Forward Pass INSIDE context
    # This satisfies the condition in Activation.forward to use _FreeGradActivationFn
    # and snapshots the "mock_persistence_rule".
    with xg.use(rule="mock_persistence_rule", scope="activations"):
        y = act(x)

    # 2. Backward Pass OUTSIDE context
    # The context is now exited. In the *new* (fixed) logic, this doesn't matter.
    # _FreeGradActivationFn.backward will use its saved 'ctx.fg_state'.
    y.backward()

    # 3. Verification
    # The snapshotted rule (grad * 100.0) should have been applied.
    # The standard autograd gradient (1.0) should be ignored.
    assert x.grad.item() == 100.0
