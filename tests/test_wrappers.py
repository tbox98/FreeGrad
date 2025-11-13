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


def test_coverage_backward_fallback():
    """
    Covers the fallback block in _FreeGradActivationFn.backward.
    This block is reached when a rule is active during .forward() (forcing the
    usage of the custom Function) but inactive during .backward() (forcing
    a fallback to standard autograd).
    """

    # Define a mock rule to confirm that if the rule *were* applied,
    # the gradient would be different (e.g., scaled by 100).
    @xg.register("mock_coverage_rule")
    def mock_rule(ctx, grad_out, x, **kwargs):
        return grad_out * 100.0

    x = torch.tensor([2.0], requires_grad=True)
    act = Activation("Linear")

    # 1. Forward Pass INSIDE context
    # This satisfies the condition in Activation.forward to use _FreeGradActivationFn
    with xg.use(rule="mock_coverage_rule", scope="activations"):
        y = act(x)

    # 2. Backward Pass OUTSIDE context
    # The context is now exited. _ctx_get() will return None/default.
    # Inside _FreeGradActivationFn.backward, the check for rule/scope will fail,
    # triggering the fallback block that computes gradients via standard autograd.
    y.backward()

    # 3. Verification
    # If the rule was active, grad would be 100.0.
    # Since fallback was used on "Linear", grad should be 1.0.
    assert x.grad.item() == 1.0
