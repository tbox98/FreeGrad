import pytest

from freegrad.wrappers import Activation


def test_activation_unsupported_forward_raises():
    with pytest.raises(ValueError):
        Activation("DefinitelyNotAnActivation")
