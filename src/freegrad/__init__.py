"""
– alternative backward rules alongside PyTorch autograd.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("freegrad")
except PackageNotFoundError:  # local editable install or not installed as a dist
    __version__ = "0.0.0"

from . import transforms  # noqa: F401
from .context import use  # noqa: F401

# Public API
from .registry import compose, get, register  # noqa: F401
from .wrappers import Activation  # noqa: F401

__all__ = [
    "register",
    "get",
    "compose",
    "use",
    "Activation",
    "transforms",
    "__version__",
]
