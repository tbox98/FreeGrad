"""
freegrad – alternative backward rules alongside PyTorch autograd.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("freegrad")
except PackageNotFoundError:  # local editable install
    __version__ = "0.0.0"

# Export the main public API of the package
try:
    from . import transforms  # noqa: F401
    from .context import use  # noqa: F401
    from .registry import compose, get, register  # noqa: F401
    from .wrappers import Activation  # noqa: F401
except Exception:
    # During bootstrap (e.g., initial setup) these modules may not yet be available
    pass
