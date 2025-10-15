import importlib
import sys
from importlib import metadata as importlib_metadata


def test_version_fallback(monkeypatch):
    # Simulate PackageNotFoundError so __version__ falls back to 0.0.0
    monkeypatch.setattr(
        importlib_metadata,
        "version",
        lambda name: (_ for _ in ()).throw(importlib_metadata.PackageNotFoundError()),
    )
    # Remove already-imported module to force re-exec of __init__.py
    sys.modules.pop("freegrad", None)
    import freegrad  # re-import after monkeypatch

    assert hasattr(freegrad, "__version__")
    assert freegrad.__version__ == "0.0.0"
    # clean up to avoid side effects on other tests
    sys.modules.pop("freegrad", None)
    importlib.invalidate_caches()
