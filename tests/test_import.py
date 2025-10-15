def test_imports():
    import freegrad

    assert hasattr(freegrad, "__version__")
    assert callable(getattr(freegrad, "use", None))
    assert callable(getattr(freegrad, "Activation", None))
