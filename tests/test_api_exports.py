def test_public_api_exports():
    import freegrad

    assert hasattr(freegrad, "register")
    assert hasattr(freegrad, "get")
    assert hasattr(freegrad, "compose")
    assert hasattr(freegrad, "use")
    assert hasattr(freegrad, "Activation")
    assert hasattr(freegrad, "transforms")
