import freegrad


def test_get_accepts_callable():
    def myrule(ctx, g, x, **_):
        return g

    fn = freegrad.get(myrule)
    assert fn is myrule
