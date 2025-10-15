import pytest

import freegrad


def test_get_unknown_rule_raises():
    with pytest.raises(KeyError):
        freegrad.get("THIS_RULE_DOES_NOT_EXIST")
