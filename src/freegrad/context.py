# `freegrad/context.py`

import contextvars
from typing import Any, Dict, Iterable, Optional, Union

from .registry import get

_current_rule = contextvars.ContextVar("freegrad_rule", default=None)
_current_params = contextvars.ContextVar("freegrad_params", default={})
_current_scope = contextvars.ContextVar("freegrad_scope", default=("all",))

ScopeLike = Union[str, Iterable[str]]


class use:
    """Context manager per applicare una regola di gradiente.
    - rule: nome o callable registrata
    - params: dict di parametri passati alla regola
    - scope: "all", "activations", "params" oppure una tupla di questi
    """

    def __init__(
        self, rule, params: Optional[Dict[str, Any]] = None, scope: ScopeLike = "all"
    ):
        self.rule = rule
        self.params = params or {}
        if isinstance(scope, str):
            scope = (scope,)
        self.scope = tuple(scope)
        self._tok_rule = self._tok_params = self._tok_scope = None

    def __enter__(self):
        self._tok_rule = _current_rule.set(get(self.rule))
        self._tok_params = _current_params.set(self.params)
        self._tok_scope = _current_scope.set(self.scope)
        return self

    def __exit__(self, exc_type, exc, tb):
        _current_rule.reset(self._tok_rule)
        _current_params.reset(self._tok_params)
        _current_scope.reset(self._tok_scope)


# Helpers interni per leggere il contesto


def _ctx_get():
    return _current_rule.get(), _current_params.get(), _current_scope.get()
