import contextvars
from contextvars import Token
from types import TracebackType
from typing import Any, Callable, Dict, Iterable, Literal, Optional, Tuple, Type, Union

from .registry import get

_current_rule: contextvars.ContextVar[Optional[Callable]] = contextvars.ContextVar(
    "freegrad_rule", default=None
)
_current_params: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "freegrad_params", default={}
)
_current_scope: contextvars.ContextVar[Tuple[str, ...]] = contextvars.ContextVar(
    "freegrad_scope", default=("all",)
)

ScopeLike = Union[str, Iterable[str]]


class use:
    """Context manager to apply a custom gradient rule.

    Args:
        rule (Union[str, Callable]): The name of a registered rule or a
            callable.
        params (Optional[Dict[str, Any]], optional): A dict of parameters
            to pass to the rule's `**params`. Defaults to None.
        scope (ScopeLike, optional): One of "all", "activations", "params",
            or a tuple of these to specify where the rule applies.
            Defaults to "all".
    """

    def __init__(
        self,
        rule: Union[str, Callable],
        params: Optional[Dict[str, Any]] = None,
        scope: ScopeLike = "all",
    ):
        self.rule: Union[str, Callable] = rule
        self.params: Dict[str, Any] = params or {}
        if isinstance(scope, str):
            scope = (scope,)
        self.scope = tuple(scope)
        self._tok_rule: Optional[Token] = None
        self._tok_params: Optional[Token] = None
        self._tok_scope: Optional[Token] = None

    def __enter__(self) -> "use":
        self._tok_rule = _current_rule.set(get(self.rule))
        self._tok_params = _current_params.set(self.params)
        self._tok_scope = _current_scope.set(self.scope)
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> Literal[False]:
        if self._tok_rule:
            _current_rule.reset(self._tok_rule)
        if self._tok_params:
            _current_params.reset(self._tok_params)
        if self._tok_scope:
            _current_scope.reset(self._tok_scope)
        return False  # Return False to not suppress exceptions


# Internal helper used to read the context
def _ctx_get() -> Tuple[Optional[Callable], Dict[str, Any], Tuple[str, ...]]:
    return _current_rule.get(), _current_params.get(), _current_scope.get()
