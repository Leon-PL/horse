"""Private working state for one feature execution, including nested scopes."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass
class FeatureContext:
    counts_state: dict | None
    timelines: dict = field(default_factory=dict)


_current: ContextVar[FeatureContext | None] = ContextVar("feature_context", default=None)


def current_context():
    return _current.get()


def current_counts_state():
    context = current_context()
    return context.counts_state if context is not None else None


@contextmanager
def counts_state_scope(state):
    """Own the caller's working state and a private cache until the scope exits."""
    context = FeatureContext(state)
    token = _current.set(context)
    try:
        yield context
    finally:
        _current.reset(token)
