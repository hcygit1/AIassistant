from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator


@dataclass
class RuntimeSecurityContext:
    user_message: str = ""
    recent_untrusted_content: bool = False


_security_context: ContextVar[RuntimeSecurityContext] = ContextVar(
    "runtime_security_context",
    default=RuntimeSecurityContext(),
)


def get_runtime_security_context() -> RuntimeSecurityContext:
    return _security_context.get()


def mark_recent_untrusted_content(value: bool = True) -> None:
    ctx = get_runtime_security_context()
    ctx.recent_untrusted_content = value
    _security_context.set(ctx)


@contextmanager
def runtime_security_context(user_message: str, *, recent_untrusted_content: bool = False) -> Iterator[None]:
    token = _security_context.set(
        RuntimeSecurityContext(
            user_message=user_message,
            recent_untrusted_content=recent_untrusted_content,
        )
    )
    try:
        yield
    finally:
        _security_context.reset(token)
