"""Requester lookup port used by session persistence compatibility paths."""

from __future__ import annotations

from typing import Callable


RequesterResolution = tuple[str, str] | None
RequesterProvider = Callable[[str], RequesterResolution]


class SessionRequesterResolver:
    def __init__(
        self,
        provider: RequesterProvider | None = None,
    ) -> None:
        self._provider = provider

    def bind(self, provider: RequesterProvider) -> None:
        self._provider = provider

    def __call__(self, child_session_key: str) -> RequesterResolution:
        provider = self._provider
        if provider is None:
            return None
        try:
            return provider(child_session_key)
        except Exception:
            return None


session_requester_resolver = SessionRequesterResolver()
