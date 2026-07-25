"""Recovery callback providers for restartable session work."""

from __future__ import annotations

import threading
from typing import Any, Callable, Iterable

from sessions.session_work_record import SessionWorkRecord

RecoveryCallbacks = dict[str, Any] | None
RecoveryCallbackProvider = Callable[[SessionWorkRecord], RecoveryCallbacks]


class SessionWorkRecoveryResolver:
    def __init__(self, *, required_kinds: Iterable[str] = ()) -> None:
        self._required_kinds = frozenset(
            self._normalize_kind(kind) for kind in required_kinds
        )
        self._providers: dict[str, RecoveryCallbackProvider] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _normalize_kind(kind: str) -> str:
        return kind.strip()

    def bind(self, kind: str, provider: RecoveryCallbackProvider) -> None:
        normalized_kind = self._normalize_kind(kind)
        if not normalized_kind:
            raise ValueError("work kind cannot be empty")
        with self._lock:
            self._providers[normalized_kind] = provider

    def resolve(self, record: SessionWorkRecord) -> RecoveryCallbacks:
        normalized_kind = self._normalize_kind(record.kind)
        with self._lock:
            provider = self._providers.get(normalized_kind)
        if provider is not None:
            return provider(record)
        if normalized_kind in self._required_kinds:
            return None
        return {}


session_work_recovery_resolver = SessionWorkRecoveryResolver(
    required_kinds={"cron"},
)
