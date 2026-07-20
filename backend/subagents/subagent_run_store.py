"""Thread-safe runtime storage and persistence boundary for subagent runs."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Callable, Iterator


class SubagentRunStore:
    def __init__(
        self,
        *,
        load_runs: Callable[[], dict[str, Any]] | None = None,
        save_runs: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._records: dict[str, Any] = {}
        self._lock = threading.RLock()
        self._load_runs = load_runs
        self._save_runs = save_runs

    @property
    def records(self) -> dict[str, Any]:
        return self._records

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    @contextmanager
    def locked_records(self) -> Iterator[dict[str, Any]]:
        with self._lock:
            yield self._records

    def restore(self) -> None:
        loaded = dict(self._load())
        with self._lock:
            self._records.clear()
            self._records.update(loaded)

    def persist(self) -> None:
        with self._lock:
            snapshot = dict(self._records)
            self._save(snapshot)

    def _load(self) -> dict[str, Any]:
        if self._load_runs is not None:
            return self._load_runs()
        from subagents.subagent_registry_state import (
            restore_registry_from_disk,
        )

        restored: dict[str, Any] = {}
        restore_registry_from_disk(restored, merge_only=False)
        return restored

    def _save(self, runs: dict[str, Any]) -> None:
        if self._save_runs is not None:
            self._save_runs(runs)
            return
        from subagents.subagent_registry_state import save_registry_to_disk

        save_registry_to_disk(runs)
