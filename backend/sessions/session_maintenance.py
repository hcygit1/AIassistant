"""Retention and disk-budget policies for persisted sessions."""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable

from sessions.session_repository import SessionRepository


logger = logging.getLogger(__name__)


class SessionMaintenanceService:
    def __init__(
        self,
        *,
        repository: SessionRepository,
        get_config: Callable[[], dict[str, Any]],
        now_ms: Callable[[], int] | None = None,
    ) -> None:
        self._repository = repository
        self._get_config = get_config
        self._now_ms = now_ms or (lambda: int(time.time() * 1000))

    def run(
        self,
        agent_id: str,
        *,
        store: dict[str, dict[str, Any]] | None = None,
        enforce: bool = False,
        dry_run: bool = False,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        with self._repository.get_agent_lock(agent_id):
            current = (
                self._repository.load_index(agent_id)
                if store is None
                else store
            )
            maintenance = self._maintenance_config()
            if (
                not enforce
                and not dry_run
                and maintenance.get("mode", "warn") == "warn"
            ):
                disk = self._enforce_disk_budget(
                    agent_id,
                    current,
                    dry_run=True,
                )
                return current, {
                    "pruned": 0,
                    "capped": 0,
                    "diskBudget": disk,
                }

            main_key = f"agent:{agent_id}:main"
            cutoff_ms = self._now_ms() - self._parse_duration_ms(
                maintenance.get("pruneAfter", "30d")
            )
            expired = {
                key
                for key, entry in current.items()
                if key != main_key
                and isinstance(entry.get("updatedAt"), (int, float))
                and entry["updatedAt"] < cutoff_ms
            }
            max_entries = max(1, int(maintenance.get("maxEntries", 500)))
            newest_non_main = sorted(
                (key for key in current if key != main_key),
                key=lambda key: current[key].get("updatedAt") or 0,
                reverse=True,
            )
            allowed_non_main = max_entries - (1 if main_key in current else 0)
            capped_keys = set(newest_non_main[max(0, allowed_non_main):])
            to_remove = expired | capped_keys

            if not dry_run:
                for key in to_remove:
                    self._archive_and_remove(agent_id, current.get(key, {}))
                    current.pop(key, None)
                if to_remove:
                    self._repository.save_index(agent_id, current)

            disk = self._enforce_disk_budget(
                agent_id,
                current,
                dry_run=dry_run,
            )
            return current, {
                "pruned": len(expired),
                "capped": len(to_remove - expired),
                "diskBudget": disk,
            }

    def _maintenance_config(self) -> dict[str, Any]:
        config = self._get_config() or {}
        return (config.get("session") or {}).get("maintenance") or {}

    @staticmethod
    def _parse_duration_ms(raw: Any) -> int:
        match = re.match(r"^(\d+)\s*(d|h|m|s)?$", str(raw).strip().lower())
        if not match:
            return 30 * 24 * 3600 * 1000
        amount = int(match.group(1))
        units = {
            "d": 24 * 3600 * 1000,
            "h": 3600 * 1000,
            "m": 60 * 1000,
            "s": 1000,
        }
        return amount * units[match.group(2) or "d"]

    @staticmethod
    def _parse_byte_size(raw: Any) -> int | None:
        if raw is None or str(raw).strip() == "":
            return None
        match = re.match(
            r"^(\d+(?:\.\d+)?)\s*(b|kb|k|mb|m|gb|g)?$",
            str(raw).strip().lower(),
        )
        if not match:
            return None
        units = {
            "b": 1,
            "kb": 1024,
            "k": 1024,
            "mb": 1024**2,
            "m": 1024**2,
            "gb": 1024**3,
            "g": 1024**3,
        }
        return int(float(match.group(1)) * units[match.group(2) or "b"])

    def _disk_limits(self) -> tuple[int | None, int | None]:
        maintenance = self._maintenance_config()
        maximum = self._parse_byte_size(maintenance.get("maxDiskBytes"))
        if maximum is None or maximum <= 0:
            return None, None
        high = self._parse_byte_size(maintenance.get("highWaterBytes"))
        if high is None:
            high = max(1, int(maximum * 0.8))
        return maximum, min(high, maximum)

    def _enforce_disk_budget(
        self,
        agent_id: str,
        store: dict[str, dict[str, Any]],
        *,
        dry_run: bool,
    ) -> dict[str, Any] | None:
        maximum, high = self._disk_limits()
        if maximum is None or high is None:
            return None
        total = self._repository.directory_size(agent_id)
        before = total
        removed_files = 0
        removed_entries = 0
        freed = 0
        if total > maximum and not dry_run:
            total, archive_count, archive_freed = (
                self._repository.prune_oldest_archives(
                    agent_id,
                    current_total=total,
                    target_total=high,
                )
            )
            removed_files += archive_count
            freed += archive_freed

            main_key = f"agent:{agent_id}:main"
            oldest = sorted(
                (key for key in store if key != main_key),
                key=lambda key: store[key].get("updatedAt") or 0,
            )
            for key in oldest:
                if total <= high:
                    break
                entry = store.get(key, {})
                session_id = str(entry.get("sessionId") or "").strip()
                if not session_id:
                    continue
                try:
                    size = self._repository.session_file_size(
                        session_id,
                        agent_id,
                    )
                    removed = self._repository.delete_session_file(
                        session_id,
                        agent_id,
                    )
                except OSError:
                    continue
                store.pop(key, None)
                removed_entries += 1
                if removed:
                    removed_files += 1
                total -= size
                freed += size
                self._cleanup_runtime(agent_id, session_id)
            if removed_entries:
                self._repository.save_index(agent_id, store)

        return {
            "totalBytesBefore": before,
            "totalBytesAfter": total,
            "removedFiles": removed_files,
            "removedEntries": removed_entries,
            "freedBytes": freed,
            "maxBytes": maximum,
            "highWaterBytes": high,
            "overBudget": total > high,
        }

    def _archive_and_remove(
        self,
        agent_id: str,
        entry: dict[str, Any],
    ) -> None:
        session_id = str(entry.get("sessionId") or "").strip()
        if not session_id:
            return
        archive = self._repository.archive_path(
            agent_id,
            f"{session_id}.deleted.{self._now_ms() // 1000}.json",
        )
        try:
            self._repository.archive_session_file(
                session_id,
                agent_id,
                archive,
            )
        except OSError:
            self._repository.delete_session_file(session_id, agent_id)
        self._cleanup_runtime(agent_id, session_id)

    @staticmethod
    def _cleanup_runtime(agent_id: str, session_id: str) -> None:
        try:
            from sessions.session_lock_manager import cleanup_session_runtime

            cleanup_session_runtime(agent_id, session_id)
        except Exception as exc:
            logger.warning("cleanup_session_runtime after session prune: %s", exc)
