"""Restart recovery state machine for persisted subagent runs."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from subagents.subagent_run_model import SubagentRunRecord

logger = logging.getLogger(__name__)

MAX_ANNOUNCE_RETRY = 3
ANNOUNCE_EXPIRY_SEC = 5 * 60
INTERRUPTED_DELIVERY_STATES = {"queued", "delivering", "retrying"}
TERMINAL_DELIVERY_STATES = {"delivered", "dropped"}

OrphanReason = Literal[
    "missing-session-entry",
    "missing-session-id",
    "missing-session-file",
]
VALIDATION_UNAVAILABLE = "validation-unavailable"
ValidationUnavailable = Literal["validation-unavailable"]
ChildSessionValidation = OrphanReason | ValidationUnavailable | None
ResolveOrphanReason = Callable[[SubagentRunRecord], ChildSessionValidation]
DeliverAnnounce = Callable[[str, SubagentRunRecord], Awaitable[bool]]


def resolve_orphan_reason(
    entry: SubagentRunRecord,
    session_manager: Any,
) -> ChildSessionValidation:
    """Check whether a restored run still has a readable child session."""
    child_key = (entry.child_session_key or "").strip()
    if not child_key:
        return "missing-session-entry"
    try:
        parts = child_key.split(":")
        if len(parts) < 4:
            return "missing-session-entry"
        agent_id = parts[1]
        session_id = parts[3]
        index_entry = session_manager.get_session_index_entry(
            session_id,
            agent_id,
        )
        if index_entry is None:
            return "missing-session-entry"
        persisted_session_id = index_entry.get("sessionId")
        if not persisted_session_id or not str(persisted_session_id).strip():
            return "missing-session-id"
        if not session_manager.session_file_exists(
            str(persisted_session_id).strip(),
            agent_id,
        ):
            return "missing-session-file"
        return None
    except Exception as exc:
        logger.warning(
            "Subagent child session validation unavailable run=%s: %s",
            entry.run_id,
            exc,
        )
        return VALIDATION_UNAVAILABLE


def reconcile_orphaned(
    registry: Any,
    run_id: str,
    entry: SubagentRunRecord,
    reason: str,
) -> bool:
    registry.remove_run(run_id)
    logger.warning(
        "Subagent orphan pruned run=%s child=%s reason=%s",
        run_id,
        entry.child_session_key,
        reason,
    )
    return True


def reset_interrupted_delivery(
    registry: Any,
    run_id: str,
    entry: SubagentRunRecord,
) -> None:
    state = getattr(entry, "result_delivery_state", "pending")
    if state in INTERRUPTED_DELIVERY_STATES:
        registry.set_result_delivery_state(run_id, "pending")


class SubagentRunResumeService:
    """Reconcile persisted subagent runs after process restart."""

    def __init__(
        self,
        *,
        registry: Any,
        resolve_orphan_reason: ResolveOrphanReason,
        deliver_announce: DeliverAnnounce,
        now: Callable[[], float],
        max_announce_retry: int = MAX_ANNOUNCE_RETRY,
        announce_expiry_sec: float = ANNOUNCE_EXPIRY_SEC,
    ) -> None:
        self._registry = registry
        self._resolve_orphan_reason = resolve_orphan_reason
        self._deliver_announce = deliver_announce
        self._now = now
        self._max_announce_retry = max_announce_retry
        self._announce_expiry_sec = announce_expiry_sec

    async def resume(self) -> None:
        for run_id, entry in self._registry.list_run_entries():
            reason = self._resolve_orphan_reason(entry)
            if reason == VALIDATION_UNAVAILABLE:
                continue
            if reason:
                reconcile_orphaned(self._registry, run_id, entry, reason)
                continue

            delivery_state = getattr(
                entry,
                "result_delivery_state",
                "pending",
            )
            if delivery_state in TERMINAL_DELIVERY_STATES:
                self._registry.remove_run(run_id)
                continue

            if entry.announce_retry_count >= self._max_announce_retry:
                self._registry.remove_run(run_id)
                continue

            if (
                entry.ended_at is not None
                and self._now() - entry.ended_at > self._announce_expiry_sec
            ):
                self._registry.remove_run(run_id)
                continue

            if entry.ended_at is not None:
                reset_interrupted_delivery(self._registry, run_id, entry)
                try:
                    queued = await self._deliver_announce(run_id, entry)
                except Exception as exc:
                    logger.warning("Resume announce error run=%s: %s", run_id, exc)
                    queued = False
                if (
                    not queued
                    and not self._registry.mark_announce_retry(run_id)
                ):
                    self._registry.mark_result_delivery_dropped(run_id)
                    self._registry.remove_run(run_id)
                continue

            self._registry.mark_terminated(run_id, "restart-interrupted")
            interrupted = self._registry.get_run(run_id)
            if interrupted is None:
                continue
            reset_interrupted_delivery(self._registry, run_id, interrupted)
            try:
                queued = await self._deliver_announce(run_id, interrupted)
            except Exception:
                queued = False
            if (
                not queued
                and not self._registry.mark_announce_retry(run_id)
            ):
                self._registry.mark_result_delivery_dropped(run_id)
                self._registry.remove_run(run_id)
