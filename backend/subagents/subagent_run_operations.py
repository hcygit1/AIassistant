"""Scoped list and termination operations for subagent runs."""

from __future__ import annotations

from typing import Any

from subagents.subagent_service_models import (
    KillResult,
    SubagentListResult,
    SubagentServiceError,
)


class SubagentRunOperations:
    def __init__(self, *, registry: Any, scope: Any) -> None:
        self._registry = registry
        self._scope = scope

    def list_runs(
        self,
        *,
        requester_agent_id: str,
        requester_session_id: str | None,
        recent_minutes: int | None = None,
        recursive: bool = False,
    ) -> SubagentListResult:
        requester_key = self._scope.requester_key(
            requester_agent_id,
            requester_session_id,
        )
        minutes = self._scope.recent_minutes(recent_minutes)
        if recursive:
            records = self._registry.list_descendant_runs(
                requester_key,
                include_recent_minutes=minutes,
            )
        else:
            records = self._registry.list_runs_for_requester(
                requester_key,
                include_recent_minutes=minutes,
            )
        return SubagentListResult(
            records=records,
            recent_minutes=minutes,
            requester_key=requester_key,
        )

    def kill(
        self,
        *,
        requester_agent_id: str,
        requester_session_id: str,
        target: str,
    ) -> KillResult:
        target = (target or "").strip()
        if not target:
            raise SubagentServiceError(
                "missing_target",
                "missing target",
            )
        requester_key = self._scope.requester_key(
            requester_agent_id,
            requester_session_id,
        )
        records = self._registry.list_descendant_runs(
            requester_key,
            include_recent_minutes=self._scope.MAX_RECENT_MINUTES,
        )

        if target in ("all", "*"):
            killed = 0
            for record in records:
                if (
                    record.ended_at is None
                    and self._registry.kill(record.run_id)
                ):
                    killed += 1
            return KillResult(
                killed=killed,
                scope=requester_key,
            )

        allowed = {record.run_id for record in records}
        if target not in allowed:
            raise SubagentServiceError(
                "out_of_scope",
                "run not found in current session scope",
            )
        record = self._registry.get_run(target)
        if record is None:
            raise SubagentServiceError(
                "not_found",
                "run not found",
            )
        if record.ended_at is not None:
            raise SubagentServiceError(
                "already_ended",
                "run already finished",
            )
        if not self._registry.kill(target):
            raise SubagentServiceError(
                "kill_failed",
                "failed to kill run",
            )
        return KillResult(
            killed=1,
            scope=requester_key,
            run_id=target,
        )
