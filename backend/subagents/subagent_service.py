"""子 Agent 的应用级编排服务。"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Callable

from sessions.session_identity import session_key_from_session_id
from subagents.subagent_run_model import (
    SubagentCapacityError,
    SubagentRunRecord,
)


class SubagentServiceError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class SpawnResult:
    record: SubagentRunRecord
    session_id: str


@dataclass(frozen=True, slots=True)
class SubagentListResult:
    records: list[SubagentRunRecord]
    recent_minutes: int
    requester_key: str


@dataclass(frozen=True, slots=True)
class KillResult:
    killed: int
    scope: str
    run_id: str | None = None


@dataclass(frozen=True, slots=True)
class SteerResult:
    record: SubagentRunRecord
    replaced_run_id: str
    label: str


RunnerFactory = Callable[[str], Any]


class SubagentService:
    MAX_STEER_MESSAGE_CHARS = 4000
    MAX_RECENT_MINUTES = 24 * 60

    def __init__(
        self,
        *,
        registry: Any = None,
        session_manager: Any = None,
        resolve_agent_config: Callable[[str], dict[str, Any]] | None = None,
        get_config: Callable[[], dict[str, Any]] | None = None,
        runner_factory: RunnerFactory | None = None,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        if registry is None:
            from subagents.subagent_registry import registry
        if session_manager is None:
            from sessions.session_manager import session_manager
        if resolve_agent_config is None:
            from config import resolve_agent_config
        if get_config is None:
            from config import get_config

        self._registry = registry
        self._session_manager = session_manager
        self._resolve_agent_config = resolve_agent_config
        self._get_config = get_config
        self._runner_factory = runner_factory
        self._id_factory = id_factory or (
            lambda: uuid.uuid4().hex[:12]
        )

    def spawn(
        self,
        *,
        requester_agent_id: str,
        requester_session_id: str,
        task: str,
        target_agent_id: str = "",
        label: str | None = None,
        model: str | None = None,
    ) -> SpawnResult:
        requester_agent_id = requester_agent_id or "main"
        target_agent_id = target_agent_id or requester_agent_id
        requester_key = self._requester_key(
            requester_agent_id,
            requester_session_id,
        )
        self._validate_target_allowed(
            requester_agent_id,
            target_agent_id,
        )

        requester_depth = self._registry.get_requester_depth(
            requester_key
        )
        child_depth = requester_depth + 1
        target_config = (
            self._resolve_agent_config(target_agent_id) or {}
        )
        subagent_config = target_config.get("subagents") or {}
        max_depth = self._nonnegative_int(
            subagent_config.get("max_spawn_depth"),
            default=1,
        )
        max_children = self._nonnegative_int(
            subagent_config.get("max_children_per_agent"),
            default=5,
        )
        if child_depth > max_depth:
            raise SubagentServiceError(
                "depth_limit",
                "Current depth limit reached "
                f"(maxSpawnDepth={max_depth})",
            )
        child_session_id = (
            f"subagent-{self._id_factory()}"
        )
        child_session_key = session_key_from_session_id(
            target_agent_id,
            child_session_id,
        )
        run_id = self._id_factory()
        try:
            record = self._registry.register_run(
                run_id=run_id,
                child_session_key=child_session_key,
                requester_session_key=requester_key,
                requester_agent_id=requester_agent_id,
                target_agent_id=target_agent_id,
                task=task,
                label=label,
                model=model,
                spawn_depth=child_depth,
                max_active_for_requester=max_children,
            )
        except SubagentCapacityError as exc:
            raise SubagentServiceError(
                "children_limit",
                "Active sub-agents limit reached "
                f"({max_children})",
            ) from exc

        try:
            self._session_manager.ensure_session(
                child_session_id,
                target_agent_id,
                spawned_by=requester_key,
                label=(
                    label or task[:60] or "Sub-agent task"
                ),
            )
            self._start_runner(
                requester_agent_id=requester_agent_id,
                run_id=run_id,
                session_id=child_session_id,
                agent_id=target_agent_id,
                task=task,
                requester_key=requester_key,
                run_timeout_seconds=self._run_timeout(
                    subagent_config
                ),
            )
        except Exception as exc:
            self._registry.mark_terminated(
                run_id,
                f"start error: {exc}",
            )
            raise SubagentServiceError(
                "start_failed",
                f"Failed to start sub-agent: {exc}",
            ) from exc

        current = self._registry.get_run(run_id) or record
        return SpawnResult(
            record=current,
            session_id=child_session_id,
        )

    def list_runs(
        self,
        *,
        requester_agent_id: str,
        requester_session_id: str | None,
        recent_minutes: int | None = None,
        recursive: bool = False,
    ) -> SubagentListResult:
        requester_key = self._requester_key(
            requester_agent_id,
            requester_session_id,
        )
        minutes = self._recent_minutes(recent_minutes)
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
        requester_key = self._requester_key(
            requester_agent_id,
            requester_session_id,
        )
        records = self._registry.list_descendant_runs(
            requester_key,
            include_recent_minutes=self.MAX_RECENT_MINUTES,
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

    def steer(
        self,
        *,
        requester_agent_id: str,
        requester_session_id: str | None,
        run_id: str,
        message: str,
    ) -> SteerResult:
        run_id = (run_id or "").strip()
        message = (message or "").strip()
        if not run_id or not message:
            raise SubagentServiceError(
                "missing_input",
                "run_id and message are required",
            )
        if len(message) > self.MAX_STEER_MESSAGE_CHARS:
            raise SubagentServiceError(
                "message_too_long",
                "message too long (>4000)",
            )

        entry = self._registry.get_run(run_id)
        if entry is None:
            raise SubagentServiceError(
                "not_found",
                f"run_id not found: {run_id}",
            )
        if requester_session_id is None:
            if entry.requester_agent_id != requester_agent_id:
                raise SubagentServiceError(
                    "out_of_scope",
                    "run does not belong to current agent scope",
                )
            requester_key = entry.requester_session_key
        else:
            requester_key = self._requester_key(
                requester_agent_id,
                requester_session_id,
            )
            allowed = {
                record.run_id
                for record in self._registry.list_descendant_runs(
                    requester_key,
                    include_recent_minutes=(
                        self.MAX_RECENT_MINUTES
                    ),
                )
            }
            if run_id not in allowed:
                raise SubagentServiceError(
                    "out_of_scope",
                    "run not found in current session scope",
                )
        if entry.ended_at is not None:
            raise SubagentServiceError(
                "already_ended",
                "run already finished",
            )
        if requester_key == entry.child_session_key:
            raise SubagentServiceError(
                "self_steer",
                "Sub-agent cannot steer itself",
            )

        parsed_child = (
            self._session_manager.session_id_from_session_key(
                entry.child_session_key
            )
        )
        if not parsed_child:
            raise SubagentServiceError(
                "invalid_child_session",
                "invalid child session key",
            )
        target_agent_id, target_session_id = parsed_child
        next_run_id = self._id_factory()
        next_record = (
            self._registry.replace_active_run_for_steer(
                previous_run_id=run_id,
                next_run_id=next_run_id,
                task=message,
            )
        )
        if next_record is None:
            raise SubagentServiceError(
                "claim_failed",
                "run ended before steer could take ownership",
            )

        target_config = (
            self._resolve_agent_config(target_agent_id) or {}
        )
        subagent_config = target_config.get("subagents") or {}
        try:
            self._session_manager.save_message(
                target_session_id,
                target_agent_id,
                "user",
                message,
            )
            self._start_runner(
                requester_agent_id=entry.requester_agent_id,
                run_id=next_run_id,
                session_id=target_session_id,
                agent_id=target_agent_id,
                task=message,
                requester_key=entry.requester_session_key,
                run_timeout_seconds=self._run_timeout(
                    subagent_config
                ),
            )
        except Exception as exc:
            self._registry.mark_terminated(
                next_run_id,
                f"start error: {exc}",
            )
            raise SubagentServiceError(
                "start_failed",
                f"Failed to apply new instruction: {exc}",
            ) from exc

        current = (
            self._registry.get_run(next_run_id)
            or next_record
        )
        label = entry.label or entry.task[:50] or "No Label"
        return SteerResult(
            record=current,
            replaced_run_id=run_id,
            label=label,
        )

    def child_requester_key(
        self,
        record: SubagentRunRecord,
    ) -> str:
        return self._registry.session_key_from_child_session_key(
            record.child_session_key
        )

    def _requester_key(
        self,
        agent_id: str,
        session_id: str | None,
    ) -> str:
        effective_session_id = (
            (session_id or "").strip()
            or self._session_manager.resolve_main_session_id(
                agent_id
            )
        )
        return self._session_manager.session_key_from_session_id(
            agent_id,
            effective_session_id,
        )

    def _validate_target_allowed(
        self,
        requester_agent_id: str,
        target_agent_id: str,
    ) -> None:
        requester_config = (
            self._resolve_agent_config(requester_agent_id) or {}
        )
        allow = (
            requester_config.get("subagents") or {}
        ).get("allow_agents") or []
        allow_any = "*" in allow
        allowed = {
            item for item in allow if item and item != "*"
        }
        if (
            not allow_any
            and target_agent_id != requester_agent_id
            and target_agent_id not in allowed
        ):
            raise SubagentServiceError(
                "target_forbidden",
                "Current agent is not allowed to spawn tasks "
                f"for '{target_agent_id}'",
            )

    def _recent_minutes(self, value: int | None) -> int:
        if value is not None and value > 0:
            return max(
                1,
                min(self.MAX_RECENT_MINUTES, int(value)),
            )
        config = self._get_config() or {}
        default = (
            config.get("agents", {})
            .get("defaults", {})
            .get("subagents", {})
            .get("recent_minutes")
        )
        if not isinstance(default, (int, float)):
            default = 30
        return max(
            1,
            min(self.MAX_RECENT_MINUTES, int(default)),
        )

    def _start_runner(self, **kwargs: Any) -> None:
        if self._runner_factory is None:
            raise RuntimeError("subagent runner is unavailable")
        requester_agent_id = kwargs.pop(
            "requester_agent_id"
        )
        runner = self._runner_factory(requester_agent_id)
        runner.start(**kwargs)

    @staticmethod
    def _nonnegative_int(value: Any, *, default: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return max(0, parsed)

    @staticmethod
    def _run_timeout(config: dict[str, Any]) -> int:
        try:
            timeout = int(config.get("run_timeout_seconds", 0))
        except (TypeError, ValueError):
            return 0
        return max(0, timeout)
