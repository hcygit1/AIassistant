"""Public result and error models for subagent application services."""

from __future__ import annotations

from dataclasses import dataclass

from subagents.subagent_run_model import SubagentRunRecord


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
