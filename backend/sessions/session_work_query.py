"""Shared SQL filter construction for session-work reads."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SessionWorkFilter:
    kind: str | None = None
    kinds: list[str] | None = None
    status: str | None = None
    agent_id: str | None = None
    session_id: str | None = None
    run_id: str | None = None
    run_id_prefix: str | None = None
    exclude_run_id_prefix: str | None = None

    def to_sql(self) -> tuple[str, list[object]]:
        conditions: list[str] = []
        params: list[object] = []
        if self.kind:
            conditions.append("kind = ?")
            params.append(self.kind)
        if self.kinds:
            placeholders = ", ".join("?" for _ in self.kinds)
            conditions.append(f"kind IN ({placeholders})")
            params.extend(self.kinds)
        if self.status:
            conditions.append("status = ?")
            params.append(self.status)
        if self.agent_id:
            conditions.append("agent_id = ?")
            params.append(self.agent_id)
        if self.session_id:
            conditions.append("session_id = ?")
            params.append(self.session_id)
        if self.run_id:
            conditions.append("run_id = ?")
            params.append(self.run_id)
        if self.run_id_prefix:
            conditions.append("run_id LIKE ?")
            params.append(f"{self.run_id_prefix}%")
        if self.exclude_run_id_prefix:
            conditions.append("(run_id IS NULL OR run_id NOT LIKE ?)")
            params.append(f"{self.exclude_run_id_prefix}%")

        where = " AND ".join(conditions) if conditions else "1=1"
        return where, params
