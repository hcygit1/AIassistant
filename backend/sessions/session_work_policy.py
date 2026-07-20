"""Canonical policy for system work submitted to a session dispatcher."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


PRIORITY_ANNOUNCE = 0
PRIORITY_CRON = 2
PRIORITY_HEARTBEAT = 3

SystemWorkKind = Literal["announce", "cron", "heartbeat"]


@dataclass(frozen=True, slots=True)
class SystemWorkPolicy:
    priority: int
    recover_on_restart: bool


_SYSTEM_WORK_POLICIES: dict[SystemWorkKind, SystemWorkPolicy] = {
    "announce": SystemWorkPolicy(
        priority=PRIORITY_ANNOUNCE,
        recover_on_restart=False,
    ),
    "cron": SystemWorkPolicy(
        priority=PRIORITY_CRON,
        recover_on_restart=True,
    ),
    "heartbeat": SystemWorkPolicy(
        priority=PRIORITY_HEARTBEAT,
        recover_on_restart=False,
    ),
}


def deliver_system_work(
    work_delivery: Any,
    *,
    kind: SystemWorkKind | str,
    content: str,
    agent_id: str,
    session_id: str,
    **delivery_options: Any,
) -> int:
    try:
        policy = _SYSTEM_WORK_POLICIES[kind]
    except KeyError as exc:
        raise ValueError(f"unknown system work kind: {kind}") from exc
    return work_delivery.deliver(
        kind=kind,
        priority=policy.priority,
        recover_on_restart=policy.recover_on_restart,
        content=content,
        agent_id=agent_id,
        session_id=session_id,
        **delivery_options,
    )
