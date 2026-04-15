from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Literal

TurnStatus = Literal["queued", "running", "done", "error", "cancelled"]
CancelReason = Literal["stopped_by_user", "client_disconnected"]


@dataclass
class UserTurnRuntime:
    """Aggregated runtime state for a single user turn."""

    turn_id: str
    agent_id: str
    session_id: str
    status: TurnStatus
    stream_queue: asyncio.Queue[str | None]
    created_at: float = field(default_factory=time.time)
    execution_task: asyncio.Task | None = None
    error: str | None = None
    cancel_reason: CancelReason | None = None
