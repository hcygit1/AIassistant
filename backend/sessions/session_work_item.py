"""Domain model for one queued unit of session work."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from turns.events import TurnEvent


@dataclass(order=False)
class SessionWorkItem:
    """A user turn or system task waiting for session execution."""

    kind: str
    priority: int
    content: str
    agent_id: str
    session_id: str
    prompt_mode: str = "minimal"
    persist_role: str = "system"
    run_id: str | None = None
    work_id: str | None = None
    created_at: float = field(default_factory=time.time)
    result_handler: Callable[[str], Awaitable[None]] | None = None
    on_success: Callable[[], Any] | None = None
    on_failure: Callable[[], Any] | None = None
    on_failure_async: Callable[[Exception], Awaitable[None]] | None = None
    on_cancel: Callable[[], Any] | None = None
    turn_id: str | None = None
    stream_queue: asyncio.Queue[TurnEvent | None] | None = None
