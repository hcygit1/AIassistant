"""Server-sent event encoding at the HTTP transport boundary."""

from __future__ import annotations

import json

from turns.events import TurnEvent


def encode_turn_event_sse(event: TurnEvent) -> str:
    data = json.dumps(event.payload, ensure_ascii=False)
    return f"event: {event.type}\ndata: {data}\n\n"
