"""Structured events emitted during a user turn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def _snapshot_json_value(value: Any, path: str) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        snapshot: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"TurnEvent payload key at {path} is not a string"
                )
            snapshot[key] = _snapshot_json_value(item, f"{path}.{key}")
        return snapshot
    if isinstance(value, (list, tuple)):
        return [
            _snapshot_json_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"TurnEvent payload value at {path} is not JSON-compatible: "
        f"{type(value).__name__}"
    )


@dataclass(frozen=True, slots=True)
class TurnEvent:
    type: str
    payload: dict[str, Any]

    def __post_init__(self) -> None:
        event_type = str(self.type or "")
        payload = _snapshot_json_value(self.payload, "payload")
        payload["type"] = event_type
        object.__setattr__(self, "type", event_type)
        object.__setattr__(self, "payload", payload)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "TurnEvent":
        data = dict(payload)
        event_type = str(data.get("type") or "")
        return cls(type=event_type, payload=data)

    @classmethod
    def error(cls, message: str) -> "TurnEvent":
        return cls.from_payload({"type": "error", "error": message})

    @property
    def error_message(self) -> str | None:
        if self.type != "error":
            return None
        message = self.payload.get("error")
        return str(message) if message else None
