"""Data classes for tool result persistence outcomes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PersistedToolResult:
    filepath: str
    original_size: int
    preview: str
    has_more: bool


@dataclass(frozen=True)
class PersistToolResultError:
    error: str


PersistOutcome = PersistedToolResult | PersistToolResultError
