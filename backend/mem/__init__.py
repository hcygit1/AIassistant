"""mem — 记忆系统包

Chunk → Task → Skill 三层记忆，基于 SQLite + FTS5 + sqlite-vec ANN。
"""

from importlib import import_module
from typing import Any

from mem.models import (
    Chunk,
    SearchHit,
    Skill,
    SkillSearchHit,
    Task,
    TaskSearchHit,
)


_LAZY_EXPORTS = {
    "MemStore": ("mem.store", "MemStore"),
    "MemEmbedder": ("mem.embedder", "MemEmbedder"),
    "MemWorker": ("mem.worker", "MemWorker"),
    "MemRecall": ("mem.recall", "MemRecall"),
    "RecallResult": ("mem.recall", "RecallResult"),
    "RecallHit": ("mem.recall", "RecallHit"),
    "TaskGroup": ("mem.recall", "TaskGroup"),
    "MemTaskProcessor": ("mem.task_processor", "MemTaskProcessor"),
    "MemSkillEvolver": ("mem.skill_evolver", "MemSkillEvolver"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})

__all__ = [
    "MemStore",
    "MemEmbedder",
    "MemWorker",
    "MemRecall",
    "MemTaskProcessor",
    "MemSkillEvolver",
    "Chunk",
    "Task",
    "Skill",
    "RecallResult",
    "RecallHit",
    "TaskGroup",
    "SearchHit",
    "TaskSearchHit",
    "SkillSearchHit",
]
