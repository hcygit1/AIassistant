"""Per-agent runtime state and persistence."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


logger = logging.getLogger("runtime.agent")


@dataclass
class AgentState:
    agent_id: str
    compaction_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_turns: int = 0
    think_level: int = 0
    verbose: bool = False
    reasoning: bool = False
    last_active: float = 0.0
    _tools_cache: list | None = field(default=None, repr=False)

    @property
    def thinking(self) -> bool:
        return self.think_level > 0

    def record_turn(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_turns += 1
        self.last_active = time.time()

    def invalidate_tools(self) -> None:
        self._tools_cache = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "compaction_count": self.compaction_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_turns": self.total_turns,
            "think_level": self.think_level,
            "verbose": self.verbose,
            "reasoning": self.reasoning,
            "last_active": self.last_active,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentState":
        return cls(
            agent_id=data.get("agent_id", ""),
            compaction_count=data.get("compaction_count", 0),
            total_input_tokens=data.get("total_input_tokens", 0),
            total_output_tokens=data.get("total_output_tokens", 0),
            total_turns=data.get("total_turns", 0),
            think_level=data.get("think_level", 0),
            verbose=data.get("verbose", False),
            reasoning=data.get("reasoning", False),
            last_active=data.get("last_active", 0.0),
        )

    def save_to_disk(self, path: Path) -> None:
        try:
            path.write_text(
                json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.debug("AgentState saved for %s", self.agent_id)
        except Exception as e:
            logger.warning("Failed to save AgentState for %s: %s", self.agent_id, e)

    @classmethod
    def load_from_disk(cls, path: Path, agent_id: str) -> "AgentState":
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                state = cls.from_dict(data)
                state.agent_id = agent_id
                logger.debug("AgentState loaded for %s", agent_id)
                return state
        except Exception as e:
            logger.warning("Failed to load AgentState for %s: %s", agent_id, e)
        return cls(agent_id=agent_id)
