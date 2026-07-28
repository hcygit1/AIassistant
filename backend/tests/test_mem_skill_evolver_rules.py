from __future__ import annotations

import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from mem.models import Chunk, Task
from mem.skill_evolver import MemSkillEvolver


def _chunk(chunk_id: str, role: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        session_key="session-1",
        turn_id="turn-1",
        seq=0,
        role=role,
        content=f"{role} content",
    )


def _task(*, status: str = "completed") -> Task:
    return Task(
        id="task-1",
        session_key="session-1",
        status=status,
        summary="s" * 100,
    )


class SkillEvolverRuleTests(unittest.TestCase):
    def test_configured_minimum_chunks_filters_before_role_checks(self) -> None:
        evolver = MemSkillEvolver(
            store=None,  # type: ignore[arg-type]
            embedder=None,  # type: ignore[arg-type]
            min_chunks_for_eval=2,
        )

        self.assertEqual(
            evolver._rule_filter([_chunk("user", "user")], _task()),
            "chunk数量不足 (1 < 2)",
        )

    def test_reaching_the_chunk_threshold_preserves_existing_rules(self) -> None:
        evolver = MemSkillEvolver(
            store=None,  # type: ignore[arg-type]
            embedder=None,  # type: ignore[arg-type]
            min_chunks_for_eval=2,
        )

        self.assertIsNone(
            evolver._rule_filter(
                [_chunk("user", "user"), _chunk("assistant", "assistant")],
                _task(),
            )
        )

    def test_skipped_task_keeps_precedence_over_chunk_count(self) -> None:
        evolver = MemSkillEvolver(
            store=None,  # type: ignore[arg-type]
            embedder=None,  # type: ignore[arg-type]
            min_chunks_for_eval=6,
        )

        self.assertEqual(
            evolver._rule_filter([], _task(status="skipped")),
            "task状态为skipped",
        )


if __name__ == "__main__":
    unittest.main()
