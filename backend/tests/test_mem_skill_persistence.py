from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from mem.models import Skill, Task
from mem.skill_evaluation import CreateEvalResult


def _persistence_symbol() -> Any:
    try:
        from mem.skill_persistence import persist_new_skill
    except ModuleNotFoundError as exc:
        raise AssertionError(
            "mem.skill_persistence should own new skill persistence"
        ) from exc
    return persist_new_skill


def _skill() -> Skill:
    return Skill(
        id="skill-1",
        name="generated-skill",
        description="generated description",
        owner="agent:main",
    )


class _FakeStore:
    def __init__(self, *, insert_error: Exception | None = None) -> None:
        self.insert_error = insert_error
        self.events: list[tuple[Any, ...]] = []

    def insert_skill(self, skill: Skill) -> None:
        self.events.append(("insert", skill))
        if self.insert_error is not None:
            raise self.insert_error

    def upsert_skill_embedding(self, skill_id: str, vec: list[float]) -> None:
        self.events.append(("upsert", skill_id, vec))


class _FakeEmbedder:
    def __init__(self, events: list[tuple[Any, ...]], error: BaseException | None = None) -> None:
        self.events = events
        self.error = error

    async def embed_query(self, text: str) -> list[float]:
        self.events.append(("embed", text))
        if self.error is not None:
            raise self.error
        return [0.1, 0.2]


class SkillPersistenceTests(unittest.IsolatedAsyncioTestCase):
    def test_persistence_has_a_neutral_owner_and_evolver_delegates(self) -> None:
        persist_new_skill = _persistence_symbol()
        persistence_path = BACKEND_DIR / "mem" / "skill_persistence.py"
        self.assertTrue(
            persistence_path.is_file(),
            "mem.skill_persistence should own new skill persistence",
        )

        evolver_source = (BACKEND_DIR / "mem" / "skill_evolver.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("from mem.skill_persistence import persist_new_skill", evolver_source)
        generate_body = evolver_source.split("async def _generate_skill(", 1)[1].split(
            "@staticmethod", 1
        )[0]
        self.assertIn("await persist_new_skill(", generate_body)
        self.assertNotIn("self.store.insert_skill(skill)", generate_body)
        self.assertNotIn("self.embedder.embed_query", generate_body)
        self.assertTrue(callable(persist_new_skill))

    async def test_insert_embed_and_upsert_keep_their_order_and_values(self) -> None:
        persist_new_skill = _persistence_symbol()
        store = _FakeStore()
        embedder = _FakeEmbedder(store.events)
        skill = _skill()

        error = await persist_new_skill(skill, store=store, embedder=embedder)

        self.assertIsNone(error)
        self.assertEqual(
            store.events,
            [
                ("insert", skill),
                ("embed", "generated-skill generated description"),
                ("upsert", "skill-1", [0.1, 0.2]),
            ],
        )

    async def test_insert_failure_propagates_before_embedding(self) -> None:
        persist_new_skill = _persistence_symbol()
        store = _FakeStore(insert_error=RuntimeError("insert failed"))
        embedder = _FakeEmbedder(store.events)

        with self.assertRaisesRegex(RuntimeError, "insert failed"):
            await persist_new_skill(_skill(), store=store, embedder=embedder)

        self.assertEqual(store.events, [("insert", _skill())])

    async def test_embedding_and_upsert_failures_return_the_original_error(self) -> None:
        persist_new_skill = _persistence_symbol()

        embed_error = RuntimeError("embed failed")
        embed_store = _FakeStore()
        observed_embed_error = await persist_new_skill(
            _skill(),
            store=embed_store,
            embedder=_FakeEmbedder(embed_store.events, embed_error),
        )
        self.assertIs(observed_embed_error, embed_error)
        self.assertEqual(
            embed_store.events,
            [("insert", _skill()), ("embed", "generated-skill generated description")],
        )

        upsert_error = RuntimeError("upsert failed")

        class FailingUpsertStore(_FakeStore):
            def upsert_skill_embedding(self, skill_id: str, vec: list[float]) -> None:
                super().upsert_skill_embedding(skill_id, vec)
                raise upsert_error

        upsert_store = FailingUpsertStore()
        observed_upsert_error = await persist_new_skill(
            _skill(),
            store=upsert_store,
            embedder=_FakeEmbedder(upsert_store.events),
        )
        self.assertIs(observed_upsert_error, upsert_error)
        self.assertEqual(upsert_store.events[-1], ("upsert", "skill-1", [0.1, 0.2]))

    async def test_embedding_cancellation_is_not_downgraded(self) -> None:
        persist_new_skill = _persistence_symbol()
        store = _FakeStore()

        with self.assertRaises(asyncio.CancelledError):
            await persist_new_skill(
                _skill(),
                store=store,
                embedder=_FakeEmbedder(store.events, asyncio.CancelledError()),
            )

    async def test_evolver_keeps_the_embedding_warning_and_returns_the_skill(self) -> None:
        _persistence_symbol()
        from mem.skill_evolver import MemSkillEvolver

        store = _FakeStore()
        persistence_error = RuntimeError("embedding failed")
        generated = (
            'description: "generated description"\n'
            "# Skill\n\nA complete reusable workflow with verification steps."
        )

        class PersistenceEvolver(MemSkillEvolver):
            async def _llm_call(
                self,
                prompt: str,
                *,
                max_tokens: int = 1024,
                temperature: float = 0.1,
            ) -> str:
                return generated

            async def _score_quality(self, content: str, task: Task) -> float:
                return 7.0

        evolver = PersistenceEvolver(store=store, embedder=_FakeEmbedder(store.events))
        persist_mock = AsyncMock(return_value=persistence_error)
        with (
            patch("mem.skill_evolver.persist_new_skill", persist_mock),
            patch("mem.skill_evolver.logger.warning") as warning,
        ):
            skill = await evolver._generate_skill(
                Task(id="task-1", session_key="session-1"),
                [],
                CreateEvalResult(suggested_name="generated-skill"),
            )

        self.assertIsNotNone(skill)
        persist_mock.assert_awaited_once_with(skill, store=store, embedder=evolver.embedder)
        warning.assert_called_once_with("Skill embedding failed: %s", persistence_error)


if __name__ == "__main__":
    unittest.main()
