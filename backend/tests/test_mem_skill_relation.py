from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from mem.models import Skill, SkillSearchHit, Task


def _relation_functions() -> tuple[Any, Any]:
    try:
        from mem.skill_relation import find_related_skill, judge_related_skill
    except ModuleNotFoundError as exc:
        raise AssertionError(
            "mem.skill_relation should own related skill matching"
        ) from exc
    return find_related_skill, judge_related_skill


def _skill(
    skill_id: str,
    *,
    owner: str = "agent:main",
    status: str = "active",
) -> Skill:
    return Skill(
        id=skill_id,
        name=f"skill-{skill_id}",
        description=f"description-{skill_id}",
        owner=owner,
        status=status,
    )


def _hit(skill_id: str) -> SkillSearchHit:
    return SkillSearchHit(
        skill_id=skill_id,
        score=1.0,
        name=f"skill-{skill_id}",
        description=f"description-{skill_id}",
    )


class _FakeEmbedder:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.queries: list[str] = []

    async def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        if self.error:
            raise self.error
        return [0.1, 0.2]


class _FakeStore:
    def __init__(
        self,
        skills: list[Skill],
        *,
        fts_hits: list[SkillSearchHit] | None = None,
        ann_hits: list[SkillSearchHit] | None = None,
        fts_error: Exception | None = None,
    ) -> None:
        self.skills = {skill.id: skill for skill in skills}
        self.fts_hits = fts_hits or []
        self.ann_hits = ann_hits or []
        self.fts_error = fts_error
        self.fts_calls: list[tuple[str, int, str | None]] = []
        self.ann_calls: list[tuple[list[float], int, str | None]] = []

    def fts_search_skills(
        self,
        query: str,
        limit: int = 10,
        owner: str | None = None,
    ) -> list[SkillSearchHit]:
        self.fts_calls.append((query, limit, owner))
        if self.fts_error:
            raise self.fts_error
        return self.fts_hits

    def ann_search_skills(
        self,
        query_vec: list[float],
        top_k: int = 10,
        owner: str | None = None,
    ) -> list[SkillSearchHit]:
        self.ann_calls.append((query_vec, top_k, owner))
        return self.ann_hits

    def get_skill(self, skill_id: str) -> Skill | None:
        return self.skills.get(skill_id)


class SkillRelationTests(unittest.IsolatedAsyncioTestCase):
    def test_related_skill_logic_has_a_neutral_owner(self) -> None:
        relation_path = BACKEND_DIR / "mem" / "skill_relation.py"
        self.assertTrue(
            relation_path.is_file(),
            "mem.skill_relation should own related skill matching",
        )

        evolver_source = (BACKEND_DIR / "mem" / "skill_evolver.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("from mem.skill_relation import", evolver_source)
        self.assertNotIn("RELATED_SKILL_JUDGE_PROMPT =", evolver_source)

    async def test_lookup_merges_and_filters_candidates_before_judging(self) -> None:
        find_related_skill, _judge_related_skill = _relation_functions()
        active = _skill("active")
        draft = _skill("draft", status="draft")
        other_owner = _skill("other", owner="agent:other")
        store = _FakeStore(
            [active, draft, other_owner],
            fts_hits=[_hit("active"), _hit("draft")],
            ann_hits=[_hit("active"), _hit("other"), _hit("missing")],
        )
        embedder = _FakeEmbedder()
        task = Task(
            id="task-1",
            session_key="session-1",
            owner="agent:main",
            summary="x" * 650,
        )
        judged: list[list[Skill]] = []

        async def judge(_task: Task, candidates: list[Skill]) -> Skill | None:
            judged.append(candidates)
            return candidates[0] if candidates else None

        related = await find_related_skill(
            task,
            store=store,
            embedder=embedder,
            judge_related=judge,
        )

        self.assertEqual(related.id, "active")
        self.assertEqual(store.fts_calls, [("x" * 600, 10, "agent:main")])
        self.assertEqual(store.ann_calls, [([0.1, 0.2], 10, "agent:main")])
        self.assertEqual(embedder.queries, ["x" * 600])
        self.assertEqual([[skill.id for skill in items] for items in judged], [["active"]])

    async def test_lookup_preserves_default_owner_filtering(self) -> None:
        find_related_skill, _judge_related_skill = _relation_functions()
        candidate = _skill("main-owner")
        store = _FakeStore([candidate], fts_hits=[_hit("main-owner")])
        judged: list[list[str]] = []

        async def judge(_task: Task, candidates: list[Skill]) -> Skill | None:
            judged.append([skill.id for skill in candidates])
            return candidates[0]

        related = await find_related_skill(
            Task(
                id="task-default-owner",
                session_key="session-1",
                owner="",
                summary="query",
            ),
            store=store,
            embedder=_FakeEmbedder(),
            judge_related=judge,
        )

        self.assertIs(related, candidate)
        self.assertEqual(store.fts_calls, [("query", 10, "")])
        self.assertEqual(judged, [["main-owner"]])

    async def test_lookup_preserves_search_fallbacks(self) -> None:
        find_related_skill, _judge_related_skill = _relation_functions()
        store = _FakeStore([], fts_error=RuntimeError("fts failed"))
        embedder = _FakeEmbedder(error=RuntimeError("embed failed"))
        judge_calls = 0

        async def judge(_task: Task, _candidates: list[Skill]) -> Skill | None:
            nonlocal judge_calls
            judge_calls += 1
            return None

        empty_summary = await find_related_skill(
            Task(id="task-empty", session_key="session-1", summary="  "),
            store=store,
            embedder=embedder,
            judge_related=judge,
        )
        failed_search = await find_related_skill(
            Task(id="task-failed", session_key="session-1", summary="query"),
            store=store,
            embedder=embedder,
            judge_related=judge,
        )

        self.assertIsNone(empty_summary)
        self.assertIsNone(failed_search)
        self.assertEqual(judge_calls, 0)

    async def test_lookup_uses_either_search_channel_when_the_other_fails(self) -> None:
        find_related_skill, _judge_related_skill = _relation_functions()
        candidate = _skill("candidate")
        task = Task(
            id="task-partial-search",
            session_key="session-1",
            owner="agent:main",
            summary="query",
        )
        judged: list[list[str]] = []

        async def judge(_task: Task, candidates: list[Skill]) -> Skill | None:
            judged.append([skill.id for skill in candidates])
            return candidates[0]

        fts_failed = await find_related_skill(
            task,
            store=_FakeStore(
                [candidate],
                ann_hits=[_hit("candidate")],
                fts_error=RuntimeError("fts failed"),
            ),
            embedder=_FakeEmbedder(),
            judge_related=judge,
        )
        vector_failed = await find_related_skill(
            task,
            store=_FakeStore([candidate], fts_hits=[_hit("candidate")]),
            embedder=_FakeEmbedder(error=RuntimeError("embed failed")),
            judge_related=judge,
        )

        self.assertIs(fts_failed, candidate)
        self.assertIs(vector_failed, candidate)
        self.assertEqual(judged, [["candidate"], ["candidate"]])

    async def test_judge_selects_only_a_valid_candidate_index(self) -> None:
        _find_related_skill, judge_related_skill = _relation_functions()
        task = Task(
            id="task-judge",
            session_key="session-1",
            title="repair database",
            summary="verified database repair workflow",
        )
        candidates = [_skill("one"), _skill("two")]
        calls: list[tuple[str, int, float]] = []

        async def llm_call(
            prompt: str,
            *,
            max_tokens: int,
            temperature: float,
        ) -> str:
            calls.append((prompt, max_tokens, temperature))
            return '{"selectedIndex": 2, "reason": "same workflow"}'

        def parse_json(raw: str, fallback: dict[str, Any]) -> dict[str, Any]:
            self.assertIn("selectedIndex", raw)
            self.assertEqual(fallback, {"selectedIndex": 0, "reason": ""})
            return {"selectedIndex": 2, "reason": "same workflow"}

        selected = await judge_related_skill(
            task,
            candidates,
            llm_call=llm_call,
            parse_json=parse_json,
        )

        self.assertIs(selected, candidates[1])
        self.assertEqual(calls[0][1:], (256, 0))
        self.assertIn("1. [skill-one]", calls[0][0])
        self.assertIn("2. [skill-two]", calls[0][0])

        async def out_of_range_call(
            _prompt: str,
            *,
            max_tokens: int,
            temperature: float,
        ) -> str:
            return "ignored"

        self.assertIsNone(
            await judge_related_skill(
                task,
                candidates,
                llm_call=out_of_range_call,
                parse_json=lambda _raw, _fallback: {"selectedIndex": 3},
            )
        )

    async def test_skill_evolver_compatibility_uses_overridable_dependencies(self) -> None:
        _relation_functions()
        from mem.skill_evolver import MemSkillEvolver

        candidate = _skill("candidate")
        store = _FakeStore(
            [candidate],
            fts_hits=[_hit("candidate")],
        )

        class CustomEvolver(MemSkillEvolver):
            def __init__(self) -> None:
                super().__init__(store=store, embedder=_FakeEmbedder())
                self.judge_calls: list[list[str]] = []
                self.llm_calls: list[tuple[int, float]] = []

            async def _judge_related(
                self,
                task: Task,
                candidates: list[Skill],
            ) -> Skill | None:
                self.judge_calls.append([skill.id for skill in candidates])
                return candidates[0]

            async def _llm_call(
                self,
                prompt: str,
                *,
                max_tokens: int = 1024,
                temperature: float = 0.1,
            ) -> str:
                self.llm_calls.append((max_tokens, temperature))
                return '{"selectedIndex": 1}'

        evolver = CustomEvolver()
        task = Task(
            id="task-compat",
            session_key="session-1",
            owner="agent:main",
            title="repair database",
            summary="verified repair workflow",
        )

        self.assertIs(await evolver._find_related_skill(task), candidate)
        self.assertEqual(evolver.judge_calls, [["candidate"]])

        selected = await MemSkillEvolver._judge_related(
            evolver,
            task,
            [candidate],
        )
        self.assertIs(selected, candidate)
        self.assertEqual(evolver.llm_calls, [(256, 0)])

        class FailingEvolver(MemSkillEvolver):
            async def _llm_call(
                self,
                prompt: str,
                *,
                max_tokens: int = 1024,
                temperature: float = 0.1,
            ) -> str:
                raise RuntimeError("judge failed")

        failing_evolver = FailingEvolver(
            store=store,
            embedder=_FakeEmbedder(),
        )
        with self.assertLogs("mem.skill_evolver", level="WARNING") as logs:
            failed = await failing_evolver._judge_related(task, [candidate])
        self.assertIsNone(failed)
        self.assertEqual(
            logs.output,
            ["WARNING:mem.skill_evolver:Skill relation judge failed: judge failed"],
        )


if __name__ == "__main__":
    unittest.main()
