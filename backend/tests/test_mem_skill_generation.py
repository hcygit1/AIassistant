from __future__ import annotations

import asyncio
import hashlib
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from mem.models import Chunk, Skill, Task
from mem.skill_evaluation import CreateEvalResult


def _generation_symbols() -> tuple[Any, Any, Any]:
    try:
        from mem.skill_generation import (
            SKILL_GENERATE_PROMPT,
            build_skill_generation_prompt,
            generate_skill_content,
        )
    except ModuleNotFoundError as exc:
        raise AssertionError(
            "mem.skill_generation should own skill content generation"
        ) from exc
    return (
        SKILL_GENERATE_PROMPT,
        build_skill_generation_prompt,
        generate_skill_content,
    )


def _chunk(chunk_id: str, role: str, content: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        session_key="session-1",
        turn_id=f"turn-{chunk_id}",
        seq=0,
        role=role,
        content=content,
    )


class _FakeStore:
    def __init__(self) -> None:
        self.inserted: list[Skill] = []
        self.embeddings: list[tuple[str, list[float]]] = []

    def insert_skill(self, skill: Skill) -> None:
        self.inserted.append(skill)

    def upsert_skill_embedding(self, skill_id: str, vec: list[float]) -> None:
        self.embeddings.append((skill_id, vec))


class _FakeEmbedder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.1, 0.2]


class SkillGenerationTests(unittest.IsolatedAsyncioTestCase):
    def test_content_generation_has_a_neutral_owner_and_compatible_prompt(self) -> None:
        generation_prompt, _build_prompt, _generate_skill_content = (
            _generation_symbols()
        )
        generation_path = BACKEND_DIR / "mem" / "skill_generation.py"
        self.assertTrue(
            generation_path.is_file(),
            "mem.skill_generation should own skill content generation",
        )

        evolver_source = (BACKEND_DIR / "mem" / "skill_evolver.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("from mem.skill_generation import", evolver_source)
        self.assertNotIn("SKILL_GENERATE_PROMPT =", evolver_source)

        from mem.skill_evolver import SKILL_GENERATE_PROMPT as compatible_prompt

        self.assertIs(compatible_prompt, generation_prompt)

    async def test_prompt_hash_and_llm_parameters_match_the_baseline(self) -> None:
        _generation_prompt, build_skill_generation_prompt, generate_skill_content = (
            _generation_symbols()
        )
        calls: list[tuple[str, int, float]] = []

        async def llm_call(
            prompt: str,
            *,
            max_tokens: int = 1024,
            temperature: float = 0.1,
        ) -> str:
            calls.append((prompt, max_tokens, temperature))
            return "generated"

        prompt = build_skill_generation_prompt(
            Task(
                id="task-prompt",
                session_key="session-1",
                title="title",
                summary="summary",
            ),
            CreateEvalResult(suggested_name="name"),
            original_goal="goal",
            evidence="evidence",
        )
        content = await generate_skill_content(
            prompt,
            llm_call=llm_call,
        )

        self.assertEqual(content, "generated")
        self.assertEqual(calls[0][1:], (4096, 0.2))
        self.assertEqual(len(calls[0][0]), 1354)
        self.assertEqual(
            hashlib.sha256(calls[0][0].encode()).hexdigest(),
            "5ab3199b46ceb7a012b201e2f0789b636db32e941d1a3f6d157889002b9abaff",
        )

    async def test_inputs_are_truncated_and_callbacks_keep_their_order(self) -> None:
        generation_prompt, build_skill_generation_prompt, generate_skill_content = (
            _generation_symbols()
        )
        chunks = [
            _chunk("user", "user", "goal"),
            _chunk("assistant", "assistant", "result"),
        ]
        task = Task(
            id="task-truncate",
            session_key="session-1",
            title="workflow",
            summary="s" * 2100,
        )
        events: list[str] = []
        observed_prompt = ""

        def extract_original_goal(received: list[Chunk]) -> str:
            self.assertIs(received, chunks)
            events.append("goal")
            return "g" * 1300

        def build_skill_evidence(received: list[Chunk]) -> str:
            self.assertIs(received, chunks)
            events.append("evidence")
            return "e" * 8100

        async def llm_call(
            prompt: str,
            *,
            max_tokens: int = 1024,
            temperature: float = 0.1,
        ) -> str:
            nonlocal observed_prompt
            events.append("llm")
            observed_prompt = prompt
            return "generated"

        original_goal = extract_original_goal(chunks)
        evidence = build_skill_evidence(chunks)
        prompt = build_skill_generation_prompt(
            task,
            CreateEvalResult(suggested_name="workflow-skill"),
            original_goal=original_goal,
            evidence=evidence,
        )
        await generate_skill_content(
            prompt,
            llm_call=llm_call,
        )

        expected_prompt = (
            generation_prompt
            .replace("{NAME}", "workflow-skill")
            .replace("{TITLE}", "workflow")
            .replace("{SUMMARY}", "s" * 2000)
            .replace("{ORIGINAL_GOAL}", "g" * 1200)
            .replace("{EVIDENCE}", "e" * 8000)
        )
        self.assertEqual(events, ["goal", "evidence", "llm"])
        self.assertEqual(observed_prompt, expected_prompt)
        self.assertNotIn("s" * 2001, observed_prompt)
        self.assertNotIn("g" * 1201, observed_prompt)
        self.assertNotIn("e" * 8001, observed_prompt)

    async def test_prompt_preserves_sequential_placeholder_replacement(self) -> None:
        _generation_prompt, build_skill_generation_prompt, generate_skill_content = (
            _generation_symbols()
        )
        observed_prompt = ""

        async def llm_call(
            prompt: str,
            *,
            max_tokens: int = 1024,
            temperature: float = 0.1,
        ) -> str:
            nonlocal observed_prompt
            observed_prompt = prompt
            return "generated"

        prompt = build_skill_generation_prompt(
            Task(
                id="task-placeholders",
                session_key="session-1",
                title="{SUMMARY}",
                summary="final-summary",
            ),
            CreateEvalResult(suggested_name="{TITLE}"),
            original_goal="{EVIDENCE}",
            evidence="final-evidence",
        )
        await generate_skill_content(
            prompt,
            llm_call=llm_call,
        )

        self.assertIn('name: "final-summary"', observed_prompt)
        self.assertIn("Task title: final-summary", observed_prompt)
        self.assertIn("Original goal:\nfinal-evidence", observed_prompt)

    async def test_compatibility_facade_uses_overrides_before_materialization(self) -> None:
        _generation_symbols()
        from mem.skill_evolver import MemSkillEvolver

        store = _FakeStore()
        embedder = _FakeEmbedder()
        events: list[str] = []
        generated_content = (
            '---\nname: "generated-skill"\n'
            'description: "Generated description"\n---\n'
            "# Generated skill\n\nA complete reusable workflow with verification."
        )

        class CustomEvolver(MemSkillEvolver):
            @staticmethod
            def _extract_original_goal(chunks: list[Chunk]) -> str:
                events.append("goal")
                return "custom goal"

            def _build_skill_evidence(self, chunks: list[Chunk]) -> str:
                events.append("evidence")
                return "custom evidence"

            async def _llm_call(
                self,
                prompt: str,
                *,
                max_tokens: int = 1024,
                temperature: float = 0.1,
            ) -> str:
                events.append("llm")
                self.llm_parameters = (max_tokens, temperature)
                return generated_content

            async def _score_quality(self, content: str, task: Task) -> float:
                events.append("quality")
                return 7.0

        evolver = CustomEvolver(store=store, embedder=embedder)
        task = Task(
            id="task-compat",
            session_key="session-1",
            owner="agent:main",
            title="Generate skill",
            summary="A verified workflow summary",
        )

        skill = await evolver._generate_skill(
            task,
            [_chunk("user", "user", "goal")],
            CreateEvalResult(suggested_name="generated-skill"),
        )

        self.assertIsNotNone(skill)
        self.assertEqual(events, ["goal", "evidence", "llm", "quality"])
        self.assertEqual(evolver.llm_parameters, (4096, 0.2))
        self.assertEqual(store.inserted, [skill])
        self.assertEqual(store.embeddings, [(skill.id, [0.1, 0.2])])
        self.assertEqual(embedder.queries, ["generated-skill Generated description"])

    async def test_facade_preserves_generation_failures_and_cancellation(self) -> None:
        _generation_symbols()
        from mem.skill_evolver import MemSkillEvolver

        class ResponseEvolver(MemSkillEvolver):
            def __init__(self, response: str | BaseException) -> None:
                super().__init__(store=_FakeStore(), embedder=_FakeEmbedder())
                self.response = response

            async def _llm_call(
                self,
                prompt: str,
                *,
                max_tokens: int = 1024,
                temperature: float = 0.1,
            ) -> str:
                if isinstance(self.response, BaseException):
                    raise self.response
                return self.response

        task = Task(id="task-failure", session_key="session-1")
        evaluation = CreateEvalResult(suggested_name="failure-skill")

        failing = ResponseEvolver(RuntimeError("generation failed"))
        with self.assertLogs("mem.skill_evolver", level="ERROR") as error_logs:
            failed = await failing._generate_skill(task, [], evaluation)
        self.assertIsNone(failed)
        self.assertEqual(
            error_logs.output,
            [
                "ERROR:mem.skill_evolver:Skill generation LLM call failed: "
                "generation failed"
            ],
        )

        short = ResponseEvolver("short")
        with self.assertLogs("mem.skill_evolver", level="WARNING") as warning_logs:
            skipped = await short._generate_skill(task, [], evaluation)
        self.assertIsNone(skipped)
        self.assertEqual(
            warning_logs.output,
            [
                "WARNING:mem.skill_evolver:"
                "Skill generation returned empty/short content"
            ],
        )

        cancelled = ResponseEvolver(asyncio.CancelledError())
        with self.assertRaises(asyncio.CancelledError):
            await cancelled._generate_skill(task, [], evaluation)

    async def test_facade_does_not_catch_goal_or_evidence_failures(self) -> None:
        _generation_symbols()
        from mem.skill_evolver import MemSkillEvolver

        class GoalFailureEvolver(MemSkillEvolver):
            @staticmethod
            def _extract_original_goal(chunks: list[Chunk]) -> str:
                raise RuntimeError("goal failed")

        class EvidenceFailureEvolver(MemSkillEvolver):
            def _build_skill_evidence(self, chunks: list[Chunk]) -> str:
                raise RuntimeError("evidence failed")

        evolver = GoalFailureEvolver(
            store=_FakeStore(),
            embedder=_FakeEmbedder(),
        )
        task = Task(id="task-goal-failure", session_key="session-1")

        with self.assertNoLogs("mem.skill_evolver"):
            with self.assertRaisesRegex(RuntimeError, "goal failed"):
                await evolver._generate_skill(
                    task,
                    [],
                    CreateEvalResult(suggested_name="failure-skill"),
                )

        evidence_failure = EvidenceFailureEvolver(
            store=_FakeStore(),
            embedder=_FakeEmbedder(),
        )
        with self.assertNoLogs("mem.skill_evolver"):
            with self.assertRaisesRegex(RuntimeError, "evidence failed"):
                await evidence_failure._generate_skill(
                    task,
                    [],
                    CreateEvalResult(suggested_name="failure-skill"),
                )

    async def test_facade_resolves_each_override_at_its_original_call_time(self) -> None:
        _generation_symbols()
        from mem.skill_evolver import MemSkillEvolver

        events: list[str] = []
        generated_content = (
            '---\nname: "dynamic-skill"\n'
            'description: "Dynamic description"\n---\n'
            "# Dynamic skill\n\nA complete reusable workflow with verification."
        )

        class DynamicEvolver(MemSkillEvolver):
            def _extract_original_goal(self, chunks: list[Chunk]) -> str:
                events.append("goal")
                self._build_skill_evidence = late_evidence  # type: ignore[method-assign]
                return "dynamic goal"

            def _build_skill_evidence(self, chunks: list[Chunk]) -> str:
                events.append("stale evidence")
                return "stale evidence"

            async def _llm_call(
                self,
                prompt: str,
                *,
                max_tokens: int = 1024,
                temperature: float = 0.1,
            ) -> str:
                events.append("stale llm")
                return generated_content

            async def _score_quality(self, content: str, task: Task) -> float:
                return 7.0

        evolver = DynamicEvolver(
            store=_FakeStore(),
            embedder=_FakeEmbedder(),
        )

        async def late_llm(
            prompt: str,
            *,
            max_tokens: int = 1024,
            temperature: float = 0.1,
        ) -> str:
            events.append("late llm")
            return generated_content

        def late_evidence(chunks: list[Chunk]) -> str:
            events.append("late evidence")
            evolver._llm_call = late_llm  # type: ignore[method-assign]
            return "dynamic evidence"

        skill = await evolver._generate_skill(
            Task(id="task-dynamic", session_key="session-1"),
            [],
            CreateEvalResult(suggested_name="dynamic-skill"),
        )

        self.assertIsNotNone(skill)
        self.assertEqual(events, ["goal", "late evidence", "late llm"])

    async def test_facade_preserves_dynamic_prompt_patch_path(self) -> None:
        _generation_symbols()
        from mem.skill_evolver import MemSkillEvolver

        observed_prompt = ""

        class PromptProbe(MemSkillEvolver):
            async def _llm_call(
                self,
                prompt: str,
                *,
                max_tokens: int = 1024,
                temperature: float = 0.1,
            ) -> str:
                nonlocal observed_prompt
                observed_prompt = prompt
                return "short"

        evolver = PromptProbe(
            store=_FakeStore(),
            embedder=_FakeEmbedder(),
        )
        prompt_template = (
            "name={NAME}|title={TITLE}|summary={SUMMARY}|"
            "goal={ORIGINAL_GOAL}|evidence={EVIDENCE}"
        )

        with (
            patch("mem.skill_evolver.SKILL_GENERATE_PROMPT", prompt_template),
            self.assertLogs("mem.skill_evolver", level="WARNING"),
        ):
            result = await evolver._generate_skill(
                Task(
                    id="task-prompt-patch",
                    session_key="session-1",
                    title="title",
                    summary="summary",
                ),
                [_chunk("user", "user", "goal")],
                CreateEvalResult(suggested_name="patched-skill"),
            )

        self.assertIsNone(result)
        self.assertEqual(
            observed_prompt,
            "name=patched-skill|title=title|summary=summary|"
            "goal=goal|evidence=1. [User] goal",
        )


if __name__ == "__main__":
    unittest.main()
