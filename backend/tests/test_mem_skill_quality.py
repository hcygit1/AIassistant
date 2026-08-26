from __future__ import annotations

import asyncio
import hashlib
import sys
import unittest
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from mem.models import Task


def _quality_symbols() -> tuple[Any, Any]:
    try:
        from mem.skill_quality import SKILL_QUALITY_PROMPT, score_skill_quality
    except ModuleNotFoundError as exc:
        raise AssertionError(
            "mem.skill_quality should own skill quality scoring"
        ) from exc
    return SKILL_QUALITY_PROMPT, score_skill_quality


class SkillQualityTests(unittest.IsolatedAsyncioTestCase):
    def test_quality_scoring_has_a_neutral_owner(self) -> None:
        _quality_symbols()
        quality_path = BACKEND_DIR / "mem" / "skill_quality.py"
        self.assertTrue(
            quality_path.is_file(),
            "mem.skill_quality should own skill quality scoring",
        )

        evolver_source = (BACKEND_DIR / "mem" / "skill_evolver.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("from mem.skill_quality import", evolver_source)
        self.assertNotIn("You are a strict skill quality reviewer", evolver_source)

    async def test_prompt_and_llm_parameters_match_the_baseline(self) -> None:
        _quality_prompt, score_skill_quality = _quality_symbols()
        calls: list[tuple[str, int, float]] = []

        async def llm_call(
            prompt: str,
            *,
            max_tokens: int = 1024,
            temperature: float = 0.1,
        ) -> str:
            calls.append((prompt, max_tokens, temperature))
            return '{"score": 5}'

        score = await score_skill_quality(
            "content",
            Task(id="task-prompt", session_key="session-1", title="title"),
            llm_call=llm_call,
            parse_json=lambda _raw, _fallback: {"score": 5},
        )

        self.assertEqual(score, 5.0)
        self.assertEqual(calls[0][1:], (512, 0))
        self.assertEqual(len(calls[0][0]), 1297)
        self.assertEqual(
            hashlib.sha256(calls[0][0].encode()).hexdigest(),
            "d93582936de2365d8a74fe9b20d76034377d0e1ea8b5057b64719e025d90d68d",
        )

    async def test_numeric_scores_are_clamped_and_content_is_truncated(self) -> None:
        _quality_prompt, score_skill_quality = _quality_symbols()
        content = "x" * 2600
        observed_prompts: list[str] = []

        async def llm_call(
            prompt: str,
            *,
            max_tokens: int = 1024,
            temperature: float = 0.1,
        ) -> str:
            observed_prompts.append(prompt)
            return "raw"

        for raw_score, expected in (
            (-3, 0.0),
            (0, 0.0),
            (True, 1.0),
            (7.25, 7.25),
            (12, 10.0),
        ):
            with self.subTest(raw_score=raw_score):
                score = await score_skill_quality(
                    content,
                    Task(id="task-score", session_key="session-1", title="score"),
                    llm_call=llm_call,
                    parse_json=lambda _raw, _fallback, value=raw_score: {
                        "score": value
                    },
                )
                self.assertEqual(score, expected)

        self.assertEqual(len(observed_prompts), 5)
        self.assertTrue(all("x" * 2500 in prompt for prompt in observed_prompts))
        self.assertTrue(all("x" * 2501 not in prompt for prompt in observed_prompts))

    async def test_prompt_values_are_not_reprocessed_as_template_tokens(self) -> None:
        _quality_prompt, score_skill_quality = _quality_symbols()
        observed_prompts: list[str] = []

        async def llm_call(
            prompt: str,
            *,
            max_tokens: int = 1024,
            temperature: float = 0.1,
        ) -> str:
            observed_prompts.append(prompt)
            return '{"score": 5}'

        cases = (
            ("{CONTENT}", "actual-content"),
            ("$CONTENT", "${TITLE}"),
            ("${TITLE}", "$5"),
            ("$", "trailing-$"),
        )
        for index, (title, content) in enumerate(cases):
            await score_skill_quality(
                content,
                Task(
                    id=f"task-template-token-{index}",
                    session_key="session-1",
                    title=title,
                ),
                llm_call=llm_call,
                parse_json=lambda _raw, _fallback: {"score": 5},
            )

        self.assertEqual(len(observed_prompts), len(cases))
        for prompt, (title, content) in zip(observed_prompts, cases):
            self.assertIn(f"Task title: {title}", prompt)
            self.assertIn(f"Skill content (first 2500 chars):\n{content}", prompt)

    async def test_text_fallback_and_failures_preserve_the_default_score(self) -> None:
        _quality_prompt, score_skill_quality = _quality_symbols()
        task = Task(id="task-fallback", session_key="session-1")

        async def score_raw(
            raw: str,
            parsed: dict[str, Any],
        ) -> float:
            async def llm_call(
                _prompt: str,
                *,
                max_tokens: int = 1024,
                temperature: float = 0.1,
            ) -> str:
                return raw

            return await score_skill_quality(
                "content",
                task,
                llm_call=llm_call,
                parse_json=lambda _raw, fallback: parsed or fallback,
            )

        self.assertEqual(await score_raw("score: 8.5", {}), 8.5)
        self.assertEqual(
            await score_raw('{"score": "12.5"}', {"score": "12.5"}),
            10.0,
        )
        self.assertEqual(await score_raw("no score", {}), 5.0)

        async def failing_llm(
            _prompt: str,
            *,
            max_tokens: int = 1024,
            temperature: float = 0.1,
        ) -> str:
            raise RuntimeError("llm failed")

        self.assertEqual(
            await score_skill_quality(
                "content",
                task,
                llm_call=failing_llm,
                parse_json=lambda _raw, _fallback: {},
            ),
            5.0,
        )

        async def valid_llm(
            _prompt: str,
            *,
            max_tokens: int = 1024,
            temperature: float = 0.1,
        ) -> str:
            return "score: 9"

        def failing_parser(
            _raw: str,
            _fallback: dict[str, Any],
        ) -> dict[str, Any]:
            raise RuntimeError("parse failed")

        self.assertEqual(
            await score_skill_quality(
                "content",
                task,
                llm_call=valid_llm,
                parse_json=failing_parser,
            ),
            5.0,
        )

    async def test_compatibility_facade_uses_override_and_propagates_cancel(self) -> None:
        _quality_symbols()
        from mem.skill_evolver import MemSkillEvolver

        class CustomEvolver(MemSkillEvolver):
            def __init__(self, response: str | BaseException) -> None:
                super().__init__(store=None, embedder=None)  # type: ignore[arg-type]
                self.response = response
                self.calls: list[tuple[int, float]] = []

            async def _llm_call(
                self,
                prompt: str,
                *,
                max_tokens: int = 1024,
                temperature: float = 0.1,
            ) -> str:
                self.calls.append((max_tokens, temperature))
                if isinstance(self.response, BaseException):
                    raise self.response
                return self.response

        task = Task(id="task-compat", session_key="session-1", title="quality")
        evolver = CustomEvolver('{"score": 7.5}')

        self.assertEqual(await evolver._score_quality("content", task), 7.5)
        self.assertEqual(evolver.calls, [(512, 0)])

        cancelled = CustomEvolver(asyncio.CancelledError())
        with self.assertRaises(asyncio.CancelledError):
            await cancelled._score_quality("content", task)


if __name__ == "__main__":
    unittest.main()
