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

from mem.models import Skill, Task


def _evaluation_symbols() -> tuple[Any, ...]:
    try:
        from mem.skill_evaluation import (
            CREATE_EVAL_PROMPT,
            UPGRADE_EVAL_PROMPT,
            CreateEvalResult,
            UpgradeEvalResult,
            evaluate_skill_creation,
            evaluate_skill_upgrade,
        )
    except ModuleNotFoundError as exc:
        raise AssertionError(
            "mem.skill_evaluation should own skill evaluation"
        ) from exc
    return (
        CREATE_EVAL_PROMPT,
        UPGRADE_EVAL_PROMPT,
        CreateEvalResult,
        UpgradeEvalResult,
        evaluate_skill_creation,
        evaluate_skill_upgrade,
    )


class SkillEvaluationTests(unittest.IsolatedAsyncioTestCase):
    def test_evaluation_logic_has_a_neutral_owner_and_compatible_exports(self) -> None:
        (
            create_prompt,
            upgrade_prompt,
            create_result,
            upgrade_result,
            _evaluate_creation,
            _evaluate_upgrade,
        ) = _evaluation_symbols()
        evaluation_path = BACKEND_DIR / "mem" / "skill_evaluation.py"
        self.assertTrue(
            evaluation_path.is_file(),
            "mem.skill_evaluation should own skill evaluation",
        )

        evolver_source = (BACKEND_DIR / "mem" / "skill_evolver.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("from mem.skill_evaluation import", evolver_source)
        self.assertNotIn("class CreateEvalResult", evolver_source)
        self.assertNotIn("class UpgradeEvalResult", evolver_source)
        self.assertNotIn("CREATE_EVAL_PROMPT =", evolver_source)
        self.assertNotIn("UPGRADE_EVAL_PROMPT =", evolver_source)

        from mem.skill_evolver import (
            CREATE_EVAL_PROMPT as compatible_create_prompt,
            UPGRADE_EVAL_PROMPT as compatible_upgrade_prompt,
            CreateEvalResult as CompatibleCreateResult,
            UpgradeEvalResult as CompatibleUpgradeResult,
        )

        self.assertIs(compatible_create_prompt, create_prompt)
        self.assertIs(compatible_upgrade_prompt, upgrade_prompt)
        self.assertIs(CompatibleCreateResult, create_result)
        self.assertIs(CompatibleUpgradeResult, upgrade_result)
        self.assertEqual(
            hashlib.sha256(create_prompt.encode()).hexdigest(),
            "a2f8afe6fd979ac97650db2477a0540069e5280c9a613ada7f9100e79546649d",
        )
        self.assertEqual(
            hashlib.sha256(upgrade_prompt.encode()).hexdigest(),
            "ee9c628fb2587c3f9efd9ff8fa5bf877710a6ad56189ee1528d8a813369882a8",
        )

    async def test_create_evaluation_preserves_prompt_mapping_and_defaults(self) -> None:
        (
            create_prompt,
            _upgrade_prompt,
            CreateEvalResult,
            _UpgradeEvalResult,
            evaluate_skill_creation,
            _evaluate_skill_upgrade,
        ) = _evaluation_symbols()
        task = Task(
            id="task-create",
            session_key="session-1",
            title="repair database",
            summary="s" * 3100,
        )
        calls: list[tuple[str, int, float]] = []

        async def llm_call(
            prompt: str,
            *,
            max_tokens: int = 1024,
            temperature: float = 0.1,
        ) -> str:
            calls.append((prompt, max_tokens, temperature))
            return "raw-create"

        def parse_json(raw: str, fallback: dict[str, Any]) -> dict[str, Any]:
            self.assertEqual(raw, "raw-create")
            self.assertEqual(fallback, {})
            return {
                "shouldGenerate": 1,
                "reason": 123,
                "suggestedName": None,
                "suggestedTags": ("database", "repair"),
                "confidence": "0.75",
            }

        result = await evaluate_skill_creation(
            task,
            llm_call=llm_call,
            parse_json=parse_json,
        )

        expected_prompt = (
            create_prompt
            .replace("{TITLE}", "repair database")
            .replace("{SUMMARY}", "s" * 3000)
        )
        self.assertEqual(calls, [(expected_prompt, 1024, 0)])
        self.assertEqual(
            result,
            CreateEvalResult(
                should_generate=True,
                reason="123",
                suggested_name="None",
                suggested_tags=("database", "repair"),
                confidence=0.75,
            ),
        )

        default_result = await evaluate_skill_creation(
            task,
            llm_call=llm_call,
            parse_json=lambda _raw, _fallback: {},
        )
        self.assertEqual(default_result, CreateEvalResult())

    async def test_upgrade_evaluation_preserves_prompt_mapping_and_defaults(self) -> None:
        (
            _create_prompt,
            upgrade_prompt,
            _CreateEvalResult,
            UpgradeEvalResult,
            _evaluate_skill_creation,
            evaluate_skill_upgrade,
        ) = _evaluation_symbols()
        task = Task(
            id="task-upgrade",
            session_key="session-1",
            title="extend database repair",
            summary="t" * 3100,
        )
        skill = Skill(
            id="skill-1",
            name="database-repair",
            description="d" * 1100,
            version=7,
        )
        calls: list[tuple[str, int, float]] = []

        async def llm_call(
            prompt: str,
            *,
            max_tokens: int = 1024,
            temperature: float = 0.1,
        ) -> str:
            calls.append((prompt, max_tokens, temperature))
            return "raw-upgrade"

        def parse_json(raw: str, fallback: dict[str, Any]) -> dict[str, Any]:
            self.assertEqual(raw, "raw-upgrade")
            self.assertEqual(fallback, {})
            return {
                "shouldUpgrade": "yes",
                "upgradeType": None,
                "dimensions": ("steps", "verification"),
                "reason": 456,
                "mergeStrategy": None,
                "confidence": 1,
            }

        result = await evaluate_skill_upgrade(
            task,
            skill,
            llm_call=llm_call,
            parse_json=parse_json,
        )

        expected_prompt = (
            upgrade_prompt
            .replace("{VERSION}", "7")
            .replace("{SKILL_NAME}", "database-repair")
            .replace("{SKILL_DESC}", "d" * 1000)
            .replace("{TITLE}", "extend database repair")
            .replace("{SUMMARY}", "t" * 3000)
        )
        self.assertEqual(calls, [(expected_prompt, 1024, 0)])
        self.assertEqual(
            result,
            UpgradeEvalResult(
                should_upgrade=True,
                upgrade_type="None",
                dimensions=("steps", "verification"),
                reason="456",
                merge_strategy="None",
                confidence=1.0,
            ),
        )

        default_result = await evaluate_skill_upgrade(
            task,
            skill,
            llm_call=llm_call,
            parse_json=lambda _raw, _fallback: {},
        )
        self.assertEqual(default_result, UpgradeEvalResult())

    async def test_create_evaluation_accepts_snake_case_and_string_boolean(self) -> None:
        (
            _create_prompt,
            _upgrade_prompt,
            CreateEvalResult,
            _UpgradeEvalResult,
            evaluate_skill_creation,
            _evaluate_skill_upgrade,
        ) = _evaluation_symbols()
        task = Task(
            id="task-snake-case",
            session_key="session-1",
            title="organize files",
            summary="A reusable file organization workflow.",
        )

        async def llm_call(*_args: Any, **_kwargs: Any) -> str:
            return "raw"

        result = await evaluate_skill_creation(
            task,
            llm_call=llm_call,
            parse_json=lambda _raw, _fallback: {
                "should_generate": "true",
                "suggested_name": "organize-files",
                "suggested_tags": ["files"],
                "reason": "reusable workflow",
                "confidence": "0.9",
            },
        )

        self.assertEqual(
            result,
            CreateEvalResult(
                should_generate=True,
                suggested_name="organize-files",
                suggested_tags=["files"],
                reason="reusable workflow",
                confidence=0.9,
            ),
        )

    def test_skill_evolver_json_parser_accepts_fenced_and_repairable_json(self) -> None:
        from mem.skill_evolver import _parse_json

        fenced = "```json\n{'shouldGenerate': true, 'confidence': 0.9,}\n```"
        parsed = _parse_json(fenced, {})

        self.assertEqual(parsed["shouldGenerate"], True)
        self.assertEqual(parsed["confidence"], 0.9)
        self.assertEqual(_parse_json("not-json", {"fallback": True}), {"fallback": True})

    async def test_compatibility_facade_uses_overridable_llm_and_error_fallbacks(
        self,
    ) -> None:
        _evaluation_symbols()
        from mem.skill_evolver import MemSkillEvolver

        class CustomEvolver(MemSkillEvolver):
            def __init__(self, responses: list[str | BaseException]) -> None:
                super().__init__(store=None, embedder=None)  # type: ignore[arg-type]
                self.responses = responses
                self.calls: list[tuple[int, float]] = []

            async def _llm_call(
                self,
                prompt: str,
                *,
                max_tokens: int = 1024,
                temperature: float = 0.1,
            ) -> str:
                self.calls.append((max_tokens, temperature))
                response = self.responses.pop(0)
                if isinstance(response, BaseException):
                    raise response
                return response

        task = Task(
            id="task-compat",
            session_key="session-1",
            title="repair database",
            summary="verified repair workflow",
        )
        skill = Skill(
            id="skill-compat",
            name="database-repair",
            description="repair workflow",
            version=2,
        )
        evolver = CustomEvolver(
            [
                '{"shouldGenerate": true, "confidence": 0.8}',
                '{"shouldUpgrade": true, "confidence": 0.9}',
            ]
        )

        create_result = await evolver._evaluate_create(task)
        upgrade_result = await evolver._evaluate_upgrade(task, skill)

        self.assertTrue(create_result.should_generate)
        self.assertEqual(create_result.confidence, 0.8)
        self.assertTrue(upgrade_result.should_upgrade)
        self.assertEqual(upgrade_result.confidence, 0.9)
        self.assertEqual(evolver.calls, [(1024, 0), (1024, 0)])

        failing = CustomEvolver(
            [RuntimeError("create failed"), RuntimeError("upgrade failed")]
        )
        with self.assertLogs("mem.skill_evolver", level="WARNING") as logs:
            failed_create = await failing._evaluate_create(task)
            failed_upgrade = await failing._evaluate_upgrade(task, skill)
        self.assertEqual(failed_create.reason, "error: create failed")
        self.assertEqual(failed_upgrade.reason, "error: upgrade failed")
        self.assertEqual(
            logs.output,
            [
                "WARNING:mem.skill_evolver:Skill create eval failed: create failed",
                "WARNING:mem.skill_evolver:Skill upgrade eval failed: upgrade failed",
            ],
        )

        cancelled = CustomEvolver([asyncio.CancelledError()])
        with self.assertRaises(asyncio.CancelledError):
            await cancelled._evaluate_create(task)

        cancelled_upgrade = CustomEvolver([asyncio.CancelledError()])
        with self.assertRaises(asyncio.CancelledError):
            await cancelled_upgrade._evaluate_upgrade(task, skill)

    async def test_conversion_errors_stay_inside_the_compatibility_facade(self) -> None:
        _evaluation_symbols()
        from mem.skill_evolver import MemSkillEvolver

        class InvalidConfidenceEvolver(MemSkillEvolver):
            async def _llm_call(
                self,
                prompt: str,
                *,
                max_tokens: int = 1024,
                temperature: float = 0.1,
            ) -> str:
                return '{"confidence": "not-a-number"}'

        evolver = InvalidConfidenceEvolver(
            store=None,
            embedder=None,
        )  # type: ignore[arg-type]
        task = Task(id="task-invalid", session_key="session-1")

        with self.assertLogs("mem.skill_evolver", level="WARNING"):
            result = await evolver._evaluate_create(task)

        self.assertTrue(result.reason.startswith("error: "))
        self.assertEqual(result.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
