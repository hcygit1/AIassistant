from __future__ import annotations

import asyncio
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from mem.models import Skill, Task
from mem.skill_evaluation import UpgradeEvalResult


def _upgrade_symbols() -> tuple[Any, ...]:
    try:
        from mem.skill_upgrade import (
            SKILL_UPGRADE_PROMPT,
            SkillUpgradeOutcome,
            build_skill_upgrade_prompt,
            execute_skill_upgrade,
        )
    except ModuleNotFoundError as exc:
        raise AssertionError("mem.skill_upgrade should own skill upgrades") from exc
    return (
        SKILL_UPGRADE_PROMPT,
        SkillUpgradeOutcome,
        build_skill_upgrade_prompt,
        execute_skill_upgrade,
    )


def _skill(*, dir_path: str = "", description: str = "existing description") -> Skill:
    return Skill(
        id="skill-1",
        name="database-repair",
        description=description,
        dir_path=dir_path,
        version=3,
        owner="agent:main",
    )


def _task() -> Task:
    return Task(
        id="task-1",
        session_key="session-1",
        title="extend repair",
        summary="new repair evidence",
    )


def _evaluation() -> UpgradeEvalResult:
    return UpgradeEvalResult(
        should_upgrade=True,
        upgrade_type="extend",
        merge_strategy="append verified steps",
        confidence=0.9,
    )


class _FakeStore:
    def __init__(self, *, update_error: Exception | None = None) -> None:
        self.update_error = update_error
        self.events: list[tuple[Any, ...]] = []

    def update_skill(self, skill_id: str, **fields: Any) -> None:
        self.events.append(("update", skill_id, fields))
        if self.update_error is not None:
            raise self.update_error

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
        return [0.3, 0.4]


class SkillUpgradeTests(unittest.IsolatedAsyncioTestCase):
    def test_upgrade_has_a_neutral_owner_and_compatible_prompt_export(self) -> None:
        prompt, _Outcome, _build_prompt, execute_upgrade = _upgrade_symbols()
        upgrade_path = BACKEND_DIR / "mem" / "skill_upgrade.py"
        self.assertTrue(upgrade_path.is_file(), "mem.skill_upgrade should own upgrades")

        evolver_source = (BACKEND_DIR / "mem" / "skill_evolver.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("from mem.skill_upgrade import", evolver_source)
        self.assertNotIn("SKILL_UPGRADE_PROMPT =", evolver_source)
        upgrade_body = evolver_source.split("async def _upgrade_skill(", 1)[1].split(
            "# LLM helper", 1
        )[0]
        self.assertIn("await execute_skill_upgrade(", upgrade_body)
        self.assertNotIn("read_text", upgrade_body)
        self.assertNotIn("update_skill", upgrade_body)

        from mem.skill_evolver import SKILL_UPGRADE_PROMPT as compatible_prompt

        self.assertIs(compatible_prompt, prompt)
        self.assertEqual(len(prompt), 463)
        self.assertEqual(
            hashlib.sha256(prompt.encode()).hexdigest(),
            "88b85998833ff998339ba07cbd2220c30d22ec07b2c48790f6a019dce85fc98d",
        )
        self.assertTrue(callable(execute_upgrade))

    def test_prompt_keeps_truncation_and_sequential_replacement(self) -> None:
        prompt_template, _Outcome, build_prompt, _execute_upgrade = _upgrade_symbols()
        task = Task(
            id="task-prompt",
            session_key="session-1",
            title="{SUMMARY}",
            summary="s" * 2100,
        )
        skill = Skill(
            id="skill-prompt",
            name="{SKILL_CONTENT}",
            description="description",
        )
        evaluation = UpgradeEvalResult(
            upgrade_type="{MERGE_STRATEGY}",
            merge_strategy="final-strategy",
        )

        prompt = build_prompt(
            task,
            skill,
            evaluation,
            existing_content="c" * 4100,
        )

        expected = (
            prompt_template
            .replace("{SKILL_NAME}", "{SKILL_CONTENT}")
            .replace("{SKILL_CONTENT}", "c" * 4000)
            .replace("{TITLE}", "{SUMMARY}")
            .replace("{SUMMARY}", "s" * 2000)
            .replace("{UPGRADE_TYPE}", "{MERGE_STRATEGY}")
            .replace("{MERGE_STRATEGY}", "final-strategy")
        )
        self.assertEqual(prompt, expected)

    async def test_existing_file_drives_prompt_and_successful_side_effects(self) -> None:
        _prompt, Outcome, _build_prompt, execute_upgrade = _upgrade_symbols()
        store = _FakeStore()
        llm_calls: list[tuple[str, int, float]] = []
        path_calls: list[str] = []
        new_content = (
            'description: "upgraded description"\n'
            "# Upgraded Skill\n\nVerified workflow with complete steps and checks."
        )

        async def llm_call(
            prompt: str,
            *,
            max_tokens: int = 1024,
            temperature: float = 0.1,
        ) -> str:
            llm_calls.append((prompt, max_tokens, temperature))
            return new_content

        def path_provider(value: str) -> Path:
            path_calls.append(str(value))
            return Path(value)

        with tempfile.TemporaryDirectory() as root:
            skill_file = Path(root) / "SKILL.md"
            skill_file.write_text("existing file content", encoding="utf-8")
            skill = _skill(dir_path=root, description="fallback description")

            outcome = await execute_upgrade(
                _task(),
                skill,
                _evaluation(),
                store=store,
                embedder=_FakeEmbedder(store.events),
                llm_call=llm_call,
                extract_description=lambda content: "upgraded description",
                path_provider=path_provider,
            )

            self.assertEqual(skill_file.read_text(encoding="utf-8"), new_content)
            self.assertEqual(path_calls, [root, root, root])

        self.assertEqual(outcome, Outcome(upgraded=True, version=4))
        self.assertEqual(llm_calls[0][1:], (4096, 0.2))
        self.assertIn("existing file content", llm_calls[0][0])
        self.assertNotIn("fallback description", llm_calls[0][0])
        self.assertEqual(
            store.events,
            [
                (
                    "update",
                    "skill-1",
                    {"description": "upgraded description", "version": 4},
                ),
                ("embed", "database-repair upgraded description"),
                ("upsert", "skill-1", [0.3, 0.4]),
            ],
        )

    async def test_description_fallback_and_short_or_failed_llm_have_no_side_effects(self) -> None:
        _prompt, Outcome, _build_prompt, execute_upgrade = _upgrade_symbols()
        store = _FakeStore()
        prompts: list[str] = []

        async def short_llm(
            prompt: str,
            *,
            max_tokens: int = 1024,
            temperature: float = 0.1,
        ) -> str:
            prompts.append(prompt)
            return "short"

        short_outcome = await execute_upgrade(
            _task(),
            _skill(description="fallback description"),
            _evaluation(),
            store=store,
            embedder=_FakeEmbedder(store.events),
            llm_call=short_llm,
            extract_description=lambda _content: "unused",
            path_provider=Path,
        )
        self.assertEqual(short_outcome, Outcome())
        self.assertIn("fallback description", prompts[0])
        self.assertEqual(store.events, [])

        llm_error = RuntimeError("llm failed")

        async def failed_llm(
            _prompt: str,
            *,
            max_tokens: int = 1024,
            temperature: float = 0.1,
        ) -> str:
            raise llm_error

        failed_outcome = await execute_upgrade(
            _task(),
            _skill(),
            _evaluation(),
            store=store,
            embedder=_FakeEmbedder(store.events),
            llm_call=failed_llm,
            extract_description=lambda _content: "unused",
            path_provider=Path,
        )
        self.assertIs(failed_outcome.llm_error, llm_error)
        self.assertFalse(failed_outcome.upgraded)
        self.assertEqual(store.events, [])

    async def test_file_write_and_update_failure_precede_embedding(self) -> None:
        _prompt, _Outcome, _build_prompt, execute_upgrade = _upgrade_symbols()
        update_error = RuntimeError("update failed")
        store = _FakeStore(update_error=update_error)
        new_content = "description: upgraded\n" + "x" * 60

        async def llm_call(
            _prompt: str,
            *,
            max_tokens: int = 1024,
            temperature: float = 0.1,
        ) -> str:
            return new_content

        with tempfile.TemporaryDirectory() as root:
            skill_file = Path(root) / "SKILL.md"
            skill_file.write_text("old content", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "update failed"):
                await execute_upgrade(
                    _task(),
                    _skill(dir_path=root),
                    _evaluation(),
                    store=store,
                    embedder=_FakeEmbedder(store.events),
                    llm_call=llm_call,
                    extract_description=lambda _content: "upgraded",
                    path_provider=Path,
                )

            self.assertEqual(skill_file.read_text(encoding="utf-8"), new_content)

        self.assertEqual(store.events[0][0], "update")
        self.assertEqual(len(store.events), 1)

    async def test_embedding_failure_is_silent_but_cancellation_propagates(self) -> None:
        _prompt, Outcome, _build_prompt, execute_upgrade = _upgrade_symbols()

        async def llm_call(
            _prompt: str,
            *,
            max_tokens: int = 1024,
            temperature: float = 0.1,
        ) -> str:
            return "description: upgraded\n" + "x" * 60

        failed_store = _FakeStore()
        failed_outcome = await execute_upgrade(
            _task(),
            _skill(),
            _evaluation(),
            store=failed_store,
            embedder=_FakeEmbedder(failed_store.events, RuntimeError("embed failed")),
            llm_call=llm_call,
            extract_description=lambda _content: "upgraded",
            path_provider=Path,
        )
        self.assertEqual(failed_outcome, Outcome(upgraded=True, version=4))
        self.assertEqual(failed_store.events[-1], ("embed", "database-repair upgraded"))

        cancelled_store = _FakeStore()
        with self.assertRaises(asyncio.CancelledError):
            await execute_upgrade(
                _task(),
                _skill(),
                _evaluation(),
                store=cancelled_store,
                embedder=_FakeEmbedder(cancelled_store.events, asyncio.CancelledError()),
                llm_call=llm_call,
                extract_description=lambda _content: "upgraded",
                path_provider=Path,
            )
        self.assertEqual(cancelled_store.events[0][0], "update")

    async def test_evolver_facade_keeps_dynamic_dependencies_and_logs(self) -> None:
        prompt, Outcome, _build_prompt, _execute_upgrade = _upgrade_symbols()
        from mem.skill_evolver import MemSkillEvolver

        evolver = MemSkillEvolver(store=_FakeStore(), embedder=_FakeEmbedder([]))
        execute_mock = AsyncMock(
            side_effect=(
                Outcome(llm_error=RuntimeError("llm failed")),
                Outcome(upgraded=True, version=4),
                Outcome(),
            )
        )
        real_path = Path
        path_calls: list[str] = []

        def patched_path(value: str) -> Path:
            path_calls.append(str(value))
            return real_path(value)

        def patched_extract(_content: str) -> str:
            return "patched"

        with (
            patch("mem.skill_evolver.execute_skill_upgrade", execute_mock),
            patch("mem.skill_evolver.SKILL_UPGRADE_PROMPT", "patched prompt"),
            patch("mem.skill_evolver._extract_description", patched_extract),
            patch("mem.skill_evolver.Path", side_effect=patched_path),
            patch("mem.skill_evolver.logger.error") as error_log,
            patch("mem.skill_evolver.logger.info") as info_log,
        ):
            for _ in range(3):
                await evolver._upgrade_skill(_task(), _skill(), _evaluation())

            first_kwargs = execute_mock.await_args_list[0].kwargs
            self.assertEqual(first_kwargs["prompt_template"], "patched prompt")
            self.assertIs(first_kwargs["extract_description"], patched_extract)
            self.assertEqual(first_kwargs["path_provider"]("path"), real_path("path"))

        self.assertEqual(path_calls, ["path"])
        error_log.assert_called_once()
        self.assertEqual(error_log.call_args.args[0], "Skill upgrade LLM call failed: %s")
        info_log.assert_called_once_with("Skill '%s' upgraded to v%d", "database-repair", 4)
        self.assertEqual(prompt.startswith("You are upgrading"), True)


if __name__ == "__main__":
    unittest.main()
