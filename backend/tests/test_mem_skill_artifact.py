from __future__ import annotations

import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from mem.models import Skill, Task
from mem.skill_evaluation import CreateEvalResult


def _artifact_symbols() -> tuple[Any, Any, Any]:
    try:
        from mem.skill_artifact import (
            build_new_skill,
            extract_skill_description,
            write_skill_file,
        )
    except ModuleNotFoundError as exc:
        raise AssertionError(
            "mem.skill_artifact should own new skill artifacts"
        ) from exc
    return extract_skill_description, write_skill_file, build_new_skill


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


class SkillArtifactTests(unittest.IsolatedAsyncioTestCase):
    def test_artifact_logic_has_a_neutral_owner_and_compatible_description(self) -> None:
        extract_description, _write_skill_file, _build_new_skill = _artifact_symbols()
        artifact_path = BACKEND_DIR / "mem" / "skill_artifact.py"
        self.assertTrue(
            artifact_path.is_file(),
            "mem.skill_artifact should own new skill artifacts",
        )

        evolver_source = (BACKEND_DIR / "mem" / "skill_evolver.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("from mem.skill_artifact import", evolver_source)
        self.assertNotIn("def _extract_description", evolver_source)
        self.assertNotIn("write_text(skill_content", evolver_source)
        self.assertNotIn("skill = Skill(", evolver_source)

        from mem.skill_evolver import _extract_description

        self.assertIs(_extract_description, extract_description)

    def test_description_parsing_matches_the_legacy_formats(self) -> None:
        extract_description, _write_skill_file, _build_new_skill = _artifact_symbols()

        cases = (
            ('description: "Quoted description"', "Quoted description"),
            ("description: 'Single quoted'", "Single quoted"),
            ("description: plain description  ", "plain description"),
            (
                'description: "first"\ndescription: "second"',
                "first",
            ),
            ("name: no-description", ""),
        )
        for content, expected in cases:
            with self.subTest(content=content):
                self.assertEqual(extract_description(content), expected)

    def test_file_writer_preserves_no_directory_and_real_file_behavior(self) -> None:
        _extract_description, write_skill_file, _build_new_skill = _artifact_symbols()
        path_calls: list[str] = []

        def path_provider(value: str) -> Path:
            path_calls.append(str(value))
            return Path(value)

        self.assertEqual(
            write_skill_file(
                "",
                "unused",
                "content",
                path_provider=path_provider,
            ),
            "",
        )
        self.assertEqual(path_calls, [])

        with tempfile.TemporaryDirectory() as root:
            skill_dir = write_skill_file(
                root,
                "generated-skill",
                "skill content",
                path_provider=path_provider,
            )

            expected_dir = str(Path(root) / "generated-skill")
            self.assertEqual(skill_dir, expected_dir)
            self.assertEqual(
                (Path(expected_dir) / "SKILL.md").read_text(encoding="utf-8"),
                "skill content",
            )
            self.assertEqual(
                path_calls,
                [root, expected_dir, expected_dir],
            )

    def test_model_builder_preserves_status_owner_and_defaults(self) -> None:
        _extract_description, _write_skill_file, build_new_skill = _artifact_symbols()
        task = Task(
            id="task-model",
            session_key="session-1",
            owner="",
        )

        active = build_new_skill(
            skill_id="skill-active",
            name="active-skill",
            description="active description",
            skill_dir="/skills/active-skill",
            task=task,
            quality_score=6.0,
        )
        draft = build_new_skill(
            skill_id="skill-draft",
            name="draft-skill",
            description="draft description",
            skill_dir="",
            task=Task(
                id="task-other-owner",
                session_key="session-1",
                owner="agent:other",
            ),
            quality_score=5.99,
        )

        self.assertEqual(
            active,
            Skill(
                id="skill-active",
                name="active-skill",
                description="active description",
                dir_path="/skills/active-skill",
                version=1,
                status="active",
                installed=0,
                owner="agent:main",
                visibility="private",
                quality_score=6.0,
            ),
        )
        self.assertEqual(draft.status, "draft")
        self.assertEqual(draft.owner, "agent:other")
        self.assertEqual(draft.visibility, "private")
        self.assertEqual(draft.installed, 0)

    async def test_facade_preserves_uuid_description_and_path_patch_points(self) -> None:
        _artifact_symbols()
        from mem.skill_evolver import MemSkillEvolver

        store = _FakeStore()
        path_calls: list[str] = []
        fixed_uuid = uuid.UUID("12345678-1234-5678-1234-567812345678")
        skill_content = (
            '---\nname: "patched-skill"\ndescription: "Original"\n---\n'
            "# Skill\n\nA complete reusable workflow with verification steps."
        )

        class ArtifactEvolver(MemSkillEvolver):
            async def _llm_call(
                self,
                prompt: str,
                *,
                max_tokens: int = 1024,
                temperature: float = 0.1,
            ) -> str:
                return skill_content

            async def _score_quality(self, content: str, task: Task) -> float:
                return 7.0

        with tempfile.TemporaryDirectory() as root:
            evolver = ArtifactEvolver(
                store=store,
                embedder=_FakeEmbedder(),
                skill_store_dir=root,
            )
            real_path = Path

            def patched_path(value: str) -> Path:
                path_calls.append(str(value))
                return real_path(value)

            with (
                patch("mem.skill_evolver.uuid.uuid4", return_value=fixed_uuid),
                patch(
                    "mem.skill_evolver._extract_description",
                    return_value="Patched description",
                ),
                patch("mem.skill_evolver.Path", side_effect=patched_path),
            ):
                skill = await evolver._generate_skill(
                    Task(
                        id="task-patches",
                        session_key="session-1",
                        title="Generate",
                        summary="summary",
                    ),
                    [],
                    CreateEvalResult(suggested_name="patched-skill"),
                )

            expected_dir = str(Path(root) / "patched-skill")
            self.assertIsNotNone(skill)
            self.assertEqual(skill.id, str(fixed_uuid))
            self.assertEqual(skill.description, "Patched description")
            self.assertEqual(skill.dir_path, expected_dir)
            self.assertEqual(path_calls, [root, expected_dir, expected_dir])
            self.assertEqual(store.inserted, [skill])
            self.assertEqual(
                (Path(expected_dir) / "SKILL.md").read_text(encoding="utf-8"),
                skill_content,
            )

    async def test_file_failure_precedes_quality_and_store(self) -> None:
        _artifact_symbols()
        from mem.skill_evolver import MemSkillEvolver

        store = _FakeStore()
        events: list[str] = []
        skill_content = (
            'description: "Description"\n'
            "# Skill\n\nA complete reusable workflow with verification steps."
        )

        class FailureEvolver(MemSkillEvolver):
            async def _llm_call(
                self,
                prompt: str,
                *,
                max_tokens: int = 1024,
                temperature: float = 0.1,
            ) -> str:
                return skill_content

            async def _score_quality(self, content: str, task: Task) -> float:
                events.append("quality")
                return 7.0

        with tempfile.TemporaryDirectory() as root:
            file_path = Path(root) / "not-a-directory"
            file_path.write_text("occupied", encoding="utf-8")
            evolver = FailureEvolver(
                store=store,
                embedder=_FakeEmbedder(),
                skill_store_dir=str(file_path),
            )

            with self.assertRaises((FileExistsError, NotADirectoryError)):
                await evolver._generate_skill(
                    Task(id="task-file-failure", session_key="session-1"),
                    [],
                    CreateEvalResult(suggested_name="failed-skill"),
                )

        self.assertEqual(events, [])
        self.assertEqual(store.inserted, [])
        self.assertEqual(store.embeddings, [])

    async def test_facade_preserves_dynamic_skill_factory_patch_path(self) -> None:
        _artifact_symbols()
        from mem.skill_evolver import MemSkillEvolver

        store = _FakeStore()
        constructed: list[dict[str, Any]] = []
        skill_content = (
            'description: "Description"\n'
            "# Skill\n\nA complete reusable workflow with verification steps."
        )

        class FactoryEvolver(MemSkillEvolver):
            async def _llm_call(
                self,
                prompt: str,
                *,
                max_tokens: int = 1024,
                temperature: float = 0.1,
            ) -> str:
                return skill_content

            async def _score_quality(self, content: str, task: Task) -> float:
                return 7.0

        def skill_factory(**fields: Any) -> Skill:
            constructed.append(fields)
            return Skill(**fields)

        evolver = FactoryEvolver(store=store, embedder=_FakeEmbedder())
        with patch("mem.skill_evolver.Skill", side_effect=skill_factory):
            skill = await evolver._generate_skill(
                Task(id="task-factory", session_key="session-1"),
                [],
                CreateEvalResult(suggested_name="factory-skill"),
            )

        self.assertIsNotNone(skill)
        self.assertEqual(len(constructed), 1)
        self.assertEqual(constructed[0]["name"], "factory-skill")
        self.assertEqual(store.inserted, [skill])

    async def test_quality_failure_precedes_model_store_and_embedding(self) -> None:
        _artifact_symbols()
        from mem.skill_evolver import MemSkillEvolver

        store = _FakeStore()
        embedder = _FakeEmbedder()
        constructed = 0
        skill_content = (
            'description: "Description"\n'
            "# Skill\n\nA complete reusable workflow with verification steps."
        )

        class QualityFailureEvolver(MemSkillEvolver):
            async def _llm_call(
                self,
                prompt: str,
                *,
                max_tokens: int = 1024,
                temperature: float = 0.1,
            ) -> str:
                return skill_content

            async def _score_quality(self, content: str, task: Task) -> float:
                raise RuntimeError("quality failed")

        def skill_factory(**fields: Any) -> Skill:
            nonlocal constructed
            constructed += 1
            return Skill(**fields)

        evolver = QualityFailureEvolver(store=store, embedder=embedder)
        with (
            patch("mem.skill_evolver.Skill", side_effect=skill_factory),
            self.assertRaisesRegex(RuntimeError, "quality failed"),
        ):
            await evolver._generate_skill(
                Task(id="task-quality-failure", session_key="session-1"),
                [],
                CreateEvalResult(suggested_name="quality-failure-skill"),
            )

        self.assertEqual(constructed, 0)
        self.assertEqual(store.inserted, [])
        self.assertEqual(store.embeddings, [])
        self.assertEqual(embedder.queries, [])


if __name__ == "__main__":
    unittest.main()
