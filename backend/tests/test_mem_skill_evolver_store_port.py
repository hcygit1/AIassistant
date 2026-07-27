from __future__ import annotations

import ast
import inspect
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any, get_type_hints


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def _skill_evolver_store_port() -> type[Any]:
    try:
        from mem.skill_evolver_store import MemSkillEvolverStore
    except ModuleNotFoundError as exc:
        raise AssertionError(
            "mem.skill_evolver_store should own the skill evolver storage port"
        ) from exc
    return MemSkillEvolverStore


class MemSkillEvolverStorePortTests(unittest.TestCase):
    def test_importing_skill_evolver_does_not_load_store_or_sqlite_vec(
        self,
    ) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import mem.skill_evolver; "
                    "blocked = sorted(name for name in sys.modules "
                    "if name in {'mem.store', 'sqlite_vec'}); "
                    "assert not blocked, blocked"
                ),
            ],
            cwd=BACKEND_DIR,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_skill_evolver_store_port_matches_consumed_store_operations(
        self,
    ) -> None:
        MemSkillEvolverStore = _skill_evolver_store_port()
        expected = {
            "ann_search_skills",
            "fts_search_skills",
            "get_chunks_by_task",
            "get_skill",
            "insert_skill",
            "update_skill",
            "upsert_skill_embedding",
        }
        operations = {
            name
            for name, value in vars(MemSkillEvolverStore).items()
            if not name.startswith("_") and callable(value)
        }

        self.assertTrue(getattr(MemSkillEvolverStore, "_is_protocol", False))
        self.assertEqual(operations, expected)

        from mem.store import MemStore

        for name in expected:
            self.assertEqual(
                str(inspect.signature(getattr(MemSkillEvolverStore, name))),
                str(inspect.signature(getattr(MemStore, name))),
                name,
            )

    def test_skill_evolver_depends_on_port_without_constructor_changes(
        self,
    ) -> None:
        evolver_tree = ast.parse(
            (BACKEND_DIR / "mem" / "skill_evolver.py").read_text(
                encoding="utf-8"
            )
        )
        imports = {
            node.module: {alias.name for alias in node.names}
            for node in ast.walk(evolver_tree)
            if isinstance(node, ast.ImportFrom)
        }

        self.assertNotIn("mem.store", imports)
        self.assertIn(
            "MemSkillEvolverStore",
            imports.get("mem.skill_evolver_store", set()),
        )

        from mem.skill_evolver import MemSkillEvolver

        signature = inspect.signature(MemSkillEvolver.__init__)
        self.assertEqual(
            list(signature.parameters),
            [
                "self",
                "store",
                "embedder",
                "llm_base_url",
                "llm_api_key",
                "llm_model",
                "skill_store_dir",
                "min_chunks_for_eval",
                "min_confidence",
                "auto_install",
                "enabled",
            ],
        )
        self.assertEqual(
            signature.parameters["store"].annotation,
            "MemSkillEvolverStore",
        )

    def test_factory_store_annotation_resolves_to_port(self) -> None:
        MemSkillEvolverStore = _skill_evolver_store_port()
        from mem.skill_evolver import MemSkillEvolver

        hints = get_type_hints(MemSkillEvolver.from_config)

        self.assertIs(hints["store"], MemSkillEvolverStore)


if __name__ == "__main__":
    unittest.main()
