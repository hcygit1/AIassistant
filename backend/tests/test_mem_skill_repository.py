from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import closing
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from mem.models import Skill


def _create_skills_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """CREATE TABLE skills (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            description TEXT NOT NULL DEFAULT '',
            dir_path TEXT NOT NULL DEFAULT '',
            version INTEGER NOT NULL DEFAULT 1,
            status TEXT NOT NULL DEFAULT 'active',
            installed INTEGER NOT NULL DEFAULT 0,
            owner TEXT NOT NULL DEFAULT 'agent:main',
            visibility TEXT NOT NULL DEFAULT 'private',
            quality_score REAL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        )"""
    )


def _repository_type() -> type[Any]:
    try:
        from mem.skill_repository import SkillRepository
    except ModuleNotFoundError as exc:
        raise AssertionError(
            "mem.skill_repository should own skill persistence"
        ) from exc
    return SkillRepository


class SkillRepositoryTests(unittest.TestCase):
    def test_skill_persistence_and_fts_cleanup_have_explicit_owners(self) -> None:
        repository_path = BACKEND_DIR / "mem" / "skill_repository.py"

        self.assertTrue(
            repository_path.is_file(),
            "mem.skill_repository should own skill persistence",
        )
        store_source = (BACKEND_DIR / "mem" / "store.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("SkillRepository(", store_source)
        self.assertNotIn("INSERT OR REPLACE INTO skills", store_source)
        self.assertNotIn("SELECT * FROM skills WHERE id = ?", store_source)
        fts_source = (BACKEND_DIR / "mem" / "fts_index.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "rowid NOT IN (SELECT rowid FROM skills)",
            fts_source,
        )

    def test_writes_preserve_mutation_commit_fields_and_fts_contract(self) -> None:
        SkillRepository = _repository_type()
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "skills.db"
            connection = sqlite3.connect(path)
            connection.row_factory = sqlite3.Row
            _create_skills_table(connection)
            times = iter([100, 200, 300])
            synced: list[str] = []
            repository = SkillRepository(
                connection,
                sync_fts=synced.append,
                now_ms=lambda: next(times),
            )
            skill = Skill(
                id="skill-1",
                name="initial-skill",
                description="initial description",
                dir_path="skills/initial",
                version=2,
                status="draft",
                installed=1,
                owner="owner-a",
                visibility="shared",
                quality_score=0.5,
            )

            repository.insert(skill)
            with closing(sqlite3.connect(path)) as observer:
                committed_after_insert = observer.execute(
                    "SELECT COUNT(*) FROM skills"
                ).fetchone()[0]
            repository.update("skill-1")
            repository.update("skill-1", quality_score=0.8, installed=0)
            repository.update(
                "skill-1",
                name="updated-skill",
                description="updated description",
                version=3,
                status="active",
            )
            stored = repository.get("skill-1")
            with closing(sqlite3.connect(path)) as observer:
                committed_name = observer.execute(
                    "SELECT name FROM skills WHERE id='skill-1'"
                ).fetchone()[0]
            missing = repository.get("missing")
            connection.close()

        self.assertEqual((skill.created_at, skill.updated_at), (100, 100))
        self.assertEqual(committed_after_insert, 1)
        self.assertEqual(synced, ["skill-1", "skill-1"])
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertEqual(stored.name, "updated-skill")
        self.assertEqual(stored.description, "updated description")
        self.assertEqual(stored.dir_path, "skills/initial")
        self.assertEqual(stored.version, 3)
        self.assertEqual(stored.status, "active")
        self.assertEqual(stored.installed, 0)
        self.assertEqual(stored.owner, "owner-a")
        self.assertEqual(stored.visibility, "shared")
        self.assertEqual(stored.quality_score, 0.8)
        self.assertEqual(stored.created_at, 100)
        self.assertEqual(stored.updated_at, 300)
        self.assertEqual(committed_name, "updated-skill")
        self.assertIsNone(missing)

    def test_store_resolves_skill_dependencies_dynamically(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                textwrap.dedent(
                    """
                    import json
                    import tempfile
                    from pathlib import Path
                    from unittest.mock import patch
                    from mem.models import Skill
                    from mem.store import MemStore

                    with tempfile.TemporaryDirectory() as root:
                        store = MemStore(str(Path(root) / "mem.db"), dimensions=3)
                        with (
                            patch("mem.store._now_ms", return_value=123),
                            patch.object(store, "_sync_skill_fts") as sync_fts,
                        ):
                            skill = Skill(id="skill-1", name="skill-one")
                            store.insert_skill(skill)
                        stored = store.get_skill("skill-1")
                        payload = {
                            "createdAt": stored.created_at,
                            "updatedAt": stored.updated_at,
                            "syncCalls": sync_fts.call_args_list[0].args,
                        }
                        store.close()
                        print(json.dumps(payload, sort_keys=True))
                    """
                ),
            ],
            cwd=BACKEND_DIR,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads(result.stdout.splitlines()[-1]),
            {
                "createdAt": 123,
                "updatedAt": 123,
                "syncCalls": ["skill-1"],
            },
        )

    def test_same_id_replace_removes_previous_fts_content(self) -> None:
        payload = self._run_replace_probe(
            """
            store.insert_skill(Skill(
                id="skill-1", name="skill-one",
                description="legacyiddescription",
            ))
            store.insert_skill(Skill(
                id="skill-1", name="skill-one",
                description="currentiddescription",
            ))
            """,
            legacy_term="legacyiddescription",
            current_term="currentiddescription",
        )

        self.assertEqual(
            payload,
            {
                "currentMatches": 1,
                "ids": ["skill-1"],
                "ftsCount": 1,
                "legacyMatches": 0,
                "skillCount": 1,
            },
        )

    def test_same_name_replace_keeps_new_id_and_removes_previous_fts(self) -> None:
        payload = self._run_replace_probe(
            """
            store.insert_skill(Skill(
                id="skill-old", name="shared-skill",
                description="legacynamedescription",
            ))
            store.insert_skill(Skill(
                id="skill-new", name="shared-skill",
                description="currentnamedescription",
            ))
            """,
            legacy_term="legacynamedescription",
            current_term="currentnamedescription",
        )

        self.assertEqual(
            payload,
            {
                "currentMatches": 1,
                "ids": ["skill-new"],
                "ftsCount": 1,
                "legacyMatches": 0,
                "skillCount": 1,
            },
        )

    def _run_replace_probe(
        self,
        operations: str,
        *,
        legacy_term: str,
        current_term: str,
    ) -> dict[str, Any]:
        operations_source = textwrap.indent(
            textwrap.dedent(operations).strip(),
            "    ",
        )
        source = f"""import json
import tempfile
from pathlib import Path
from mem.models import Skill
from mem.store import MemStore

with tempfile.TemporaryDirectory() as root:
    store = MemStore(str(Path(root) / "mem.db"), dimensions=3)
{operations_source}
    payload = {{
        "skillCount": store._conn.execute(
            "SELECT COUNT(*) FROM skills"
        ).fetchone()[0],
        "ids": [row[0] for row in store._conn.execute(
            "SELECT id FROM skills ORDER BY id"
        )],
        "ftsCount": store._conn.execute(
            "SELECT COUNT(*) FROM skills_fts"
        ).fetchone()[0],
        "legacyMatches": store._conn.execute(
            "SELECT COUNT(*) FROM skills_fts "
            "WHERE skills_fts MATCH '{legacy_term}'"
        ).fetchone()[0],
        "currentMatches": store._conn.execute(
            "SELECT COUNT(*) FROM skills_fts "
            "WHERE skills_fts MATCH '{current_term}'"
        ).fetchone()[0],
    }}
    store.close()
    print(json.dumps(payload, sort_keys=True))
"""
        result = subprocess.run(
            [sys.executable, "-c", source],
            cwd=BACKEND_DIR,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout.splitlines()[-1])


if __name__ == "__main__":
    unittest.main()
