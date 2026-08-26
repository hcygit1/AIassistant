from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mem.skill_version_store import SkillVersionStore


class SkillVersionStoreTests(unittest.TestCase):
    def test_candidate_is_isolated_and_accepted_candidate_can_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = SkillVersionStore(
                evolution_root=root / "evolution", active_root=root / "skills",
            )

            candidate = store.create_candidate(
                skill_id="organize-files", content="---\nname: organize-files\n---\n", 
                source="skilllearnbench", reason="initial candidate",
            )

            self.assertEqual(candidate.version, 1)
            self.assertTrue((root / "evolution/organize-files/candidates/v1/SKILL.md").is_file())
            self.assertFalse((root / "skills/organize-files/SKILL.md").exists())
            with self.assertRaises(ValueError):
                store.publish(candidate)

            accepted = store.set_candidate_status(candidate, "accepted")
            store.publish(accepted)
            self.assertEqual(store.active_version("organize-files"), 1)
            self.assertTrue((root / "skills/organize-files/versions/v1/SKILL.md").is_file())

            second = store.create_candidate(
                skill_id="organize-files", content="---\nname: organize-files\ndescription: x\n---\n" + "x" * 50,
                source="test", reason="second",
            )
            store.publish(store.set_candidate_status(second, "accepted"))
            self.assertEqual(store.rollback("organize-files"), 1)
            self.assertEqual(store.active_version("organize-files"), 1)
