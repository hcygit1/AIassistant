from __future__ import annotations

import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import subagents.subagent_registry_state as registry_state
from subagents.subagent_run_model import SubagentRunRecord


class SubagentRegistryStateTests(unittest.TestCase):
    @staticmethod
    def _record() -> SubagentRunRecord:
        return SubagentRunRecord(
            run_id="run-1",
            child_session_key="child",
            requester_session_key="requester",
            requester_agent_id="main",
            target_agent_id="worker",
            task="work",
        )

    def test_failed_replace_preserves_existing_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runs.json"
            original = '{"version": 2, "runs": {"existing": {}}}'
            path.write_text(original, encoding="utf-8")
            record = self._record()

            with (
                patch.object(registry_state, "_registry_path", return_value=path),
                patch.object(os, "replace", side_effect=OSError("replace failed")),
            ):
                registry_state.save_registry_to_disk({record.run_id: record})

            self.assertEqual(path.read_text(encoding="utf-8"), original)
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_failed_serialization_preserves_existing_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runs.json"
            original = '{"version": 2, "runs": {}}'
            path.write_text(original, encoding="utf-8")

            with (
                patch.object(registry_state, "_registry_path", return_value=path),
                patch.object(
                    registry_state.json,
                    "dump",
                    side_effect=OSError("write failed"),
                ),
            ):
                registry_state.save_registry_to_disk(
                    {"run-1": self._record()}
                )

            self.assertEqual(path.read_text(encoding="utf-8"), original)
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_interrupted_serialization_cleans_temp_and_propagates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runs.json"
            original = '{"version": 2, "runs": {}}'
            path.write_text(original, encoding="utf-8")

            with (
                patch.object(registry_state, "_registry_path", return_value=path),
                patch.object(
                    registry_state.json,
                    "dump",
                    side_effect=KeyboardInterrupt,
                ),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    registry_state.save_registry_to_disk(
                        {"run-1": self._record()}
                    )

            self.assertEqual(path.read_text(encoding="utf-8"), original)
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_successful_save_round_trips_and_preserves_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runs.json"
            path.write_text('{"version": 2, "runs": {}}', encoding="utf-8")
            path.chmod(0o640)

            with patch.object(
                registry_state,
                "_registry_path",
                return_value=path,
            ):
                registry_state.save_registry_to_disk(
                    {"run-1": self._record()}
                )
                loaded = registry_state.load_registry_from_disk()

            self.assertEqual(set(loaded), {"run-1"})
            self.assertEqual(loaded["run-1"].task, "work")
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o640)
            self.assertEqual(list(path.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
