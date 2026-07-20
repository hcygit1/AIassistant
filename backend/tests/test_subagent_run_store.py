from __future__ import annotations

import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from subagents.subagent_run_store import SubagentRunStore


class SubagentRunStoreTests(unittest.TestCase):
    def test_restore_and_persist_use_isolated_snapshots(self) -> None:
        original = {"run-1": object()}
        saved: list[dict[str, object]] = []
        store = SubagentRunStore(
            load_runs=lambda: original,
            save_runs=lambda runs: saved.append(dict(runs)),
        )

        store.restore()
        original.clear()
        with store.locked_records() as records:
            records["run-2"] = object()
        store.persist()

        self.assertEqual(set(store.records), {"run-1", "run-2"})
        self.assertEqual(set(saved[0]), {"run-1", "run-2"})

    def test_persist_holds_lock_until_snapshot_is_saved(self) -> None:
        lock_owned: list[bool] = []
        store: SubagentRunStore

        def save_runs(_runs: dict[str, object]) -> None:
            lock_owned.append(store.lock._is_owned())

        store = SubagentRunStore(
            load_runs=lambda: {},
            save_runs=save_runs,
        )

        store.persist()

        self.assertEqual(lock_owned, [True])


if __name__ == "__main__":
    unittest.main()
