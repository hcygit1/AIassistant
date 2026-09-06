from __future__ import annotations

import unittest

from infra.run_tracker import RunTracker


class RunTrackerRetentionTests(unittest.TestCase):
    def test_completed_history_is_bounded(self) -> None:
        tracker = RunTracker(max_history=2)
        runs = [tracker.start_turn("main", f"session-{i}") for i in range(3)]
        for run in runs:
            tracker.complete_turn(run.run_id)

        self.assertEqual(
            [record.session_id for record in tracker._history],
            ["session-1", "session-2"],
        )

    def test_zero_history_keeps_usage_cache_empty(self) -> None:
        tracker = RunTracker(max_history=0)
        run = tracker.start_turn("main", "session-1")
        tracker.complete_turn(run.run_id)

        self.assertEqual(tracker._history, [])
        self.assertEqual(tracker.get_cumulative_usage("main"), {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "total_tokens": 0,
            "turns": 0,
        })


if __name__ == "__main__":
    unittest.main()
