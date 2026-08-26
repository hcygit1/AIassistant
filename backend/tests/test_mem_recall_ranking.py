from __future__ import annotations

import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


class MemRecallRankingTests(unittest.TestCase):
    def test_global_ranking_allows_later_task_chunk_into_top_k(self) -> None:
        from mem.recall import RecallHit, _rank_chunk_candidates

        ranked = _rank_chunk_candidates([
            RecallHit("task-1-low", 0.010, task_id="task-1"),
            RecallHit("task-1-high", 0.020, task_id="task-1"),
            RecallHit("task-2-high", 0.030, task_id="task-2"),
        ])

        self.assertEqual(ranked[0].chunk_id, "task-2-high")

    def test_orphan_competes_with_task_chunks_when_fallback_runs(self) -> None:
        from mem.recall import RecallHit, _rank_chunk_candidates

        ranked = _rank_chunk_candidates([
            RecallHit("task-chunk", 0.020, task_id="task-1"),
            RecallHit("orphan", 0.025),
        ])

        self.assertEqual([hit.chunk_id for hit in ranked], ["orphan", "task-chunk"])

    def test_final_limit_counts_only_chunks(self) -> None:
        from mem.recall import RecallHit, _trim_ranked_hits

        selected = _trim_ranked_hits(
            [RecallHit(f"chunk-{index}", 1.0 - index / 10) for index in range(8)],
            max_results=5,
            budget_chars=1000,
        )

        self.assertEqual(len(selected), 5)


if __name__ == "__main__":
    unittest.main()
