from __future__ import annotations

import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


class LoCoMoMetricsTests(unittest.TestCase):
    def test_recall_and_mrr_use_independent_golden_evidence(self) -> None:
        from evaluation.locomo_runner import evaluate_case

        result = evaluate_case(
            retrieved_ids=["wrong", "D1:2", "D1:1"],
            golden_ids=["D1:1"],
            answer="pottery",
            context="Caroline started pottery.",
            k=3,
        )

        self.assertEqual(result.recall_at_k, 1.0)
        self.assertEqual(result.hit_rate_at_k, 1.0)
        self.assertEqual(result.evidence_recall_at_k, 1.0)
        self.assertAlmostEqual(result.mrr, 1 / 3)
        self.assertEqual(result.required_fact_coverage, 1.0)

    def test_empty_evidence_is_abstention_and_does_not_count_as_recall(self) -> None:
        from evaluation.locomo_runner import evaluate_case

        result = evaluate_case(
            retrieved_ids=["D1:1"],
            golden_ids=[],
            answer="Not mentioned",
            context="No relevant evidence.",
            k=5,
            is_abstention=True,
        )

        self.assertEqual(result.recall_at_k, 0.0)
        self.assertEqual(result.mrr, 0.0)
        self.assertEqual(result.required_fact_coverage, 0.0)

    def test_evidence_recall_measures_partial_gold_coverage(self) -> None:
        from evaluation.locomo_runner import evaluate_case

        result = evaluate_case(
            retrieved_ids=["wrong", "D1:2"],
            golden_ids=["D1:1", "D1:2"],
            answer="one and two",
            context="two",
            k=2,
        )

        self.assertEqual(result.hit_rate_at_k, 1.0)
        self.assertEqual(result.evidence_recall_at_k, 0.5)
        self.assertEqual(result.mrr, 0.5)


if __name__ == "__main__":
    unittest.main()
