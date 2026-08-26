from __future__ import annotations

import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


class LoCoMoMemRunnerTests(unittest.TestCase):
    def test_context_merge_deduplicates_prefix_summary_and_excerpt(self) -> None:
        from evaluation.locomo_mem_runner import _merge_context_text

        self.assertEqual(
            _merge_context_text("same text", "same text"),
            "same text",
        )
        self.assertEqual(
            _merge_context_text("same text and more", "same text"),
            "same text and more",
        )

    def test_token_usage_separates_index_query_context_and_model_output(self) -> None:
        from evaluation.locomo_mem_runner import build_token_usage

        usage = build_token_usage(
            index_texts=["one two", "three"],
            queries=["where"],
            contexts=["one two three"],
            count_tokens=lambda text: len(text.split()),
        )

        self.assertEqual(usage["index_tokens"], 3)
        self.assertEqual(usage["query_tokens"], 1)
        self.assertEqual(usage["context_tokens"], 3)
        self.assertEqual(usage["estimated_llm_input_tokens"], 4)
        self.assertEqual(usage["llm_output_tokens"], 0)

    def test_retrieval_summary_reports_quality_latency_and_token_averages(self) -> None:
        from evaluation.locomo_mem_runner import summarize_runs

        summary = summarize_runs(
            system="waterfall",
            runs=[
                {"hit_rate_at_k": 1.0, "evidence_recall_at_k": 0.5, "mrr": 0.5, "task_hit_rate_at_k": 1.0, "task_evidence_recall_at_k": 1.0, "context_evidence_coverage": 0.5, "latency_ms": 10.0, "query_tokens": 2, "context_tokens": 8},
                {"hit_rate_at_k": 0.0, "evidence_recall_at_k": 0.0, "mrr": 0.0, "task_hit_rate_at_k": 0.0, "task_evidence_recall_at_k": 0.0, "context_evidence_coverage": 0.0, "latency_ms": 30.0, "query_tokens": 4, "context_tokens": 12},
            ],
            index_tokens=100,
        )

        self.assertEqual(summary["hit_rate_at_k"], 0.5)
        self.assertEqual(summary["evidence_recall_at_k"], 0.25)
        self.assertEqual(summary["mrr"], 0.25)
        self.assertEqual(summary["task_hit_rate_at_k"], 0.5)
        self.assertEqual(summary["p95_latency_ms"], 30.0)
        self.assertEqual(summary["avg_query_tokens"], 3.0)
        self.assertEqual(summary["avg_context_tokens"], 10.0)
        self.assertEqual(summary["avg_estimated_llm_input_tokens"], 13.0)
        self.assertEqual(summary["total_query_tokens"], 6)
        self.assertEqual(summary["total_context_tokens"], 20)
        self.assertEqual(summary["total_estimated_llm_input_tokens"], 26)
        self.assertEqual(summary["llm_output_tokens"], 0)

    def test_task_route_summary_keeps_routes_separate(self) -> None:
        from evaluation.locomo_mem_runner import _summarize_task_routes

        summaries = _summarize_task_routes(
            {
                "fts5": [{"hit": 1.0, "recall": 0.5, "mrr": 0.5}],
                "dense": [{"hit": 0.0, "recall": 0.0, "mrr": 0.0}],
                "rrf": [{"hit": 1.0, "recall": 1.0, "mrr": 1.0}],
            },
            top_k=5,
        )

        self.assertEqual([item["route"] for item in summaries], ["fts5", "dense", "rrf"])
        self.assertEqual(summaries[0]["task_evidence_recall_at_k"], 0.5)
        self.assertEqual(summaries[1]["task_hit_rate_at_k"], 0.0)
        self.assertEqual(summaries[2]["task_mrr"], 1.0)
        self.assertTrue(all(item["top_k"] == 5 for item in summaries))

    def test_chunk_route_summary_separates_candidate_and_final_recall(self) -> None:
        from evaluation.locomo_mem_runner import _summarize_chunk_routes

        summaries = _summarize_chunk_routes(
            {
                "fts5": [{"full_ranked_count": 10.0, "full_recall": 0.5, "candidate_recall": 0.5, "final_recall": 0.25, "final_hit": 1.0, "reachable_rate": 0.75}],
                "dense": [{"full_ranked_count": 20.0, "full_recall": 1.0, "candidate_recall": 1.0, "final_recall": 0.5, "final_hit": 1.0, "reachable_rate": 0.75}],
                "rrf": [{"full_ranked_count": 20.0, "full_recall": 1.0, "candidate_recall": 1.0, "final_recall": 0.75, "final_hit": 1.0, "reachable_rate": 0.75}],
            },
            top_k=5,
            candidate_limit=25,
        )

        self.assertEqual(summaries[0]["candidate_evidence_recall_at_k"], 0.5)
        self.assertEqual(summaries[1]["final_evidence_recall_at_k"], 0.5)
        self.assertEqual(summaries[2]["reachable_evidence_rate"], 0.75)
        self.assertEqual(summaries[2]["candidate_top_k"], 25)
        self.assertEqual(summaries[0]["full_range_evidence_recall"], 0.5)
        self.assertEqual(summaries[1]["avg_full_ranked_chunks"], 20.0)

    def test_chunk_scope_audit_reports_full_dense_coverage(self) -> None:
        from evaluation.locomo_mem_runner import _summarize_chunk_scope_audits

        summary = _summarize_chunk_scope_audits([
            {
                "scoped_chunk_count": 20,
                "dense_ranked_chunk_count": 20,
                "dense_missing_chunk_count": 0,
                "dense_out_of_scope_count": 0,
                "scope_complete": True,
            },
            {
                "scoped_chunk_count": 30,
                "dense_ranked_chunk_count": 30,
                "dense_missing_chunk_count": 0,
                "dense_out_of_scope_count": 0,
                "scope_complete": True,
            },
        ])

        self.assertTrue(summary["all_cases_scope_complete"])
        self.assertEqual(summary["avg_scoped_chunk_count"], 25.0)
        self.assertEqual(summary["min_scoped_chunk_count"], 20)
        self.assertEqual(summary["max_scoped_chunk_count"], 30)
        self.assertEqual(summary["total_dense_missing_chunks"], 0)
        self.assertEqual(summary["total_dense_out_of_scope_chunks"], 0)


if __name__ == "__main__":
    unittest.main()
