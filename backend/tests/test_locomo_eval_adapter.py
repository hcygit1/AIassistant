from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


class LoCoMoAdapterTests(unittest.TestCase):
    def test_adapts_sessions_turns_and_evidence_to_memory_records(self) -> None:
        from evaluation.locomo_adapter import adapt_conversation

        conversation = {
            "sample_id": "sample-1",
            "conversation": {
                "session_1": [
                    {"speaker": "Caroline", "text": "I started pottery.", "dia_id": "D1:1"},
                    {"speaker": "Melanie", "text": "That sounds fun.", "dia_id": "D1:2"},
                ],
                "session_1_date_time": "1:00 pm on 8 May, 2023",
                "session_2": [
                    {"speaker": "Caroline", "text": "I now prefer watercolor.", "dia_id": "D2:1"},
                ],
                "session_2_date_time": "2:00 pm on 9 May, 2023",
            },
            "qa": [
                {
                    "question": "What hobby did Caroline start?",
                    "answer": "Pottery",
                    "evidence": ["D1:1"],
                    "category": 1,
                },
                {
                    "question": "What is Caroline's favorite food?",
                    "answer": "Not mentioned",
                    "evidence": [],
                    "category": 5,
                },
            ],
        }

        adapted = adapt_conversation(conversation, max_questions=20)

        self.assertEqual([task.id for task in adapted.tasks], ["sample-1:session_1", "sample-1:session_2"])
        self.assertEqual([chunk.id for chunk in adapted.chunks], ["D1:1", "D1:2", "D2:1"])
        self.assertEqual(adapted.chunks[0].task_id, "sample-1:session_1")
        self.assertEqual(adapted.chunks[0].role, "user")
        self.assertEqual(adapted.chunks[1].role, "assistant")
        self.assertEqual(adapted.cases[0].golden_task_ids, ["sample-1:session_1"])
        self.assertEqual(adapted.cases[0].golden_chunk_ids, ["D1:1"])
        self.assertFalse(adapted.cases[0].is_abstention)
        self.assertTrue(adapted.cases[1].is_abstention)

    def test_loads_first_conversation_and_selects_cases_by_category(self) -> None:
        from evaluation.locomo_adapter import load_pilot

        fixture = [
            {
                "sample_id": "first",
                "conversation": {
                    "session_1": [
                        {"speaker": "A", "text": "Fact A", "dia_id": "D1:1"},
                    ],
                    "session_1_date_time": "1:00 pm on 8 May, 2023",
                },
                "qa": [
                    {"question": "q1a", "answer": "a1", "evidence": ["D1:1"], "category": 1},
                    {"question": "q1b", "answer": "a1", "evidence": ["D1:1"], "category": 1},
                    {"question": "q3", "answer": "a3", "evidence": ["D1:1"], "category": 3},
                    {"question": "q2", "answer": "a2", "evidence": [], "category": 5},
                ],
            },
            {"sample_id": "second", "conversation": {}, "qa": []},
        ]
        path = Path(self.id().replace(".", "_") + ".json")
        try:
            path.write_text(json.dumps(fixture), encoding="utf-8")
            adapted = load_pilot(path, sample_index=0, max_questions=2)
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual({case.category for case in adapted.cases}, {1, 3})

    def test_splits_compound_evidence_ids(self) -> None:
        from evaluation.locomo_adapter import adapt_conversation

        source = {
            "sample_id": "sample-1",
            "conversation": {
                "session_1": [
                    {"speaker": "A", "text": "One", "dia_id": "D1:1"},
                    {"speaker": "B", "text": "Two", "dia_id": "D1:2"},
                ],
            },
            "qa": [{
                "question": "What happened?",
                "answer": "One and two",
                "evidence": ["D1:1; D1:2"],
                "category": 1,
            }],
        }

        adapted = adapt_conversation(source, max_questions=1)

        self.assertEqual(adapted.cases[0].golden_chunk_ids, ["D1:1", "D1:2"])


if __name__ == "__main__":
    unittest.main()
