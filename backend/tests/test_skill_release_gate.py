from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from evaluation.skill_release_gate import VariantMetrics, evaluate_release_gate, static_skill_check


class SkillReleaseGateTests(unittest.TestCase):
    def test_static_skill_check_requires_frontmatter_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "SKILL.md"
            path.write_text("---\nname: demo\ndescription: demo\n---\n" + "x" * 50, encoding="utf-8")
            self.assertEqual(static_skill_check(path), (True, ()))
    def test_accepts_candidate_that_beats_baseline_and_active(self) -> None:
        decision = evaluate_release_gate(
            static_check_passed=True,
            validation={
                "without_skill": VariantMetrics(0.3, 100),
                "active_skill": VariantMetrics(0.6, 110),
                "candidate_skill": VariantMetrics(0.8, 120),
            },
            regression_candidate_passed=True,
        )
        self.assertEqual(decision.status, "accepted")

    def test_rejects_candidate_that_regresses_active(self) -> None:
        decision = evaluate_release_gate(
            static_check_passed=True,
            validation={
                "without_skill": VariantMetrics(0.3),
                "active_skill": VariantMetrics(0.8),
                "candidate_skill": VariantMetrics(0.6),
            },
            regression_candidate_passed=True,
        )
        self.assertEqual(decision.status, "rejected")
        self.assertIn("regressed", decision.reasons[0])
