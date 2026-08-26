from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.skill_evolution_runner import gate_report


class SkillEvolutionRunnerTests(unittest.TestCase):
    def test_gate_report_accepts_three_variant_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "report.json"
            path.write_text(json.dumps({"systems": [
                {"system": "without_skill", "pass_rate": 0.2, "avg_tokens": 100},
                {"system": "active_skill", "pass_rate": 0.5, "avg_tokens": 110},
                {"system": "candidate_skill", "pass_rate": 0.7, "avg_tokens": 120},
            ]}), encoding="utf-8")
            result = gate_report(path, static_check_passed=True, regression_candidate_passed=True)
            self.assertEqual(result["status"], "accepted")


if __name__ == "__main__":
    unittest.main()
