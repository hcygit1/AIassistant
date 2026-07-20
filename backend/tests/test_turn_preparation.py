from __future__ import annotations

import sys
import unittest
from pathlib import Path

from langchain_core.messages import HumanMessage

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from runtime.turn_preparation import TurnPreparation


class TurnPreparationTests(unittest.TestCase):
    def test_build_messages_preserves_system_history_semantics(self) -> None:
        preparation = TurnPreparation()

        messages = preparation.build_messages(
            [
                {"role": "system", "content": "summary"},
                {"role": "user", "content": "old question"},
            ],
            "new question",
        )

        self.assertEqual(messages[0].type, "system")
        self.assertEqual(messages[0].content, "summary")
        self.assertEqual(messages[1].type, "human")
        self.assertEqual(messages[1].content, "old question")
        self.assertEqual(messages[-1], HumanMessage(content="new question"))


if __name__ == "__main__":
    unittest.main()
