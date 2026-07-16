from __future__ import annotations

import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sessions.session_title import SessionTitleService


class SessionTitleServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = SessionTitleService()

    def test_explicit_label_has_priority(self) -> None:
        title = self.service.derive(
            {
                "label": "Pinned title",
                "messages": [{"role": "user", "content": "ignored"}],
            },
            max_length=20,
        )
        self.assertEqual(title, "Pinned title")

    def test_bootstrap_commands_and_urls_are_skipped(self) -> None:
        title = self.service.derive(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "[System Message] session created",
                    },
                    {"role": "user", "content": "/new"},
                    {"role": "user", "content": "https://example.com"},
                    {"role": "user", "content": "Actual question"},
                ]
            },
            max_length=20,
        )
        self.assertEqual(title, "Actual question")

    def test_long_title_prefers_punctuation_boundary(self) -> None:
        title = self.service.derive(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "1234567890. this text is longer",
                    }
                ]
            },
            max_length=20,
        )
        self.assertEqual(title, "1234567890…")


if __name__ == "__main__":
    unittest.main()
