from __future__ import annotations

import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from api.sse import encode_turn_event_sse
from turns.events import TurnEvent


class SseEncodingTests(unittest.TestCase):
    def test_turn_event_is_encoded_at_transport_boundary(self) -> None:
        event = TurnEvent.from_payload(
            {
                "type": "token",
                "content": "你好",
            }
        )

        encoded = encode_turn_event_sse(event)

        self.assertEqual(
            encoded,
            'event: token\ndata: {"type": "token", "content": "你好"}\n\n',
        )


if __name__ == "__main__":
    unittest.main()
