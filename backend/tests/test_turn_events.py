from __future__ import annotations

import importlib
import importlib.util
import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


class TurnEventTests(unittest.TestCase):
    def test_event_copies_payload_and_normalizes_type(self) -> None:
        spec = importlib.util.find_spec("turns.events")
        self.assertIsNotNone(spec, "turns.events should be defined")
        module = importlib.import_module("turns.events")
        turn_event = getattr(module, "TurnEvent", None)
        self.assertIsNotNone(turn_event, "TurnEvent should be defined")

        source = {"type": "error", "error": "model failed"}
        event = turn_event.from_payload(source)
        source["error"] = "mutated"

        self.assertEqual(event.type, "error")
        self.assertEqual(event.payload["type"], "error")
        self.assertEqual(event.payload["error"], "model failed")

    def test_error_event_exposes_error_message(self) -> None:
        spec = importlib.util.find_spec("turns.events")
        self.assertIsNotNone(spec, "turns.events should be defined")
        module = importlib.import_module("turns.events")
        turn_event = module.TurnEvent

        event = turn_event.error("model failed")

        self.assertEqual(event.type, "error")
        self.assertEqual(event.error_message, "model failed")

    def test_constructor_enforces_type_and_deep_payload_snapshot(self) -> None:
        module = importlib.import_module("turns.events")
        source = {
            "type": "done",
            "nested": {"items": ["original"]},
        }

        event = module.TurnEvent(type="error", payload=source)
        source["nested"]["items"].append("mutated")

        self.assertEqual(event.type, "error")
        self.assertEqual(event.payload["type"], "error")
        self.assertEqual(event.payload["nested"]["items"], ["original"])

    def test_event_rejects_non_json_payload_values(self) -> None:
        module = importlib.import_module("turns.events")

        with self.assertRaisesRegex(TypeError, "not JSON-compatible"):
            module.TurnEvent.from_payload(
                {
                    "type": "token",
                    "content": object(),
                }
            )


if __name__ == "__main__":
    unittest.main()
