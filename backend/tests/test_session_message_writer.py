from __future__ import annotations

import sys
import unittest
from contextlib import contextmanager
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sessions.session_message_writer import SessionMessageWriter


class SessionMessageWriterTests(unittest.TestCase):
    def _build_writer(
        self,
        sessions: dict[tuple[str, str], dict] | None = None,
    ) -> tuple[SessionMessageWriter, list[tuple[str, str, dict]]]:
        sessions = sessions if sessions is not None else {}
        saved: list[tuple[str, str, dict]] = []

        @contextmanager
        def transaction(session_id: str, agent_id: str):
            yield

        def load_session(session_id: str, agent_id: str):
            return sessions.get((session_id, agent_id))

        def save_session_data(session_id: str, agent_id: str, data: dict) -> None:
            sessions[(session_id, agent_id)] = data
            saved.append((session_id, agent_id, data))

        writer = SessionMessageWriter(
            transaction=transaction,
            load_session=load_session,
            persist_session=save_session_data,
            session_key_from_id=lambda agent_id, session_id: (
                f"agent:{agent_id}:subagent:{session_id}"
            ),
            resolve_requester=lambda _key: None,
            now=lambda: 100.0,
        )
        return writer, saved

    def test_save_message_creates_and_persists_session(self) -> None:
        writer, saved = self._build_writer()

        writer.save_message("s1", "main", "user", "hello")

        data = saved[-1][2]
        self.assertEqual(data["messages"], [
            {"role": "user", "content": "hello"},
        ])
        self.assertEqual(data["updated_at"], 100.0)

    def test_ensure_session_resolves_parent_for_subagent(self) -> None:
        writer, saved = self._build_writer()
        writer._resolve_requester = lambda key: ("agent:parent:main", key)

        data = writer.ensure_session("subagent-1", "agent-1", label=" child ")

        self.assertEqual(data["label"], "child")
        self.assertEqual(data["spawned_by"], "agent:parent:main")
        self.assertEqual(saved[-1][2]["spawned_by"], "agent:parent:main")

    def test_save_message_preserves_tool_calls(self) -> None:
        writer, saved = self._build_writer()

        writer.save_message(
            "s1",
            "main",
            "assistant",
            "done",
            tool_calls=[{"name": "search", "input": "query"}],
        )

        self.assertEqual(
            saved[-1][2]["messages"],
            [{
                "role": "assistant",
                "content": "done",
                "tool_calls": [{"name": "search", "input": "query"}],
            }],
        )

    def test_rollback_last_turn_removes_only_user_assistant_pair(self) -> None:
        sessions = {
            ("s1", "main"): {
                "messages": [
                    {"role": "user", "content": "old"},
                    {"role": "assistant", "content": "old answer"},
                    {"role": "user", "content": "latest"},
                    {"role": "assistant", "content": "latest answer"},
                ],
            }
        }
        writer, saved = self._build_writer(sessions)

        self.assertTrue(writer.rollback_last_turn("s1", "main"))
        self.assertEqual(len(saved), 1)
        self.assertEqual(
            sessions[("s1", "main")]["messages"],
            [
                {"role": "user", "content": "old"},
                {"role": "assistant", "content": "old answer"},
            ],
        )

    def test_rollback_last_turn_does_not_remove_unmatched_messages(self) -> None:
        sessions = {
            ("s1", "main"): {
                "messages": [
                    {"role": "assistant", "content": "answer"},
                    {"role": "user", "content": "next"},
                ],
            }
        }
        writer, saved = self._build_writer(sessions)

        self.assertFalse(writer.rollback_last_turn("s1", "main"))
        self.assertEqual(saved, [])


if __name__ == "__main__":
    unittest.main()
