from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from runtime.session_lifecycle import SessionLifecycle


class SessionLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_auto_compact_emits_lifecycle_events_and_records_audit(self) -> None:
        events: list[dict] = []
        audit: list[tuple[str, dict]] = []
        compress = AsyncMock()
        lifecycle = SessionLifecycle(
            compactor=Mock(),
            resolve_agent_config=lambda _agent_id: {
                "compaction": {"enabled": True},
            },
            load_session=lambda _session_id, _agent_id: {
                "messages": [{"role": "user", "content": "hello"}],
            },
            detect_compaction_level=lambda *_args, **_kwargs: "sliding",
            audit_log=lambda _agent_id, event_type, data: audit.append(
                (event_type, data)
            ),
            emit_event=lambda _agent_id, event: events.append(event),
            get_store=lambda _agent_id: None,
            get_state=lambda _agent_id: None,
            batch_ingest_messages=AsyncMock(),
            pending_tasks=set(),
        )

        await lifecycle.maybe_auto_compact(
            "s1",
            "main",
            compress_session=compress,
        )

        compress.assert_awaited_once_with("s1", "main", level="sliding")
        self.assertEqual(
            [event["event"] for event in events],
            ["auto_compact_start", "auto_compact_done"],
        )
        self.assertEqual(audit[0][0], "auto_compact_trigger")


if __name__ == "__main__":
    unittest.main()
