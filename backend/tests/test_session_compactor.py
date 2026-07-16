from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from runtime.session_compactor import SessionCompactor


class SessionCompactorTests(unittest.IsolatedAsyncioTestCase):
    def _build_compactor(
        self,
        *,
        messages: list[dict] | None = None,
    ) -> tuple[SessionCompactor, Mock, Mock]:
        compress_history = Mock(
            return_value={
                "archived_count": 4,
                "remaining_count": 4,
            }
        )
        log_compress = Mock()
        compactor = SessionCompactor(
            resolve_agent_config=lambda _agent_id: {
                "compaction": {
                    "keepRecentTurns": 2,
                    "forcedKeepRecentTurns": 1,
                }
            },
            load_session=lambda _session_id, _agent_id: {
                "messages": messages or [],
            },
            compress_history=compress_history,
            get_llm=lambda _agent_id: object(),
            log_compress=log_compress,
        )
        return compactor, compress_history, log_compress

    async def test_compress_session_owns_summary_and_archive_workflow(
        self,
    ) -> None:
        messages = [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
            {"role": "assistant", "content": "a3"},
            {"role": "user", "content": "u4"},
            {"role": "assistant", "content": "a4"},
        ]
        compactor, compress_history, log_compress = self._build_compactor(
            messages=messages,
        )
        store = SimpleNamespace(upsert_session_summary=Mock())
        state = SimpleNamespace(compaction_count=0)
        generate_summary = AsyncMock(
            return_value={"raw_summary": "summary"}
        )
        batch_ingest = AsyncMock()
        pending_tasks: set[asyncio.Task] = set()

        result = await compactor.compress_session(
            "s1",
            "main",
            level="sliding",
            get_store=lambda _agent_id: store,
            get_state=lambda _agent_id: state,
            generate_summary=generate_summary,
            batch_ingest_messages=batch_ingest,
            pending_tasks=pending_tasks,
        )
        await asyncio.sleep(0)

        self.assertEqual(result["compress"]["level"], "sliding")
        generate_summary.assert_awaited_once()
        store.upsert_session_summary.assert_called_once_with(
            "s1",
            "main",
            {"raw_summary": "summary"},
        )
        compress_history.assert_called_once_with("s1", "main", 4)
        self.assertEqual(state.compaction_count, 1)
        log_compress.assert_called_once_with("main", "s1", 4, 4)
        batch_ingest.assert_awaited_once_with(
            "main",
            "s1",
            messages[:4],
            session_end=False,
        )

    async def test_generate_structured_summary_parses_json_response(
        self,
    ) -> None:
        llm = SimpleNamespace(
            ainvoke=AsyncMock(
                return_value=SimpleNamespace(
                    content='{"goal":"ship","decisions":[]}'
                )
            )
        )
        compactor = SessionCompactor(
            resolve_agent_config=lambda _agent_id: {},
            load_session=lambda _session_id, _agent_id: None,
            compress_history=lambda _session_id, _agent_id, _count: {},
            get_llm=lambda _agent_id: llm,
            log_compress=lambda *_args: None,
        )
        plain_fallback = AsyncMock()

        result = await compactor.generate_structured_summary(
            "main",
            "s1",
            [{"role": "user", "content": "ship"}],
            "[user] ship",
            store=None,
            plain_fallback=plain_fallback,
        )

        self.assertEqual(result["goal"], "ship")
        self.assertEqual(result["raw_summary"], '{"goal":"ship","decisions":[]}')
        self.assertGreaterEqual(result["token_count"], 0)
        plain_fallback.assert_not_awaited()

    async def test_generate_structured_summary_uses_plain_fallback_after_retry_failure(
        self,
    ) -> None:
        compactor, _compress_history, _log_compress = (
            self._build_compactor()
        )
        messages = [{"role": "user", "content": "ship"}]
        plain_fallback = AsyncMock(
            return_value={"raw_summary": "fallback"}
        )

        with patch(
            "runtime.session_compactor.retry_async",
            new=AsyncMock(side_effect=RuntimeError("invalid json")),
        ):
            result = await compactor.generate_structured_summary(
                "main",
                "s1",
                messages,
                "[user] ship",
                store=None,
                plain_fallback=plain_fallback,
            )

        self.assertEqual(result, {"raw_summary": "fallback"})
        plain_fallback.assert_awaited_once_with(
            "main",
            messages,
            "[user] ship",
        )

    def test_calc_compress_count_by_turns_keeps_complete_recent_turns(
        self,
    ) -> None:
        messages = [
            {"role": "system", "content": "summary"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
            {"role": "assistant", "content": "a3"},
        ]

        self.assertEqual(
            SessionCompactor.calc_compress_count_by_turns(
                messages,
                keep_turns=2,
            ),
            3,
        )
        self.assertEqual(
            SessionCompactor.calc_compress_count_by_turns(
                messages,
                keep_turns=3,
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main()
