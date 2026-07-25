from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sessions.session_work_query import SessionWorkFilter
from sessions.session_work_store import SessionWorkStore


class SessionWorkFilterTests(unittest.TestCase):
    def test_store_query_and_count_forward_every_filter(self) -> None:
        filters = {
            "kind": "cron",
            "kinds": ["cron", "announce"],
            "status": "failed",
            "agent_id": "main",
            "session_id": "main-main",
            "run_id": "run-1",
            "run_id_prefix": "run-",
            "exclude_run_id_prefix": "heartbeat-",
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = SessionWorkStore(Path(tmp_dir) / "session-work.db")
            with patch(
                "sessions.session_work_store.SessionWorkFilter"
            ) as filter_type:
                filter_type.return_value.to_sql.side_effect = (
                    lambda: ("1=1", [])
                )

                store.query(**filters)
                store.count(**filters)

        self.assertEqual(
            filter_type.call_args_list,
            [call(**filters), call(**filters)],
        )

    def test_store_query_and_count_use_the_shared_filter(self) -> None:
        source = (
            BACKEND_DIR / "sessions" / "session_work_store.py"
        ).read_text(encoding="utf-8")

        self.assertEqual(source.count("SessionWorkFilter("), 2)
        self.assertNotIn("conditions: list[str]", source)

    def test_all_filters_keep_existing_sql_and_parameter_order(self) -> None:
        where, params = SessionWorkFilter(
            kind="cron",
            kinds=["cron", "announce"],
            status="failed",
            agent_id="main",
            session_id="main-main",
            run_id="run-1",
            run_id_prefix="run-",
            exclude_run_id_prefix="heartbeat-",
        ).to_sql()

        self.assertEqual(
            where,
            "kind = ? AND kind IN (?, ?) AND status = ? AND agent_id = ? "
            "AND session_id = ? AND run_id = ? AND run_id LIKE ? AND "
            "(run_id IS NULL OR run_id NOT LIKE ?)",
        )
        self.assertEqual(
            params,
            [
                "cron",
                "cron",
                "announce",
                "failed",
                "main",
                "main-main",
                "run-1",
                "run-%",
                "heartbeat-%",
            ],
        )

    def test_empty_filter_values_keep_unfiltered_query(self) -> None:
        where, params = SessionWorkFilter(
            kind="",
            kinds=[],
            status=None,
            agent_id="",
            session_id=None,
            run_id="",
            run_id_prefix=None,
            exclude_run_id_prefix="",
        ).to_sql()

        self.assertEqual(where, "1=1")
        self.assertEqual(params, [])


if __name__ == "__main__":
    unittest.main()
