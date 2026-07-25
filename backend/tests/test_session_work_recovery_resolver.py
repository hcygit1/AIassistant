from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from sessions.session_work_store import SessionWorkRecord


class SessionWorkRecoveryResolverTests(unittest.TestCase):
    def _record(self, kind: str = "cron") -> SessionWorkRecord:
        return SessionWorkRecord(
            id="work-1",
            kind=kind,
            agent_id="main",
            session_id="main-main",
            content="recover",
            priority=2,
            run_id="cron-1",
            recover_on_restart=True,
        )

    def test_required_kind_is_rejected_until_provider_is_bound(self) -> None:
        from sessions.session_work_recovery_resolver import (
            SessionWorkRecoveryResolver,
        )

        resolver = SessionWorkRecoveryResolver(required_kinds={"cron"})

        self.assertIsNone(resolver.resolve(self._record()))

    def test_bound_provider_resolves_callbacks_for_its_kind(self) -> None:
        from sessions.session_work_recovery_resolver import (
            SessionWorkRecoveryResolver,
        )

        callbacks = {"on_success": Mock()}
        provider = Mock(return_value=callbacks)
        resolver = SessionWorkRecoveryResolver(required_kinds={"cron"})
        resolver.bind("cron", provider)
        record = self._record()

        self.assertIs(resolver.resolve(record), callbacks)
        provider.assert_called_once_with(record)

    def test_unknown_kind_keeps_empty_callback_compatibility(self) -> None:
        from sessions.session_work_recovery_resolver import (
            SessionWorkRecoveryResolver,
        )

        resolver = SessionWorkRecoveryResolver(required_kinds={"cron"})

        self.assertEqual(resolver.resolve(self._record("future-kind")), {})

    def test_kind_normalization_is_consistent_across_configuration(self) -> None:
        from sessions.session_work_recovery_resolver import (
            SessionWorkRecoveryResolver,
        )

        provider = Mock(return_value={})
        resolver = SessionWorkRecoveryResolver(required_kinds={" cron "})
        resolver.bind(" cron ", provider)
        record = self._record(" cron ")

        self.assertEqual(resolver.resolve(record), {})
        provider.assert_called_once_with(record)


if __name__ == "__main__":
    unittest.main()
