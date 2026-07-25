from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


class SessionDispatcherFactoryTests(unittest.TestCase):
    def test_factory_forwards_runtime_dependencies_to_dispatcher(self) -> None:
        from sessions.session_dispatcher_factory import (
            SessionDispatcherFactory,
        )

        work_store = Mock()
        system_stream = Mock()
        user_stream = Mock()
        coordinator = Mock()
        dispatcher_type = Mock()
        dispatcher = Mock()
        dispatcher_type.return_value = dispatcher
        factory = SessionDispatcherFactory(
            work_store=work_store,
            system_stream=system_stream,
            user_stream=user_stream,
            turn_coordinator=coordinator,
            dispatcher_type=dispatcher_type,
        )
        lock = asyncio.Lock()

        created = factory.create(lock=lock)

        self.assertIs(created, dispatcher)
        dispatcher_type.assert_called_once_with(
            lock=lock,
            work_store=work_store,
            system_stream=system_stream,
            user_stream=user_stream,
            turn_coordinator=coordinator,
        )
        self.assertIs(factory.work_store, work_store)

    def test_factory_resolves_dispatcher_type_lazily(self) -> None:
        from sessions.session_dispatcher_factory import (
            SessionDispatcherFactory,
        )

        factory = SessionDispatcherFactory(work_store=Mock())
        dispatcher_type = Mock(return_value=Mock())

        with patch(
            "sessions.session_dispatcher.SessionDispatcher",
            dispatcher_type,
        ):
            factory.create(lock=asyncio.Lock())

        dispatcher_type.assert_called_once()


if __name__ == "__main__":
    unittest.main()
