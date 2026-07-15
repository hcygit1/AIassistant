from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from system_messages.heartbeat import HeartbeatRunner


class HeartbeatRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_stop_waits_for_cancelled_tasks_to_finish(self) -> None:
        runner = HeartbeatRunner()
        runner._running = True
        cleanup_finished = asyncio.Event()

        async def _task_with_async_cleanup() -> None:
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                await asyncio.sleep(0)
                cleanup_finished.set()
                raise

        task = asyncio.create_task(_task_with_async_cleanup())
        await asyncio.sleep(0)
        runner._tasks["main"] = task

        await runner.stop()

        self.assertTrue(cleanup_finished.is_set())
        self.assertTrue(task.done())
        self.assertEqual(runner.active_agents, [])


if __name__ == "__main__":
    unittest.main()
