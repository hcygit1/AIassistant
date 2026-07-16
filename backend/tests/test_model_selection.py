from __future__ import annotations

import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from llm.model_selection import (
    ModelCandidateError,
    run_with_fallback_stream,
)
from llm.models_config import ModelRef


class FallbackStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_does_not_switch_models_after_event_was_emitted(
        self,
    ) -> None:
        attempts: list[str] = []

        async def _run_model(_provider: str, model: str):
            attempts.append(model)
            yield {"type": "token", "content": model}
            if model == "first":
                raise RuntimeError("stream interrupted")

        candidates = [
            ModelRef(provider="fake", model="first"),
            ModelRef(provider="fake", model="second"),
        ]
        events = []

        with self.assertRaisesRegex(
            RuntimeError,
            "stream interrupted",
        ) as raised:
            async for event in run_with_fallback_stream(
                candidates,
                _run_model,
            ):
                events.append(event)

        self.assertEqual(attempts, ["first"])
        self.assertEqual(
            events,
            [{"type": "token", "content": "first"}],
        )
        self.assertTrue(
            getattr(raised.exception, "committed", False)
        )

    async def test_does_not_switch_on_non_model_error_before_event(
        self,
    ) -> None:
        attempts: list[str] = []

        async def _run_model(_provider: str, model: str):
            attempts.append(model)
            if False:
                yield {}
            raise RuntimeError("tracker unavailable")

        candidates = [
            ModelRef(provider="fake", model="first"),
            ModelRef(provider="fake", model="second"),
        ]

        with self.assertRaisesRegex(
            RuntimeError,
            "tracker unavailable",
        ):
            async for _event in run_with_fallback_stream(
                candidates,
                _run_model,
            ):
                pass

        self.assertEqual(attempts, ["first"])

    async def test_all_failed_error_preserves_last_model_error(
        self,
    ) -> None:
        async def _run_model(_provider: str, model: str):
            if False:
                yield {}
            raise ModelCandidateError(
                f"503 upstream unavailable from {model}"
            )

        candidates = [
            ModelRef(provider="fake", model="first"),
            ModelRef(provider="fake", model="second"),
        ]

        with self.assertRaisesRegex(
            RuntimeError,
            "503 upstream unavailable from second",
        ):
            async for _event in run_with_fallback_stream(
                candidates,
                _run_model,
            ):
                pass


if __name__ == "__main__":
    unittest.main()
