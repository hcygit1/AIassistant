from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from llm.models_config import ModelRef
from runtime.model_runtime import ModelRuntime
from runtime.agent import AgentManager
from api.config_api import (
    ModelSwitchRequest,
    _reload_subsystems,
    get_current_model,
    switch_model as switch_model_endpoint,
)


class ModelRuntimeTests(unittest.TestCase):
    def test_agent_manager_switch_updates_turn_candidates(
        self,
    ) -> None:
        manager = AgentManager()
        runtime = manager._model_runtime
        runtime._resolve_configured_model = (
            lambda _agent_id: ModelRef(
                provider="fake",
                model="old",
            )
        )
        runtime._resolve_configured_candidates = (
            lambda _agent_id: [
                ModelRef(provider="fake", model="old")
            ]
        )
        runtime._get_model = (
            lambda ref: SimpleNamespace(
                id=ref.model,
                name=ref.model,
            )
        )
        runtime._invalidate_llm = Mock()
        runtime._get_or_create_llm = Mock(
            return_value=object()
        )
        runtime._get_display_name = (
            lambda ref: str(ref)
        )

        manager.switch_model("main", "fake/new")

        self.assertEqual(
            manager.get_current_model_ref("main"),
            ModelRef(provider="fake", model="new"),
        )
        self.assertEqual(
            manager._turn_service._ports.resolve_candidates(
                "main"
            )[0],
            ModelRef(provider="fake", model="new"),
        )

    def _build_runtime(
        self,
        *,
        create_llm=None,
    ) -> tuple[ModelRuntime, Mock, Mock]:
        old = ModelRef(provider="fake", model="old")
        fallback = ModelRef(
            provider="backup",
            model="fallback",
        )
        invalidate = Mock()
        get_or_create = create_llm or Mock(
            return_value=object()
        )
        runtime = ModelRuntime(
            resolve_configured_model=lambda _agent_id: old,
            resolve_configured_candidates=(
                lambda _agent_id: [old, fallback]
            ),
            find_model=lambda model_id: (
                (
                    SimpleNamespace(id="fake"),
                    SimpleNamespace(id=model_id),
                )
                if model_id in {"new", "old"}
                else None
            ),
            get_model=lambda ref: (
                SimpleNamespace(
                    id=ref.model.lower(),
                    name=ref.model,
                )
                if ref.provider in {"fake", "backup"}
                else None
            ),
            invalidate_llm=invalidate,
            get_or_create_llm=get_or_create,
            get_display_name=lambda ref: (
                f"Display {ref.provider}/{ref.model}"
            ),
        )
        return runtime, invalidate, get_or_create

    def test_switch_override_becomes_current_and_primary(
        self,
    ) -> None:
        runtime, invalidate, get_or_create = (
            self._build_runtime()
        )

        name = runtime.switch("main", "fake/new")

        self.assertEqual(
            runtime.resolve_current("main"),
            ModelRef(provider="fake", model="new"),
        )
        self.assertEqual(
            runtime.resolve_candidates("main"),
            [
                ModelRef(provider="fake", model="new"),
                ModelRef(provider="fake", model="old"),
                ModelRef(
                    provider="backup",
                    model="fallback",
                ),
            ],
        )
        self.assertEqual(name, "Display fake/new")
        invalidate.assert_called_once_with("main")
        get_or_create.assert_called_once_with(
            "main",
            ModelRef(provider="fake", model="new"),
        )

    def test_switch_resolves_model_without_provider(
        self,
    ) -> None:
        runtime, _, _ = self._build_runtime()

        runtime.switch("main", "new")

        self.assertEqual(
            runtime.resolve_current("main"),
            ModelRef(provider="fake", model="new"),
        )

    def test_failed_switch_does_not_replace_current_model(
        self,
    ) -> None:
        get_or_create = Mock(
            side_effect=RuntimeError("provider unavailable")
        )
        runtime, _, _ = self._build_runtime(
            create_llm=get_or_create
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "provider unavailable",
        ):
            runtime.switch("main", "fake/new")

        self.assertEqual(
            runtime.resolve_current("main"),
            ModelRef(provider="fake", model="old"),
        )

    def test_candidates_do_not_duplicate_override(
        self,
    ) -> None:
        runtime, _, _ = self._build_runtime()

        runtime.switch("main", "fake/old")

        self.assertEqual(
            runtime.resolve_candidates("main"),
            [
                ModelRef(provider="fake", model="old"),
                ModelRef(
                    provider="backup",
                    model="fallback",
                ),
            ],
        )

    def test_switch_normalizes_catalog_model_id(
        self,
    ) -> None:
        runtime, _, _ = self._build_runtime()

        runtime.switch("main", "fake/NEW")

        self.assertEqual(
            runtime.resolve_current("main"),
            ModelRef(provider="fake", model="new"),
        )
        self.assertEqual(
            [str(ref) for ref in runtime.resolve_candidates("main")],
            ["fake/new", "fake/old", "backup/fallback"],
        )

    def test_display_name_failure_does_not_publish_override(
        self,
    ) -> None:
        runtime, _, _ = self._build_runtime()
        runtime._get_display_name = Mock(
            side_effect=RuntimeError("display failed")
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "display failed",
        ):
            runtime.switch("main", "fake/new")

        self.assertIsNone(runtime.get_override("main"))

    def test_restore_override_reverts_runtime_selection(
        self,
    ) -> None:
        runtime, invalidate, _ = self._build_runtime()
        runtime.switch("main", "fake/new")
        invalidate.reset_mock()

        runtime.restore_override("main", None)

        self.assertIsNone(runtime.get_override("main"))
        self.assertEqual(
            runtime.resolve_current("main"),
            ModelRef(provider="fake", model="old"),
        )
        invalidate.assert_called_once_with("main")

    def test_candidates_drop_stale_override(
        self,
    ) -> None:
        runtime, _, _ = self._build_runtime()
        runtime.switch("main", "fake/new")
        runtime._get_model = lambda ref: (
            None
            if ref == ModelRef(
                provider="fake",
                model="new",
            )
            else SimpleNamespace(id=ref.model)
        )

        candidates = runtime.resolve_candidates("main")

        self.assertIsNone(runtime.get_override("main"))
        self.assertEqual(
            candidates,
            [
                ModelRef(provider="fake", model="old"),
                ModelRef(
                    provider="backup",
                    model="fallback",
                ),
            ],
        )

    def test_candidates_normalize_configured_model_ids(
        self,
    ) -> None:
        runtime, _, _ = self._build_runtime()
        runtime._resolve_configured_candidates = (
            lambda _agent_id: [
                ModelRef(provider="fake", model="NEW"),
                ModelRef(provider="fake", model="new"),
            ]
        )

        candidates = runtime.resolve_candidates("main")

        self.assertEqual(
            candidates,
            [ModelRef(provider="fake", model="new")],
        )


class ModelRuntimeApiTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_current_model_endpoint_uses_runtime_model(
        self,
    ) -> None:
        current = ModelRef(
            provider="fake",
            model="runtime",
        )
        model_definition = SimpleNamespace(
            reasoning=True,
            input=["text"],
            context_window=64000,
            max_tokens=4096,
        )

        with (
            patch(
                "runtime.agent.agent_manager.get_current_model_ref",
                return_value=current,
            ) as get_current,
            patch(
                "llm.model_selection.get_model_display_name",
                return_value="Runtime Model",
            ),
            patch(
                "llm.models_config.models_config.get_model",
                return_value=model_definition,
            ),
            patch(
                "llm.models_config.models_config.resolve_api_protocol",
                return_value="openai-completions",
            ),
        ):
            result = await get_current_model("main")

        get_current.assert_called_once_with("main")
        self.assertEqual(result["ref"], "fake/runtime")
        self.assertEqual(result["name"], "Runtime Model")

    async def test_switch_endpoint_restores_override_when_save_fails(
        self,
    ) -> None:
        previous = ModelRef(
            provider="fake",
            model="old",
        )
        restore = Mock()

        with (
            patch(
                "runtime.agent.agent_manager.get_model_override",
                return_value=previous,
            ),
            patch(
                "runtime.agent.agent_manager.switch_model",
                return_value="New Model",
            ),
            patch(
                "runtime.agent.agent_manager.get_current_model_ref",
                return_value=ModelRef(
                    provider="fake",
                    model="new",
                ),
            ),
            patch(
                "runtime.agent.agent_manager.restore_model_override",
                new=restore,
            ),
            patch(
                "config.get_raw_config",
                return_value={
                    "agents": {
                        "defaults": {
                            "model": "fake/old"
                        },
                        "list": [],
                    }
                },
            ),
            patch(
                "config.save_config",
                side_effect=OSError("disk full"),
            ),
        ):
            result = await switch_model_endpoint(
                "main",
                ModelSwitchRequest(model="fake/new"),
            )

        self.assertEqual(result["status"], "error")
        restore.assert_called_once_with("main", previous)

    async def test_agent_manager_close_clears_model_overrides(
        self,
    ) -> None:
        manager = AgentManager()
        manager._model_runtime._overrides["main"] = ModelRef(
            provider="fake",
            model="runtime",
        )

        await manager.close()

        self.assertIsNone(
            manager._model_runtime.get_override("main")
        )

    async def test_model_reload_clears_runtime_overrides(
        self,
    ) -> None:
        clear = Mock()

        with (
            patch(
                "runtime.agent.agent_manager.clear_model_overrides",
                new=clear,
            ),
            patch(
                "llm.models_config.models_config.reload"
            ),
            patch(
                "llm.llm_factory.llm_cache.invalidate_all"
            ),
            patch(
                "config.get_config",
                return_value={"models": {}},
            ),
        ):
            _reload_subsystems({"models": {}})

        clear.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
