from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config import (
    DEFAULT_CONFIG,
    TEMPLATE_PATH,
    merge_config_defaults,
    resolve_agent_config,
)
from config_schema import HeartbeatConfig, validate_config
import config_schema


class ConfigDefaultConsistencyTests(unittest.TestCase):
    def test_empty_config_validation_uses_runtime_defaults(self) -> None:
        result = validate_config({})

        self.assertTrue(result.ok, result.errors)
        self.assertEqual(result.config, DEFAULT_CONFIG)

    def test_heartbeat_schema_default_matches_runtime_default(self) -> None:
        runtime_enabled = DEFAULT_CONFIG["agents"]["defaults"]["heartbeat"][
            "enabled"
        ]

        self.assertFalse(runtime_enabled)
        self.assertEqual(HeartbeatConfig().enabled, runtime_enabled)

    def test_config_template_matches_runtime_defaults(self) -> None:
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as config_file:
            template = json.load(config_file)

        self.assertEqual(template, DEFAULT_CONFIG)

    def test_partial_agent_override_is_not_expanded_with_schema_defaults(
        self,
    ) -> None:
        result = validate_config({
            "agents": {
                "list": [
                    {
                        "id": "main",
                        "heartbeat": {"enabled": True},
                    }
                ]
            }
        })

        self.assertTrue(result.ok, result.errors)
        self.assertFalse(
            result.config["agents"]["defaults"]["heartbeat"]["enabled"]
        )
        self.assertEqual(
            result.config["agents"]["list"][0]["heartbeat"],
            {"enabled": True},
        )

    def test_partial_subagent_override_inherits_runtime_defaults(self) -> None:
        result = validate_config({
            "agents": {
                "list": [
                    {
                        "id": "main",
                        "subagents": {"max_children_per_agent": 9},
                    }
                ]
            }
        })
        self.assertTrue(result.ok, result.errors)
        self.assertEqual(
            result.config["agents"]["list"][0]["subagents"],
            {"max_children_per_agent": 9},
        )

        with patch("config.get_config", return_value=result.config):
            resolved = resolve_agent_config("main")

        self.assertEqual(resolved["subagents"]["max_children_per_agent"], 9)
        self.assertIn("allow_agents", resolved["subagents"])
        self.assertIn("max_spawn_depth", resolved["subagents"])
        self.assertEqual(resolved["subagents"]["allow_agents"], ["*"])
        self.assertEqual(resolved["subagents"]["max_spawn_depth"], 2)

    def test_empty_agent_list_uses_runtime_default_agent(self) -> None:
        result = validate_config({"agents": {"list": []}})

        self.assertTrue(result.ok, result.errors)
        self.assertEqual(
            result.config["agents"]["list"],
            DEFAULT_CONFIG["agents"]["list"],
        )

    def test_empty_heartbeat_interval_uses_runtime_default(self) -> None:
        with patch.dict(
            config_schema._HEARTBEAT_DEFAULTS,
            {"every": "45m"},
        ):
            heartbeat = HeartbeatConfig(every="")

        self.assertEqual(heartbeat.every, "45m")

    def test_merged_config_does_not_mutate_runtime_defaults(self) -> None:
        merged = merge_config_defaults({})

        merged["agents"]["defaults"]["heartbeat"]["enabled"] = True

        self.assertFalse(
            DEFAULT_CONFIG["agents"]["defaults"]["heartbeat"]["enabled"]
        )


if __name__ == "__main__":
    unittest.main()
