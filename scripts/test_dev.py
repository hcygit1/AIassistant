import re
import unittest

from scripts import dev


class DevEnvironmentTests(unittest.TestCase):
    def test_frontend_env_uses_requested_ports(self) -> None:
        builder = getattr(dev, "build_frontend_env", None)
        self.assertIsNotNone(builder, "build_frontend_env should be defined")

        original = {
            "KEEP": "value",
            "NEXT_PUBLIC_API_URL": "https://api.example.test/api",
        }
        env = builder(original, backend_port=9123, frontend_port=4100)

        self.assertEqual(env["KEEP"], "value")
        self.assertEqual(env["PORT"], "4100")
        self.assertEqual(
            env["NEXT_PUBLIC_API_URL"],
            "https://api.example.test/api",
        )
        self.assertEqual(env["NEXT_PUBLIC_API_PORT"], "9123")
        self.assertNotIn("NEXT_PUBLIC_API_PORT", original)

    def test_backend_env_keeps_default_origins(self) -> None:
        builder = getattr(dev, "build_backend_env", None)
        self.assertIsNotNone(builder, "build_backend_env should be defined")

        env = builder({}, frontend_port=3000)

        self.assertEqual(
            env["PIPIXIA_CORS_ORIGINS"].split(","),
            [
                "http://localhost:3000",
                "http://127.0.0.1:3000",
                "http://localhost:8002",
                "http://127.0.0.1:8002",
            ],
        )
        origin_regex = env["PIPIXIA_CORS_ORIGIN_REGEX"]
        for origin in (
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://10.0.0.8:3000",
            "http://172.16.0.8:3000",
            "http://172.31.255.8:3000",
            "http://192.168.1.8:3000",
            "http://pipixia.local:3000",
        ):
            self.assertIsNotNone(re.fullmatch(origin_regex, origin), origin)

        for origin in (
            "https://evil.example:3000",
            "http://172.32.0.8:3000",
            "http://192.169.1.8:3000",
            "http://pipixia.local:4100",
        ):
            self.assertIsNone(re.fullmatch(origin_regex, origin), origin)

    def test_backend_env_merges_custom_origin_without_duplicates(self) -> None:
        builder = getattr(dev, "build_backend_env", None)
        self.assertIsNotNone(builder, "build_backend_env should be defined")

        env = builder(
            {
                "PIPIXIA_CORS_ORIGINS": (
                    "https://example.test, http://localhost:4100"
                ),
                "PIPIXIA_CORS_ORIGIN_REGEX": r"^https://trusted\.test$",
            },
            frontend_port=4100,
        )

        self.assertEqual(
            env["PIPIXIA_CORS_ORIGINS"].split(","),
            [
                "https://example.test",
                "http://localhost:4100",
                "http://127.0.0.1:4100",
            ],
        )
        self.assertEqual(
            env["PIPIXIA_CORS_ORIGIN_REGEX"],
            r"^https://trusted\.test$",
        )


if __name__ == "__main__":
    unittest.main()
