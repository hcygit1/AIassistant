import unittest

from scripts import dev


class DevEnvironmentTests(unittest.TestCase):
    def test_frontend_env_uses_requested_ports(self) -> None:
        builder = getattr(dev, "build_frontend_env", None)
        self.assertIsNotNone(builder, "build_frontend_env should be defined")

        original = {"KEEP": "value", "NEXT_PUBLIC_API_URL": "http://stale/api"}
        env = builder(original, backend_port=9123, frontend_port=4100)

        self.assertEqual(env["KEEP"], "value")
        self.assertEqual(env["PORT"], "4100")
        self.assertEqual(
            env["NEXT_PUBLIC_API_URL"],
            "http://localhost:9123/api",
        )
        self.assertEqual(original["NEXT_PUBLIC_API_URL"], "http://stale/api")

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

    def test_backend_env_merges_custom_origin_without_duplicates(self) -> None:
        builder = getattr(dev, "build_backend_env", None)
        self.assertIsNotNone(builder, "build_backend_env should be defined")

        env = builder(
            {
                "PIPIXIA_CORS_ORIGINS": (
                    "https://example.test, http://localhost:4100"
                ),
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


if __name__ == "__main__":
    unittest.main()
