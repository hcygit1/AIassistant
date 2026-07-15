import { expect, test } from "@playwright/test";

import { getTurnStatus, resolveApiBase } from "../src/lib/api";

const originalFetch = globalThis.fetch;

test.afterEach(() => {
  globalThis.fetch = originalFetch;
});

test.describe("resolveApiBase", () => {
  test("prefers and normalizes the public API URL", () => {
    expect(
      resolveApiBase("  http://localhost:9123/api/  ", "ignored-host"),
    ).toBe("http://localhost:9123/api");
  });

  test("falls back to the browser hostname on port 8002", () => {
    expect(resolveApiBase(undefined, "devbox.local")).toBe(
      "http://devbox.local:8002/api",
    );
  });

  test("uses the configured API port with the browser hostname", () => {
    expect(resolveApiBase(undefined, "devbox.local", "9123")).toBe(
      "http://devbox.local:9123/api",
    );
  });
});

test.describe("getTurnStatus", () => {
  test("returns a terminal status reported by the backend", async () => {
    globalThis.fetch = async () =>
      new Response(
        JSON.stringify({
          turn_id: "turn-1",
          status: "done",
          position: 0,
          session_id: "main-main",
          agent_id: "main",
          error: null,
        }),
        {
          status: 200,
          headers: { "Content-Type": "application/json" },
        },
      );

    await expect(getTurnStatus("turn-1")).resolves.toMatchObject({
      turn_id: "turn-1",
      status: "done",
    });
  });

  test("does not treat an unknown turn as completed", async () => {
    globalThis.fetch = async () =>
      new Response(
        JSON.stringify({ detail: "unknown turn_id" }),
        {
          status: 404,
          headers: { "Content-Type": "application/json" },
        },
      );

    await expect(getTurnStatus("missing-turn")).rejects.toThrow(
      "unknown turn_id",
    );
  });
});
