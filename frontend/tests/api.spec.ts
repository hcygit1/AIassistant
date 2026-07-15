import { afterEach, describe, expect, test } from "@playwright/test";

import { getTurnStatus } from "../src/lib/api";

const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
});

describe("getTurnStatus", () => {
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
