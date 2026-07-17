import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";

import type { SSEEvent, SubagentTreeItem } from "../src/lib/api";
import * as subagentState from "../src/lib/subagentState";

function subagent(overrides: Partial<SubagentTreeItem> = {}): SubagentTreeItem {
  return {
    run_id: "run-1",
    label: "Researcher",
    task: "Investigate the current implementation",
    target_agent_id: "research",
    status: "running",
    state: "running",
    elapsed: 3,
    result_summary: "",
    messages: [],
    created_at: 1,
    result_delivery_state: "pending",
    ...overrides,
  };
}

test("derives running and inline subagent views from one flat snapshot", () => {
  const deriveSubagentViews = Reflect.get(subagentState, "deriveSubagentViews");
  expect(typeof deriveSubagentViews).toBe("function");

  const longTask = "x".repeat(80);
  const flat = [
    subagent({ task: longTask }),
    subagent({
      run_id: "run-2",
      status: "succeeded",
      state: "succeeded",
      messages: [{ role: "assistant", content: "done" }],
      elapsed: 7,
      result_summary: "summary",
      result_delivery_state: "delivered",
      terminal_reason: "completed",
    }),
  ];

  const views = deriveSubagentViews(flat, "run-2");
  expect(views.runningSubagents).toEqual([
    { run_id: "run-1", task: "x".repeat(60), status: "running" },
  ]);
  expect(views.inline).toMatchObject({
    messages: [{ role: "assistant", content: "done" }],
    state: "succeeded",
    elapsed: 7,
    result_summary: "summary",
    result_delivery_state: "delivered",
    terminal_reason: "completed",
    label: "Researcher",
  });
});

test("maps every supported subagent event without changing panel text semantics", () => {
  const mapSubagentEvent = Reflect.get(subagentState, "mapSubagentEvent");
  expect(typeof mapSubagentEvent).toBe("function");
  const at = 1234;
  const cases: Array<[SSEEvent, string, string, boolean]> = [
    [{ type: "subagent_start", run_id: "r", task: "task" } as SSEEvent, "start", "开始执行：task", true],
    [{ type: "subagent_tool", run_id: "r", tool: "read" }, "tool", "调用工具：read", true],
    [{ type: "subagent_tool_end", run_id: "r", tool: "read", output_preview: "ok" } as SSEEvent, "tool_end", "工具完成：read -> ok", true],
    [{ type: "subagent_progress", run_id: "r", elapsed_s: 4, chars: 20 } as SSEEvent, "progress", "执行中：4s，输出 20 chars", false],
    [{ type: "subagent_done", run_id: "r", result: "result" }, "done", "执行完成，等待结果合并：result", true],
    [{ type: "subagent_error", run_id: "r", error: "bad" }, "error", "执行失败：bad", true],
    [{ type: "subagent_killed", run_id: "r" }, "killed", "已终止", true],
    [{ type: "subagent_announce", run_id: "r", result_delivery_state: "queued" } as SSEEvent, "announce", "等待主会话队列", true],
  ];

  for (const [event, type, text, shouldRefresh] of cases) {
    expect(mapSubagentEvent(event, at)).toEqual({
      runId: "r",
      trace: { ts: at, type, text },
      shouldRefresh,
      doneDelayMs: null,
    });
  }

  expect(mapSubagentEvent({ type: "subagent_announce", run_id: "r", result_delivery_state: "delivered" }, at)?.doneDelayMs).toBe(250);
  expect(mapSubagentEvent({ type: "subagent_announce", run_id: "r", result_delivery_state: "dropped" }, at)?.doneDelayMs).toBe(100);
  expect(mapSubagentEvent({ type: "message", run_id: "r" }, at)).toBeNull();
});

test("ignores stale responses after the agent or session scope changes", () => {
  const createSubagentState = Reflect.get(subagentState, "createSubagentState");
  const subagentStateReducer = Reflect.get(subagentState, "subagentStateReducer");
  expect(typeof createSubagentState).toBe("function");
  expect(typeof subagentStateReducer).toBe("function");

  const oldScope = "main\u0000session-1";
  const newScope = "writer\u0000session-2";
  const initial = createSubagentState(oldScope);
  const switched = subagentStateReducer(initial, { type: "scope", scopeKey: newScope });
  const stale = subagentStateReducer(switched, {
    type: "success",
    scopeKey: oldScope,
    tree: [subagent()],
    flat: [subagent()],
  });

  expect(stale).toBe(switched);
  expect(stale).toMatchObject({ scopeKey: newScope, tree: [], flat: [], traceMap: {}, loading: false });
});

test("keeps fetch polling and subagent event subscription in the shared hook only", () => {
  const hook = readFileSync("src/lib/hooks/useSubagents.ts", "utf8");
  const panel = readFileSync("src/components/inspector/SubagentPanel.tsx", "utf8");
  const inline = readFileSync("src/components/chat/SubagentInlineCard.tsx", "utf8");

  expect(hook.match(/api\.fetchSubagents\(/g) ?? []).toHaveLength(1);
  expect(hook.match(/api\.subscribeAgentEvents\(/g) ?? []).toHaveLength(1);
  expect(panel).not.toContain("fetchSubagents(");
  expect(panel).not.toContain("subscribeAgentEvents(");
  expect(inline).not.toContain("fetchSubagents(");
  expect(inline).not.toContain("subscribeAgentEvents(");
});
