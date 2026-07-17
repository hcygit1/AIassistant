import { expect, test } from "@playwright/test";

import { createChatStreamEventHandler } from "../src/lib/chatStreamEvents";
import { getAgentChatRuntime } from "../src/lib/chatState";
import type { AgentChatState, ChatMessage } from "../src/lib/chatState";

test("reduces token, tool and done events for one assistant message", () => {
  let messages: ChatMessage[] = [{
    id: "assistant-1",
    role: "assistant",
    content: "",
    createdAt: 10,
    isStreaming: true,
    toolCalls: [],
  }];
  const patches: Partial<AgentChatState>[] = [];
  const streamState = { doneReceived: false, terminalErrorReceived: false };
  const runtime = getAgentChatRuntime(new Map(), "main");
  runtime.assistantMessageId = "assistant-1";
  const handler = createChatStreamEventHandler({
    assistantMessageId: "assistant-1",
    runtime,
    streamState,
    updateMessages: (updater: (previous: ChatMessage[]) => ChatMessage[]) => {
      messages = updater(messages);
    },
    patchState: (patch: Partial<AgentChatState>) => patches.push(patch),
  });

  handler({ type: "token", content: "working" });
  handler({ type: "tool_start", tool: "read", input: { path: "README.md" } });
  handler({ type: "tool_end", tool: "read", output: "ok" });
  handler({
    type: "done",
    content: "final",
    usage: { input_tokens: 2, output_tokens: 3, total_tokens: 5, duration_ms: 12 },
    context_utilization: 0.25,
  });

  expect(streamState).toEqual({ doneReceived: true, terminalErrorReceived: false });
  expect(messages[0]).toMatchObject({
    content: "final",
    isStreaming: false,
    streamDurationMs: 12,
    toolCalls: [{ tool: "read", output: "ok" }],
  });
  expect(patches).toContainEqual({
    lastUsage: { input_tokens: 2, output_tokens: 3, total_tokens: 5, duration_ms: 12 },
    contextUtilization: 0.25,
  });
});

test("forwards lifecycle and compact events without mutating messages", () => {
  let lifecycleCount = 0;
  let compactCount = 0;
  const messages: ChatMessage[] = [];
  const runtime = getAgentChatRuntime(new Map(), "writer");
  const handler = createChatStreamEventHandler({
    assistantMessageId: "assistant-2",
    runtime,
    streamState: { doneReceived: false, terminalErrorReceived: false },
    updateMessages: () => {
      throw new Error("message state should not change");
    },
    patchState: () => {},
    addLifecycleEvent: () => { lifecycleCount += 1; },
    onSessionCompacted: () => { compactCount += 1; },
  });

  handler({ type: "lifecycle", event: "skills_updated" });
  handler({ type: "session_compacted" });

  expect(messages).toEqual([]);
  expect(lifecycleCount).toBe(1);
  expect(compactCount).toBe(1);
});
