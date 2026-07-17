import { expect, test } from "@playwright/test";

import { startPendingTurnRecovery } from "../src/lib/chatTurnRecovery";
import { getAgentChatRuntime } from "../src/lib/chatState";
import type { AgentChatState, ChatMessage } from "../src/lib/chatState";

test("restores a running turn and finalizes its assistant stream", async () => {
  let messages: ChatMessage[] = [{
    id: "user-1",
    role: "user",
    content: "continue",
    createdAt: 1,
  }];
  const patches: Partial<AgentChatState>[] = [];
  const finalized: Array<{ assistantId: string; doneReceived: boolean }> = [];
  const runtime = getAgentChatRuntime(new Map(), "main");

  const recovery = startPendingTurnRecovery({
    agentId: "main",
    sessionId: "session-1",
    runtime,
    loadMessages: async () => messages,
    updateMessages: (updater: (previous: ChatMessage[]) => ChatMessage[]) => {
      messages = updater(messages);
    },
    patchState: (patch: Partial<AgentChatState>) => patches.push(patch),
    getTimeoutMs: async () => 5000,
    createEventHandler: (_assistantId: string, streamState: { doneReceived: boolean }) => (
      () => { streamState.doneReceived = true; }
    ),
    finalize: async (assistantId: string, streamState: { doneReceived: boolean }) => {
      finalized.push({ assistantId, doneReceived: streamState.doneReceived });
    },
    transport: {
      fetchPendingTurn: async () => ({ turn_id: "turn-1", status: "running" }),
      waitUntilTurnRunning: async () => true,
      streamTurn: async (_turnId: string, onEvent: (event: { type: string }) => void) => {
        onEvent({ type: "done" });
      },
    },
  });

  await recovery.done;

  expect(messages.map((message) => message.id)).toEqual(["user-1", "assistant-resume-turn-1"]);
  expect(runtime).toMatchObject({
    sessionId: "session-1",
    turnId: "turn-1",
    assistantMessageId: "assistant-resume-turn-1",
  });
  expect(patches).toContainEqual({ isStreaming: true });
  expect(finalized).toEqual([{ assistantId: "assistant-resume-turn-1", doneReceived: true }]);
});

test("does not restore when another turn already owns the agent runtime", async () => {
  const runtime = getAgentChatRuntime(new Map(), "main");
  runtime.controller = new AbortController();
  let loadCount = 0;

  const recovery = startPendingTurnRecovery({
    agentId: "main",
    sessionId: "session-1",
    runtime,
    loadMessages: async () => { loadCount += 1; return []; },
    updateMessages: () => {},
    patchState: () => {},
    getTimeoutMs: async () => undefined,
    createEventHandler: () => () => {},
    finalize: async () => {},
    transport: {
      fetchPendingTurn: async () => ({ turn_id: "turn-1", status: "running" }),
      waitUntilTurnRunning: async () => true,
      streamTurn: async () => {},
    },
  });

  await recovery.done;
  expect(loadCount).toBe(0);
});
