import { expect, test } from "@playwright/test";

import { submitChatTurn } from "../src/lib/chatTurnSubmission";
import { getAgentChatRuntime } from "../src/lib/chatState";
import { clear, dequeue } from "../src/lib/messageQueue";
import type {
  AgentChatState,
  ChatMessage,
} from "../src/lib/chatState";

test.beforeEach(() => clear());
test.afterEach(() => clear());

test("queues a message while the agent already owns a turn", async () => {
  const runtime = getAgentChatRuntime(new Map(), "main");
  let messages: ChatMessage[] = [];

  await submitChatTurn({
    agentId: "main",
    text: " queued ",
    knownSessionId: "session-1",
    runtime,
    isStreaming: true,
    updateMessages: (updater) => { messages = updater(messages); },
    patchState: () => {},
    getTimeoutMs: async () => undefined,
    loadMessages: async () => [],
    createEventHandler: () => () => {},
    finalize: async () => {},
    now: () => 100,
  });

  expect(dequeue("main")).toMatchObject({
    agentId: "main",
    text: "queued",
    messageId: "user-100",
  });
  expect(messages).toEqual([{
    id: "user-100",
    role: "user",
    content: "queued",
    createdAt: 100,
  }]);
});

test("resolves the main session and streams one submitted turn", async () => {
  const runtime = getAgentChatRuntime(new Map(), "main");
  let messages: ChatMessage[] = [];
  const patches: Partial<AgentChatState>[] = [];
  const resolvedSessions: string[] = [];
  const finalized: Array<{
    assistantId: string;
    sessionId: string;
    doneReceived: boolean;
  }> = [];

  await submitChatTurn({
    agentId: "main",
    text: " hello ",
    knownSessionId: null,
    runtime,
    isStreaming: false,
    onSessionResolved: (sessionId) => resolvedSessions.push(sessionId),
    updateMessages: (updater) => { messages = updater(messages); },
    patchState: (patch) => patches.push(patch),
    getTimeoutMs: async () => 5000,
    loadMessages: async () => messages,
    createEventHandler: (_assistantId, streamState) => (
      () => { streamState.doneReceived = true; }
    ),
    finalize: async (assistantId, sessionId, streamState) => {
      finalized.push({
        assistantId,
        sessionId,
        doneReceived: streamState.doneReceived,
      });
    },
    transport: {
      fetchMainSession: async () => ({ session_id: "session-1" }),
      submitChat: async () => ({
        turn_id: "turn-1",
        position: 1,
        status: "queued",
        session_id: "session-1",
      }),
      waitUntilTurnRunning: async () => true,
      streamTurn: async (_turnId, onEvent) => { onEvent({ type: "done" }); },
    },
    now: () => 100,
  });

  expect(resolvedSessions).toEqual(["session-1"]);
  expect(runtime).toMatchObject({
    sessionId: "session-1",
    turnId: "turn-1",
    assistantMessageId: "assistant-100",
  });
  expect(messages.map((message) => message.id)).toEqual([
    "user-100",
    "assistant-100",
  ]);
  expect(patches).toContainEqual({ isStreaming: true });
  expect(finalized).toEqual([{
    assistantId: "assistant-100",
    sessionId: "session-1",
    doneReceived: true,
  }]);
});

test("records a connection error and still finalizes the turn", async () => {
  const runtime = getAgentChatRuntime(new Map(), "main");
  let messages: ChatMessage[] = [];
  const finalized: Array<{ done: boolean; terminal: boolean }> = [];

  await submitChatTurn({
    agentId: "main",
    text: "hello",
    knownSessionId: "session-1",
    runtime,
    isStreaming: false,
    updateMessages: (updater) => { messages = updater(messages); },
    patchState: () => {},
    getTimeoutMs: async () => undefined,
    loadMessages: async () => messages,
    createEventHandler: () => () => {},
    finalize: async (_assistantId, _sessionId, streamState) => {
      finalized.push({
        done: streamState.doneReceived,
        terminal: streamState.terminalErrorReceived,
      });
    },
    transport: {
      fetchMainSession: async () => ({ session_id: "session-1" }),
      submitChat: async () => { throw new Error("offline"); },
      waitUntilTurnRunning: async () => true,
      streamTurn: async () => {},
    },
    now: () => 100,
  });

  expect(messages[1]).toMatchObject({
    id: "assistant-100",
    content: "\n\n**Connection error:** offline",
    isStreaming: false,
  });
  expect(finalized).toEqual([{ done: true, terminal: true }]);
});
