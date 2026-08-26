import { expect, test } from "@playwright/test";

import { finalizeChatTurn } from "../src/lib/chatTurnFinalization";
import { getAgentChatRuntime } from "../src/lib/chatState";
import type {
  AgentChatState,
  ChatMessage,
} from "../src/lib/chatState";

test("clears runtime and finishes a completed assistant message", async () => {
  const runtime = getAgentChatRuntime(new Map(), "main");
  runtime.sessionId = "session-1";
  runtime.turnId = "turn-1";
  runtime.assistantMessageId = "assistant-1";
  runtime.controller = new AbortController();
  runtime.userStopped = true;
  let messages: ChatMessage[] = [{
    id: "assistant-1",
    role: "assistant",
    content: "done",
    createdAt: 1,
    isStreaming: true,
  }];
  const patches: Partial<AgentChatState>[] = [];
  const completed: string[] = [];

  await finalizeChatTurn({
    agentId: "main",
    assistantMessageId: "assistant-1",
    sessionId: "session-1",
    streamState: { doneReceived: true, terminalErrorReceived: false },
    runtime,
    dequeueLocal: false,
    updateMessages: (updater) => { messages = updater(messages); },
    patchState: (patch) => patches.push(patch),
    loadMessages: async () => null,
    onTurnComplete: (agentId) => completed.push(agentId),
    sendQueuedMessage: async () => {},
    now: () => 100,
  });

  expect(runtime).toMatchObject({
    turnId: null,
    assistantMessageId: null,
    controller: null,
    userStopped: false,
  });
  expect(messages[0]).toMatchObject({
    isStreaming: false,
    finishedAt: 100,
  });
  expect(patches).toEqual([{ isStreaming: false }]);
  expect(completed).toEqual(["main"]);
});

test("only reloads an interrupted non-terminal turn and never dequeues recovery", async () => {
  const runtime = getAgentChatRuntime(new Map(), "main");
  let reloads = 0;
  let dequeues = 0;
  const dequeueMessage = () => {
    dequeues += 1;
    return {
      agentId: "main",
      text: "queued",
      messageId: "user-queued",
      timestamp: 1,
    };
  };

  await finalizeChatTurn({
    agentId: "main",
    assistantMessageId: "assistant-1",
    sessionId: "session-1",
    streamState: { doneReceived: false, terminalErrorReceived: false },
    runtime,
    dequeueLocal: false,
    updateMessages: () => {},
    patchState: () => {},
    loadMessages: async () => { reloads += 1; return []; },
    sendQueuedMessage: async () => {},
    dequeueMessage,
  });

  runtime.userStopped = true;
  await finalizeChatTurn({
    agentId: "main",
    assistantMessageId: "assistant-2",
    sessionId: "session-1",
    streamState: { doneReceived: false, terminalErrorReceived: false },
    runtime,
    dequeueLocal: false,
    updateMessages: () => {},
    patchState: () => {},
    loadMessages: async () => { reloads += 1; return []; },
    sendQueuedMessage: async () => {},
    dequeueMessage,
  });

  await finalizeChatTurn({
    agentId: "main",
    assistantMessageId: "assistant-3",
    sessionId: "session-1",
    streamState: { doneReceived: false, terminalErrorReceived: true },
    runtime,
    dequeueLocal: false,
    updateMessages: () => {},
    patchState: () => {},
    loadMessages: async () => { reloads += 1; return []; },
    sendQueuedMessage: async () => {},
    dequeueMessage,
  });

  expect(reloads).toBe(2);
  expect(dequeues).toBe(0);
});

test("dequeues and schedules the next local message after submission", async () => {
  const runtime = getAgentChatRuntime(new Map(), "main");
  runtime.sessionId = "session-1";
  const sent: unknown[][] = [];
  let scheduledDelay = 0;

  await finalizeChatTurn({
    agentId: "main",
    assistantMessageId: "assistant-1",
    sessionId: "session-1",
    streamState: { doneReceived: true, terminalErrorReceived: false },
    runtime,
    dequeueLocal: true,
    updateMessages: () => {},
    patchState: () => {},
    loadMessages: async () => [],
    sendQueuedMessage: async (...args) => { sent.push(args); },
    dequeueMessage: () => ({
      agentId: "main",
      text: "next",
      messageId: "user-2",
      timestamp: 1,
    }),
    schedule: (callback, delay) => {
      scheduledDelay = delay;
      callback();
    },
  });

  expect(scheduledDelay).toBe(50);
  expect(sent).toEqual([[
    "main",
    "next",
    "session-1",
    "user-2",
  ]]);
});
