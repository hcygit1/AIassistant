import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";

import { clear, dequeue, enqueue } from "../src/lib/messageQueue";
import {
  chatStateReducer,
  clearAgentChatRuntime,
  createChatState,
  getAgentChatRuntime,
  selectAgentChatState,
} from "../src/lib/chatState";
import * as chatStateModule from "../src/lib/chatState";
import type { ChatMessage } from "../src/lib/chatState";

test.beforeEach(() => {
  clear();
});

test.afterEach(() => {
  clear();
});

test("keeps queued messages isolated by agent", () => {
  enqueue("main", "first", "user-main");
  enqueue("writer", "second", "user-writer");

  expect(dequeue("writer")).toEqual({
    agentId: "writer",
    text: "second",
    messageId: "user-writer",
    timestamp: expect.any(Number),
  });
  expect(dequeue("main")).toEqual({
    agentId: "main",
    text: "first",
    messageId: "user-main",
    timestamp: expect.any(Number),
  });
});

test("clears one agent queue without dropping another agent messages", () => {
  enqueue("main", "first");
  enqueue("writer", "second");

  clear("main");

  expect(dequeue("main")).toBeNull();
  expect(dequeue("writer")?.text).toBe("second");
});

test("updates and clears chat view state for one agent only", () => {
  let state = createChatState();
  state = chatStateReducer(state, {
    type: "messages",
    agentId: "main",
    updater: [{ id: "m1", role: "user", content: "main", createdAt: 1 }],
  });
  state = chatStateReducer(state, {
    type: "lifecycle",
    agentId: "main",
    event: { type: "lifecycle", event: "turn_started", timestamp: 3 },
  });
  state = chatStateReducer(state, {
    type: "messages",
    agentId: "writer",
    updater: [{ id: "w1", role: "user", content: "writer", createdAt: 2 }],
  });
  state = chatStateReducer(state, {
    type: "patch",
    agentId: "main",
    patch: { isStreaming: true, sessionError: "main error" },
  });

  expect(selectAgentChatState(state, "main")).toMatchObject({
    messages: [{ id: "m1", content: "main" }],
    isStreaming: true,
    sessionError: "main error",
    lifecycleEvents: [{ event: "turn_started" }],
  });
  expect(selectAgentChatState(state, "writer")).toMatchObject({
    messages: [{ id: "w1", content: "writer" }],
    isStreaming: false,
    sessionError: null,
  });

  state = chatStateReducer(state, { type: "clear", agentId: "main" });
  expect(selectAgentChatState(state, "main").messages).toEqual([]);
  expect(selectAgentChatState(state, "writer").messages).toHaveLength(1);
});

test("keeps active turn runtime isolated by agent", () => {
  const registry = new Map();
  const main = getAgentChatRuntime(registry, "main");
  const writer = getAgentChatRuntime(registry, "writer");

  main.turnId = "turn-main";
  main.assistantMessageId = "assistant-main";
  main.userStopped = true;
  main.segmentToolCalls.push({ tool: "read", input: {}, output: "" });

  expect(writer).not.toBe(main);
  expect(writer).toMatchObject({
    turnId: null,
    assistantMessageId: null,
    userStopped: false,
    segmentToolCalls: [],
  });
  expect(getAgentChatRuntime(registry, "main")).toBe(main);

  clearAgentChatRuntime(registry, "main");
  expect(getAgentChatRuntime(registry, "main")).not.toBe(main);
  expect(getAgentChatRuntime(registry, "writer")).toBe(writer);
});

test("routes active turn refs and local queue operations through agent scope", () => {
  const source = readFileSync("src/lib/hooks/useChat.ts", "utf8");

  for (const legacyRef of [
    "abortRef",
    "userStoppedRef",
    "streamingAssistantIdRef",
    "currentTurnIdRef",
    "segmentToolCallsRef",
    "sendMessageRef",
  ]) {
    expect(source).not.toContain(legacyRef);
  }
  expect(source).toContain("getAgentChatRuntime(runtimeRegistryRef.current, agentId)");
  expect(source).toContain("enqueue(agentId, normalized, messageId)");
  expect(source).toContain("dequeue(agentId)");
  expect(source).toContain("appendTurnMessages(");
});

test("inserts a queued assistant directly after its displayed user message", () => {
  const appendTurnMessages = Reflect.get(chatStateModule, "appendTurnMessages");
  expect(typeof appendTurnMessages).toBe("function");

  const messages: ChatMessage[] = [
    { id: "assistant-current", role: "assistant", content: "done", createdAt: 1 },
    { id: "user-1", role: "user", content: "first", createdAt: 2 },
    { id: "user-2", role: "user", content: "second", createdAt: 3 },
  ];
  const assistant: ChatMessage = { id: "assistant-1", role: "assistant", content: "", createdAt: 4 };

  expect(appendTurnMessages(messages, null, assistant, "user-1").map((item: { id: string }) => item.id)).toEqual([
    "assistant-current",
    "user-1",
    "assistant-1",
    "user-2",
  ]);
});
