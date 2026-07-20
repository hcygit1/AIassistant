import * as api from "./api";
import type { ChatSubmitResponse, SSEEvent } from "./api";
import { appendTurnMessages } from "./chatState";
import type {
  AgentChatRuntime,
  AgentChatState,
  ChatMessage,
} from "./chatState";
import type { ChatStreamState } from "./chatStreamEvents";
import { enqueue } from "./messageQueue";

export interface ChatSubmissionTransport {
  fetchMainSession: (agentId: string) => Promise<{ session_id: string }>;
  submitChat: (
    message: string,
    sessionId: string,
    agentId: string,
  ) => Promise<ChatSubmitResponse>;
  waitUntilTurnRunning: (
    turnId: string,
    signal: AbortSignal,
  ) => Promise<boolean>;
  streamTurn: (
    turnId: string,
    onEvent: (event: SSEEvent) => void,
    options: { signal: AbortSignal; timeoutMs?: number },
  ) => Promise<void>;
}

interface ChatTurnSubmissionOptions {
  agentId: string;
  text: string;
  knownSessionId: string | null;
  displayedMessageId?: string;
  runtime: AgentChatRuntime;
  isStreaming: boolean;
  onSessionResolved?: (sessionId: string) => void;
  updateMessages: (
    updater: (messages: ChatMessage[]) => ChatMessage[],
  ) => void;
  patchState: (patch: Partial<AgentChatState>) => void;
  getTimeoutMs: () => Promise<number | undefined>;
  loadMessages: (
    agentId: string,
    sessionId: string,
  ) => Promise<ChatMessage[] | null>;
  createEventHandler: (
    assistantMessageId: string,
    streamState: ChatStreamState,
  ) => (event: SSEEvent) => void;
  finalize: (
    assistantMessageId: string,
    sessionId: string,
    streamState: ChatStreamState,
  ) => Promise<void>;
  transport?: ChatSubmissionTransport;
  now?: () => number;
}

const defaultTransport: ChatSubmissionTransport = {
  fetchMainSession: api.fetchMainSession,
  submitChat: api.submitChat,
  waitUntilTurnRunning: api.waitUntilTurnRunning,
  streamTurn: api.streamTurn,
};

export async function submitChatTurn({
  agentId,
  text,
  knownSessionId,
  displayedMessageId,
  runtime,
  isStreaming,
  onSessionResolved,
  updateMessages,
  patchState,
  getTimeoutMs,
  loadMessages,
  createEventHandler,
  finalize,
  transport = defaultTransport,
  now = Date.now,
}: ChatTurnSubmissionOptions): Promise<void> {
  const normalized = text.trim();
  if (!normalized) return;

  if (isStreaming || runtime.controller) {
    const createdAt = now();
    const messageId = displayedMessageId || `user-${createdAt}`;
    enqueue(agentId, normalized, messageId);
    if (!displayedMessageId) {
      updateMessages((previous) => [...previous, {
        id: messageId,
        role: "user",
        content: normalized,
        createdAt,
      }]);
    }
    return;
  }

  let sessionId = knownSessionId || runtime.sessionId;
  if (!sessionId) {
    try {
      const session = await transport.fetchMainSession(agentId);
      sessionId = session.session_id;
      onSessionResolved?.(sessionId);
      patchState({ sessionError: null });
    } catch (error: unknown) {
      const message = error instanceof Error
        ? error.message
        : "Failed to fetch session";
      patchState({ sessionError: message });
      return;
    }
  }
  const resolvedSessionId = sessionId;
  if (!resolvedSessionId) return;
  runtime.sessionId = resolvedSessionId;

  const createdAt = now();
  const userMessage: ChatMessage = {
    id: displayedMessageId || `user-${createdAt}`,
    role: "user",
    content: normalized,
    createdAt,
  };
  const assistantMessage: ChatMessage = {
    id: `assistant-${createdAt}`,
    role: "assistant",
    content: "",
    createdAt,
    toolCalls: [],
    retrievals: [],
    isStreaming: true,
  };
  const assistantMessageId = assistantMessage.id;
  runtime.assistantMessageId = assistantMessageId;

  updateMessages((previous) => appendTurnMessages(
    previous,
    userMessage,
    assistantMessage,
    displayedMessageId,
  ));
  patchState({ isStreaming: true });

  const controller = new AbortController();
  runtime.controller = controller;
  runtime.userStopped = false;
  const streamState: ChatStreamState = {
    doneReceived: false,
    terminalErrorReceived: false,
  };
  const timeoutMs = await getTimeoutMs();

  try {
    const submitted = await transport.submitChat(
      normalized,
      resolvedSessionId,
      agentId,
    );
    runtime.turnId = submitted.turn_id;
    const shouldStream = await transport.waitUntilTurnRunning(
      submitted.turn_id,
      controller.signal,
    );
    if (!shouldStream) {
      streamState.doneReceived = true;
      await loadMessages(agentId, resolvedSessionId);
    } else {
      const onEvent = createEventHandler(
        assistantMessageId,
        streamState,
      );
      await transport.streamTurn(
        submitted.turn_id,
        onEvent,
        { signal: controller.signal, timeoutMs },
      );
    }
  } catch (error: unknown) {
    const isAbort = error instanceof Error && error.name === "AbortError";
    if (!isAbort) {
      streamState.terminalErrorReceived = true;
      streamState.doneReceived = true;
      const message = error instanceof Error ? error.message : String(error);
      const friendly = message.includes("timeout")
        ? message
        : `**Connection error:** ${message}`;
      updateMessages((previous) => {
        const targetId = runtime.assistantMessageId || assistantMessageId;
        const index = previous.findIndex((item) => item.id === targetId);
        const last = index >= 0 ? previous[index] : null;
        if (index >= 0 && last?.role === "assistant") {
          const updated = previous.slice();
          updated[index] = {
            ...last,
            content: last.content + `\n\n${friendly}`,
            isStreaming: false,
          };
          return updated;
        }
        return previous;
      });
    }
  } finally {
    await finalize(
      assistantMessageId,
      resolvedSessionId,
      streamState,
    );
  }
}
