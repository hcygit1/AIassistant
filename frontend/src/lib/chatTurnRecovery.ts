import * as api from "./api";
import type { SSEEvent } from "./api";
import type {
  AgentChatRuntime,
  AgentChatState,
  ChatMessage,
} from "./chatState";
import type { ChatStreamState } from "./chatStreamEvents";

interface PendingTurn {
  turn_id?: string | null;
  status?: string | null;
}

export interface ChatRecoveryTransport {
  fetchPendingTurn: (sessionId: string, agentId: string) => Promise<PendingTurn>;
  waitUntilTurnRunning: (turnId: string, signal: AbortSignal) => Promise<boolean>;
  streamTurn: (
    turnId: string,
    onEvent: (event: SSEEvent) => void,
    options: { signal: AbortSignal; timeoutMs?: number },
  ) => Promise<void>;
}

export interface PendingTurnRecoveryOptions {
  agentId: string;
  sessionId: string;
  runtime: AgentChatRuntime;
  loadMessages: () => Promise<ChatMessage[] | null>;
  updateMessages: (updater: (messages: ChatMessage[]) => ChatMessage[]) => void;
  patchState: (patch: Partial<AgentChatState>) => void;
  getTimeoutMs: () => Promise<number | undefined>;
  createEventHandler: (
    assistantMessageId: string,
    streamState: ChatStreamState,
  ) => (event: SSEEvent) => void;
  finalize: (
    assistantMessageId: string,
    streamState: ChatStreamState,
  ) => Promise<void>;
  transport?: ChatRecoveryTransport;
}

export interface PendingTurnRecovery {
  cancel: () => void;
  done: Promise<void>;
}

const defaultTransport: ChatRecoveryTransport = {
  fetchPendingTurn: api.fetchPendingTurn,
  waitUntilTurnRunning: api.waitUntilTurnRunning,
  streamTurn: api.streamTurn,
};

export function startPendingTurnRecovery({
  agentId,
  sessionId,
  runtime,
  loadMessages,
  updateMessages,
  patchState,
  getTimeoutMs,
  createEventHandler,
  finalize,
  transport = defaultTransport,
}: PendingTurnRecoveryOptions): PendingTurnRecovery {
  let cancelled = false;
  runtime.sessionId = sessionId;

  const done = (async () => {
    const streamState: ChatStreamState = {
      doneReceived: false,
      terminalErrorReceived: false,
    };
    let assistantMessageId = "";

    try {
      const pending = await transport.fetchPendingTurn(sessionId, agentId);
      if (cancelled) return;
      if (!pending.turn_id || (pending.status !== "queued" && pending.status !== "running")) return;
      if (runtime.controller) return;

      const messages = await loadMessages();
      if (cancelled || messages === null) return;

      const last = messages[messages.length - 1];
      const resumeAssistantId = `assistant-resume-${pending.turn_id}`;
      if (last?.role === "assistant") {
        assistantMessageId = last.id;
        runtime.assistantMessageId = last.id;
        updateMessages((current) => current.map((message, index) => (
          index === current.length - 1 ? { ...message, isStreaming: true } : message
        )));
      } else {
        assistantMessageId = resumeAssistantId;
        runtime.assistantMessageId = resumeAssistantId;
        updateMessages((current) => [...current, {
          id: resumeAssistantId,
          role: "assistant",
          content: "",
          createdAt: Date.now(),
          toolCalls: [],
          retrievals: [],
          isStreaming: true,
        }]);
      }

      patchState({ isStreaming: true });
      const controller = new AbortController();
      runtime.controller = controller;
      runtime.turnId = pending.turn_id;
      runtime.userStopped = false;

      const timeoutMs = await getTimeoutMs();
      if (pending.status === "queued") {
        const shouldStream = await transport.waitUntilTurnRunning(pending.turn_id, controller.signal);
        if (!shouldStream) {
          streamState.doneReceived = true;
          await loadMessages();
          return;
        }
      }
      if (cancelled) return;

      const onEvent = createEventHandler(assistantMessageId, streamState);
      await transport.streamTurn(pending.turn_id, onEvent, { signal: controller.signal, timeoutMs });
    } catch (error: unknown) {
      if (!(error instanceof Error && error.name === "AbortError")) {
        streamState.terminalErrorReceived = true;
        streamState.doneReceived = true;
        try {
          await loadMessages();
        } catch {
          // Best-effort recovery reload.
        }
      }
    } finally {
      if (!assistantMessageId) return;
      if (cancelled) {
        if (runtime.assistantMessageId === assistantMessageId) {
          runtime.turnId = null;
          runtime.assistantMessageId = null;
          runtime.controller = null;
          patchState({ isStreaming: false });
        }
        return;
      }
      await finalize(assistantMessageId, streamState);
    }
  })();

  return {
    cancel: () => { cancelled = true; },
    done,
  };
}
