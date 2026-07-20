"use client";

import { useCallback, useRef, useEffect, useReducer } from "react";
import * as api from "../api";
import type { SSEEvent } from "../api";
import { createChatStreamEventHandler } from "../chatStreamEvents";
import { finalizeChatTurn } from "../chatTurnFinalization";
import { submitChatTurn } from "../chatTurnSubmission";
import { startPendingTurnRecovery } from "../chatTurnRecovery";
import {
  chatStateReducer,
  clearAgentChatRuntime,
  createChatState,
  getAgentChatRuntime,
  selectAgentChatState,
} from "../chatState";
import { clear as clearQueuedMessages } from "../messageQueue";
import type {
  AgentChatState,
  ChatMessage,
  ChatRuntimeRegistry,
} from "../chatState";

export type { ChatMessage, LifecycleEvent } from "../chatState";

interface UseChatOptions {
  onAgentCreated?: (agentId: string) => void;
  onSessionCompacted?: (agentId: string) => void;
  onTurnComplete?: (agentId: string) => void;
  formatCommandResponse?: (raw: string) => string;
}

export function useChat(
  currentAgentId: string,
  currentSessionId: string | null,
  setCurrentSessionId: (id: string | null) => void,
  options?: UseChatOptions,
) {
  const [chatState, dispatch] = useReducer(chatStateReducer, undefined, createChatState);
  const currentState = selectAgentChatState(chatState, currentAgentId);
  const {
    messages,
    isStreaming,
    lifecycleEvents,
    lastUsage,
    contextUtilization,
    sessionError,
  } = currentState;

  const setMessagesForAgent = useCallback((agentId: string, updater: React.SetStateAction<ChatMessage[]>) => {
    dispatch({ type: "messages", agentId, updater });
  }, []);

  const patchAgentState = useCallback((agentId: string, patch: Partial<AgentChatState>) => {
    dispatch({ type: "patch", agentId, patch });
  }, []);

  const setMessages = useCallback((updater: React.SetStateAction<ChatMessage[]>) => {
    setMessagesForAgent(currentAgentId, updater);
  }, [currentAgentId, setMessagesForAgent]);

  const chatTimeoutRef = useRef<number | null>(null);
  const runtimeRegistryRef = useRef<ChatRuntimeRegistry>(new Map());
  const queuedSenderRef = useRef<(
    agentId: string,
    text: string,
    sessionId: string | null,
    displayedMessageId?: string,
  ) => Promise<void>>(async () => {});

  const addLifecycleEvent = useCallback((agentId: string, event: SSEEvent) => {
    if (event.type === "lifecycle" && event.event) {
      dispatch({ type: "lifecycle", agentId, event: {
        type: event.type,
        event: event.event!,
        run_id: event.run_id,
        timestamp: Date.now(),
        data: event.usage || event,
      } });
    }
  }, []);

  const getTimeoutMs = useCallback(async (): Promise<number | undefined> => {
    if (chatTimeoutRef.current === null) {
      try {
        const config = await api.fetchChatTimeout();
        chatTimeoutRef.current = config.timeoutSeconds ?? 120;
      } catch {
        chatTimeoutRef.current = 120;
      }
    }
    return chatTimeoutRef.current > 0
      ? chatTimeoutRef.current * 1000
      : undefined;
  }, []);

  const loadMessages = useCallback(async (agentId: string, sessionId: string): Promise<ChatMessage[] | null> => {
    try {
      const data = await api.fetchMainSessionMessages(agentId);
      const now = Date.now();
      const msgs: ChatMessage[] = (data.messages || []).map((m: any, i: number) => ({
        id: `${sessionId}-${i}`,
        role: m.role,
        content: m.content,
        createdAt: now,
        toolCalls: m.tool_calls,
      }));
      setMessagesForAgent(agentId, msgs);
      patchAgentState(agentId, { sessionError: null });
      return msgs;
    } catch {
      setMessagesForAgent(agentId, []);
      return null;
    }
  }, [patchAgentState, setMessagesForAgent]);

  const createStreamEventHandler = useCallback(
    (
      agentId: string,
      assistantMessageId: string,
      streamState: { doneReceived: boolean; terminalErrorReceived: boolean },
    ) => {
      const runtime = getAgentChatRuntime(runtimeRegistryRef.current, agentId);
      return createChatStreamEventHandler({
        assistantMessageId,
        runtime,
        streamState,
        updateMessages: (updater) => setMessagesForAgent(agentId, updater),
        patchState: (patch) => patchAgentState(agentId, patch),
        addLifecycleEvent: (event) => addLifecycleEvent(agentId, event),
        onSessionCompacted: () => options?.onSessionCompacted?.(agentId),
        formatCommandResponse: options?.formatCommandResponse,
      });
    },
    [addLifecycleEvent, options, patchAgentState, setMessagesForAgent],
  );

  const finalizeStreamTurn = useCallback(
    (
      agentId: string,
      assistantMsgId: string,
      sessionId: string | null,
      streamState: { doneReceived: boolean; terminalErrorReceived: boolean },
      dequeueLocal: boolean,
    ) => {
      const runtime = getAgentChatRuntime(runtimeRegistryRef.current, agentId);
      return finalizeChatTurn({
        agentId,
        assistantMessageId: assistantMsgId,
        sessionId,
        streamState,
        runtime,
        dequeueLocal,
        updateMessages: (updater) => setMessagesForAgent(agentId, updater),
        patchState: (patch) => patchAgentState(agentId, patch),
        loadMessages,
        onTurnComplete: options?.onTurnComplete,
        sendQueuedMessage: (...args) => queuedSenderRef.current(...args),
      });
    },
    [loadMessages, options, patchAgentState, setMessagesForAgent],
  );

  const sendMessageForAgent = useCallback(async (
    agentId: string,
    text: string,
    knownSessionId: string | null,
    displayedMessageId?: string,
  ) => {
    const runtime = getAgentChatRuntime(runtimeRegistryRef.current, agentId);
    const agentState = selectAgentChatState(chatState, agentId);
    await submitChatTurn({
      agentId,
      text,
      knownSessionId,
      displayedMessageId,
      runtime,
      isStreaming: agentState.isStreaming,
      onSessionResolved: agentId === currentAgentId
        ? setCurrentSessionId
        : undefined,
      updateMessages: (updater) => setMessagesForAgent(agentId, updater),
      patchState: (patch) => patchAgentState(agentId, patch),
      getTimeoutMs,
      loadMessages,
      createEventHandler: (assistantMessageId, streamState) => (
        createStreamEventHandler(agentId, assistantMessageId, streamState)
      ),
      finalize: (assistantMessageId, sessionId, streamState) => (
        finalizeStreamTurn(
          agentId,
          assistantMessageId,
          sessionId,
          streamState,
          true,
        )
      ),
    });
  }, [
    chatState,
    createStreamEventHandler,
    currentAgentId,
    finalizeStreamTurn,
    getTimeoutMs,
    loadMessages,
    patchAgentState,
    setCurrentSessionId,
    setMessagesForAgent,
  ]);

  useEffect(() => {
    queuedSenderRef.current = sendMessageForAgent;
  }, [sendMessageForAgent]);

  const sendMessage = useCallback((text: string) => (
    sendMessageForAgent(currentAgentId, text, currentSessionId)
  ), [currentAgentId, currentSessionId, sendMessageForAgent]);

  /** 刷新后：若服务端仍有未完成的用户 turn，拉历史并续接 SSE */
  useEffect(() => {
    if (!currentSessionId) return;
    const agentId = currentAgentId;
    const sessionId = currentSessionId;
    const runtime = getAgentChatRuntime(runtimeRegistryRef.current, agentId);
    const recovery = startPendingTurnRecovery({
      agentId,
      sessionId,
      runtime,
      loadMessages: () => loadMessages(agentId, sessionId),
      updateMessages: (updater) => setMessagesForAgent(agentId, updater),
      patchState: (patch) => patchAgentState(agentId, patch),
      getTimeoutMs,
      createEventHandler: (assistantMessageId, streamState) => (
        createStreamEventHandler(agentId, assistantMessageId, streamState)
      ),
      finalize: (assistantMessageId, streamState) => (
        finalizeStreamTurn(agentId, assistantMessageId, sessionId, streamState, false)
      ),
    });
    return recovery.cancel;
  }, [
    createStreamEventHandler,
    currentAgentId,
    currentSessionId,
    finalizeStreamTurn,
    getTimeoutMs,
    loadMessages,
    patchAgentState,
    setMessagesForAgent,
  ]);

  const stopStreaming = useCallback(async () => {
    const agentId = currentAgentId;
    const runtime = getAgentChatRuntime(runtimeRegistryRef.current, agentId);
    runtime.userStopped = true;
    const sessionId = currentSessionId || runtime.sessionId;
    if (sessionId) {
      try {
        await api.abortChat(agentId, sessionId, {
          userInitiated: true,
          turnId: runtime.turnId ?? undefined,
        });
      } catch {
        // 后端 abort 失败时，降级为前端本地断流。
      }
    }
    runtime.controller?.abort();
    patchAgentState(agentId, { isStreaming: false });
    const targetAssistantId = runtime.assistantMessageId;
    setMessagesForAgent(agentId, prev => {
      if (!targetAssistantId) return prev;
      const idx = prev.findIndex(m => m.id === targetAssistantId);
      const last = idx >= 0 ? prev[idx] : null;
      if (idx >= 0 && last?.role === "assistant" && last.isStreaming) {
        const updated = prev.slice();
        updated[idx] = { ...last, isStreaming: false, finishedAt: Date.now() };
        return updated;
      }
      return prev;
    });
  }, [currentAgentId, currentSessionId, patchAgentState, setMessagesForAgent]);

  const clearAgent = useCallback((agentId: string) => {
    const runtime = runtimeRegistryRef.current.get(agentId);
    if (!runtime?.controller) {
      clearAgentChatRuntime(runtimeRegistryRef.current, agentId);
    }
    clearQueuedMessages(agentId);
    dispatch({ type: "clear", agentId });
  }, []);

  const clearChat = useCallback(() => {
    clearAgent(currentAgentId);
  }, [clearAgent, currentAgentId]);

  const setSessionError = useCallback((value: string | null) => {
    patchAgentState(currentAgentId, { sessionError: value });
  }, [currentAgentId, patchAgentState]);

  return {
    messages,
    setMessages,
    isStreaming,
    lifecycleEvents,
    lastUsage,
    contextUtilization,
    sessionError,
    setSessionError,
    sendMessage,
    stopStreaming,
    loadMessages,
    clearChat,
    clearAgent,
  };
}
