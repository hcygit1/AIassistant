"use client";

import { useCallback, useRef, useEffect, useReducer } from "react";
import * as api from "../api";
import type { SSEEvent } from "../api";
import { createChatStreamEventHandler } from "../chatStreamEvents";
import {
  chatStateReducer,
  clearAgentChatRuntime,
  createChatState,
  getAgentChatRuntime,
  appendTurnMessages,
  selectAgentChatState,
} from "../chatState";
import { clear as clearQueuedMessages, dequeue, enqueue } from "../messageQueue";
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
    async (
      agentId: string,
      assistantMsgId: string,
      sessionId: string | null,
      streamState: { doneReceived: boolean; terminalErrorReceived: boolean },
      dequeueLocal: boolean,
    ) => {
      const runtime = getAgentChatRuntime(runtimeRegistryRef.current, agentId);
      runtime.turnId = null;
      const stoppedByUser = runtime.userStopped;
      if (runtime.assistantMessageId === assistantMsgId) {
        runtime.assistantMessageId = null;
      }
      patchAgentState(agentId, { isStreaming: false });
      runtime.controller = null;
      runtime.userStopped = false;
      if (!streamState.doneReceived) {
        if (!stoppedByUser && !streamState.terminalErrorReceived) {
          try {
            if (sessionId) await loadMessages(agentId, sessionId);
          } catch { /* best-effort reload */ }
        }
      } else {
        setMessagesForAgent(agentId, prev => {
          const targetId = runtime.assistantMessageId || assistantMsgId;
          const idx = prev.findIndex(m => m.id === targetId);
          const last = idx >= 0 ? prev[idx] : null;
          if (idx >= 0 && last?.role === "assistant" && last.isStreaming) {
            const updated = prev.slice();
            updated[idx] = { ...last, isStreaming: false, finishedAt: Date.now() };
            return updated;
          }
          return prev;
        });
      }
      options?.onTurnComplete?.(agentId);
      if (dequeueLocal) {
        try {
          const next = dequeue(agentId);
          if (next) {
            setTimeout(() => {
              void queuedSenderRef.current(
                agentId,
                next.text,
                runtime.sessionId,
                next.messageId,
              );
            }, 50);
          }
        } catch { /* ignore */ }
      }
    },
    [loadMessages, options, patchAgentState, setMessagesForAgent],
  );

  const sendMessageForAgent = useCallback(async (
    agentId: string,
    text: string,
    knownSessionId: string | null,
    displayedMessageId?: string,
  ) => {
    const normalized = text.trim();
    if (!normalized) return;

    const runtime = getAgentChatRuntime(runtimeRegistryRef.current, agentId);
    const agentState = selectAgentChatState(chatState, agentId);
    if (agentState.isStreaming || runtime.controller) {
      const now = Date.now();
      const messageId = displayedMessageId || `user-${now}`;
      enqueue(agentId, normalized, messageId);
      if (!displayedMessageId) {
        setMessagesForAgent(agentId, (previous) => [...previous, {
          id: messageId,
          role: "user",
          content: normalized,
          createdAt: now,
        }]);
      }
      return;
    }

    let sessionId = knownSessionId || runtime.sessionId;
    if (!sessionId) {
      try {
        const session = await api.fetchMainSession(agentId);
        sessionId = session.session_id;
        if (agentId === currentAgentId) setCurrentSessionId(sessionId);
        patchAgentState(agentId, { sessionError: null });
      } catch (error: unknown) {
        const message = error instanceof Error ? error.message : "Failed to fetch session";
        patchAgentState(agentId, { sessionError: message });
        return;
      }
    }
    const resolvedSessionId = sessionId;
    if (!resolvedSessionId) return;
    runtime.sessionId = resolvedSessionId;

    const now = Date.now();
    const userMsg: ChatMessage = {
      id: displayedMessageId || `user-${now}`,
      role: "user",
      content: normalized,
      createdAt: now,
    };
    const assistantMsg: ChatMessage = {
      id: `assistant-${now}`,
      role: "assistant",
      content: "",
      createdAt: now,
      toolCalls: [],
      retrievals: [],
      isStreaming: true,
    };
    const assistantMsgId = assistantMsg.id;
    runtime.assistantMessageId = assistantMsgId;

    setMessagesForAgent(agentId, (previous) => appendTurnMessages(
      previous,
      userMsg,
      assistantMsg,
      displayedMessageId,
    ));
    patchAgentState(agentId, { isStreaming: true });

    const controller = new AbortController();
    runtime.controller = controller;
    runtime.userStopped = false;

    const streamState = { doneReceived: false, terminalErrorReceived: false };

    let timeoutMs: number | undefined;
    if (chatTimeoutRef.current === null) {
      try {
        const cfg = await api.fetchChatTimeout();
        chatTimeoutRef.current = cfg.timeoutSeconds ?? 120;
      } catch {
        chatTimeoutRef.current = 120;
      }
    }
    if (chatTimeoutRef.current !== null && chatTimeoutRef.current > 0) {
      timeoutMs = chatTimeoutRef.current * 1000;
    }

    try {
      const sub = await api.submitChat(normalized, resolvedSessionId, agentId);
      runtime.turnId = sub.turn_id;
      const shouldStream = await api.waitUntilTurnRunning(sub.turn_id, controller.signal);
      if (!shouldStream) {
        streamState.doneReceived = true;
        await loadMessages(agentId, resolvedSessionId);
      } else {
        const onEvent = createStreamEventHandler(agentId, assistantMsgId, streamState);
        await api.streamTurn(sub.turn_id, onEvent, { signal: controller.signal, timeoutMs });
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
        setMessagesForAgent(agentId, (previous) => {
          const targetId = runtime.assistantMessageId || assistantMsgId;
          const idx = previous.findIndex((item) => item.id === targetId);
          const last = idx >= 0 ? previous[idx] : null;
          if (idx >= 0 && last?.role === "assistant") {
            const updated = previous.slice();
            updated[idx] = { ...last, content: last.content + `\n\n${friendly}`, isStreaming: false };
            return updated;
          }
          return previous;
        });
      }
    } finally {
      await finalizeStreamTurn(agentId, assistantMsgId, resolvedSessionId, streamState, true);
    }
  }, [
    chatState,
    createStreamEventHandler,
    currentAgentId,
    finalizeStreamTurn,
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
    runtime.sessionId = sessionId;
    let cancelled = false;

    void (async () => {
      const streamState = { doneReceived: false, terminalErrorReceived: false };
      let streamAssistantId = "";
      try {
        const p = await api.fetchPendingTurn(sessionId, agentId);
        if (cancelled) return;
        if (!p.turn_id || (p.status !== "queued" && p.status !== "running")) return;
        if (runtime.controller) return;

        const msgs = await loadMessages(agentId, sessionId);
        if (cancelled || msgs === null) return;

        const last = msgs[msgs.length - 1];
        const resumeAssistantId = `assistant-resume-${p.turn_id}`;

        if (last?.role === "user") {
          streamAssistantId = resumeAssistantId;
          runtime.assistantMessageId = resumeAssistantId;
          setMessagesForAgent(agentId, prev => [...prev, {
            id: resumeAssistantId,
            role: "assistant",
            content: "",
            createdAt: Date.now(),
            toolCalls: [],
            retrievals: [],
            isStreaming: true,
          }]);
        } else if (last?.role === "assistant") {
          streamAssistantId = last.id;
          runtime.assistantMessageId = last.id;
          setMessagesForAgent(agentId, prev =>
            prev.map((m, i) =>
              i === prev.length - 1 ? { ...m, isStreaming: true } : m
            )
          );
        } else {
          streamAssistantId = resumeAssistantId;
          runtime.assistantMessageId = resumeAssistantId;
          setMessagesForAgent(agentId, prev => [...prev, {
            id: resumeAssistantId,
            role: "assistant",
            content: "",
            createdAt: Date.now(),
            toolCalls: [],
            retrievals: [],
            isStreaming: true,
          }]);
        }

        patchAgentState(agentId, { isStreaming: true });
        const controller = new AbortController();
        runtime.controller = controller;
        runtime.turnId = p.turn_id;
        runtime.userStopped = false;

        let timeoutMs: number | undefined;
        if (chatTimeoutRef.current === null) {
          try {
            const cfg = await api.fetchChatTimeout();
            chatTimeoutRef.current = cfg.timeoutSeconds ?? 120;
          } catch {
            chatTimeoutRef.current = 120;
          }
        }
        if (chatTimeoutRef.current !== null && chatTimeoutRef.current > 0) {
          timeoutMs = chatTimeoutRef.current * 1000;
        }

        if (p.status === "queued") {
          const shouldStream = await api.waitUntilTurnRunning(p.turn_id, controller.signal);
          if (!shouldStream) {
            streamState.doneReceived = true;
            await loadMessages(agentId, sessionId);
            return;
          }
        }
        if (cancelled) return;

        const onEvent = createStreamEventHandler(agentId, streamAssistantId, streamState);
        await api.streamTurn(p.turn_id, onEvent, { signal: controller.signal, timeoutMs });
      } catch (error: unknown) {
        if (!(error instanceof Error && error.name === "AbortError")) {
          streamState.terminalErrorReceived = true;
          streamState.doneReceived = true;
          try {
            await loadMessages(agentId, sessionId);
          } catch { /* ignore */ }
        }
      } finally {
        if (!streamAssistantId) return;
        if (cancelled) {
          if (runtime.assistantMessageId === streamAssistantId) {
            runtime.turnId = null;
            runtime.assistantMessageId = null;
            runtime.controller = null;
            patchAgentState(agentId, { isStreaming: false });
          }
          return;
        }
        await finalizeStreamTurn(agentId, streamAssistantId, sessionId, streamState, false);
      }
    })();

    return () => {
      cancelled = true;
    };
    // 仅随会话变化重试恢复；避免 createStreamEventHandler 等引用变化导致反复执行
    // eslint-disable-next-line react-hooks/exhaustive-deps -- 同上
  }, [currentAgentId, currentSessionId]);

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
