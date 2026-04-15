"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import * as api from "../api";
import type { SSEEvent, TokenUsage } from "../api";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system" | "command";
  content: string;
  createdAt: number;
  finishedAt?: number;
  streamDurationMs?: number;
  toolCalls?: { tool?: string; name?: string; input?: any; output?: string; result?: string }[];
  retrievals?: any[];
  isStreaming?: boolean;
  usage?: TokenUsage;
}

export interface LifecycleEvent {
  type: string;
  event: string;
  run_id?: string;
  timestamp: number;
  data?: any;
}

interface UseChatOptions {
  onAgentCreated?: () => void;
  onSessionCompacted?: () => void;
  onTurnComplete?: () => void;
  formatCommandResponse?: (raw: string) => string;
}

export function useChat(
  currentAgentId: string,
  currentSessionId: string | null,
  setCurrentSessionId: (id: string | null) => void,
  options?: UseChatOptions,
) {
  const [messagesByAgent, setMessagesByAgent] = useState<Map<string, ChatMessage[]>>(new Map());
  const [isStreamingByAgent, setIsStreamingByAgent] = useState<Map<string, boolean>>(new Map());

  const messages = messagesByAgent.get(currentAgentId) || [];
  const isStreaming = isStreamingByAgent.get(currentAgentId) || false;

  const setMessages = useCallback((updater: React.SetStateAction<ChatMessage[]>) => {
    setMessagesByAgent(prev => {
      const next = new Map(prev);
      const current = next.get(currentAgentId) || [];
      const updated = typeof updater === "function" ? updater(current) : updater;
      next.set(currentAgentId, updated);
      return next;
    });
  }, [currentAgentId]);

  const setIsStreaming = useCallback((value: boolean) => {
    setIsStreamingByAgent(prev => {
      const next = new Map(prev);
      next.set(currentAgentId, value);
      return next;
    });
  }, [currentAgentId]);

  const [lifecycleEvents, setLifecycleEvents] = useState<LifecycleEvent[]>([]);
  const [lastUsage, setLastUsage] = useState<TokenUsage | null>(null);
  const [contextUtilization, setContextUtilization] = useState<number | null>(null);
  const [sessionError, setSessionError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const chatTimeoutRef = useRef<number | null>(null);
  const userStoppedRef = useRef(false);
  const streamingAssistantIdRef = useRef<string | null>(null);
  const currentTurnIdRef = useRef<string | null>(null);
  const sendMessageRef = useRef<((text: string) => Promise<void>) | null>(null);
  const segmentToolCallsRef = useRef<{ tool: string; input: any; output: string }[]>([]);
  const isStreamingRef = useRef(false);
  useEffect(() => {
    isStreamingRef.current = isStreaming;
  }, [isStreaming]);

  const addLifecycleEvent = useCallback((event: SSEEvent) => {
    if (event.type === "lifecycle" && event.event) {
      setLifecycleEvents(prev => [...prev, {
        type: event.type,
        event: event.event!,
        run_id: event.run_id,
        timestamp: Date.now(),
        data: event.usage || event,
      }]);
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
      setMessagesByAgent(prev => {
        const next = new Map(prev);
        next.set(agentId, msgs);
        return next;
      });
      setSessionError(null);
      return msgs;
    } catch {
      setMessagesByAgent(prev => {
        const next = new Map(prev);
        next.set(agentId, []);
        return next;
      });
      return null;
    }
  }, []);

  const createStreamEventHandler = useCallback(
    (assistantMsgId: string, streamState: { doneReceived: boolean; terminalErrorReceived: boolean }) => {
      segmentToolCallsRef.current = [];
      return (event: SSEEvent) => {
        switch (event.type) {
          case "token":
            setMessages(prev => {
              const targetId = streamingAssistantIdRef.current || assistantMsgId;
              const idx = prev.findIndex(m => m.id === targetId);
              const last = idx >= 0 ? prev[idx] : null;
              if (idx >= 0 && last?.role === "assistant") {
                const updated = prev.slice();
                updated[idx] = { ...last, content: last.content + (event.content || "") };
                return updated;
              }
              return prev;
            });
            break;

          case "clear_content":
            setMessages(prev => {
              const targetId = streamingAssistantIdRef.current || assistantMsgId;
              const idx = prev.findIndex(m => m.id === targetId);
              const last = idx >= 0 ? prev[idx] : null;
              if (idx >= 0 && last?.role === "assistant") {
                const updated = prev.slice();
                updated[idx] = { ...last, content: "" };
                return updated;
              }
              return prev;
            });
            break;

          case "content_refresh":
            setMessages(prev => {
              const targetId = streamingAssistantIdRef.current || assistantMsgId;
              const idx = prev.findIndex(m => m.id === targetId);
              const last = idx >= 0 ? prev[idx] : null;
              if (idx >= 0 && last?.role === "assistant" && typeof event.content === "string") {
                const updated = prev.slice();
                updated[idx] = { ...last, content: event.content };
                return updated;
              }
              return prev;
            });
            break;

          case "tool_start": {
            const newTc = {
              tool: event.tool || event.name || "",
              input: event.input ?? event.args ?? {},
              output: "",
            };
            segmentToolCallsRef.current = [...segmentToolCallsRef.current, newTc];
            setMessages(prev => {
              const targetId = streamingAssistantIdRef.current || assistantMsgId;
              const idx = prev.findIndex(m => m.id === targetId);
              const last = idx >= 0 ? prev[idx] : null;
              if (idx >= 0 && last?.role === "assistant") {
                const updated = prev.slice();
                updated[idx] = { ...last, toolCalls: segmentToolCallsRef.current };
                return updated;
              }
              return prev;
            });
            break;
          }

          case "tool_end": {
            const output = event.output || event.result || "";
            const toolName = event.tool || event.name || "";
            setMessages(prev => {
              const targetId = streamingAssistantIdRef.current || assistantMsgId;
              const idx = prev.findIndex(m => m.id === targetId);
              const last = idx >= 0 ? prev[idx] : null;
              if (idx >= 0 && last?.role === "assistant" && last.toolCalls?.length) {
                const tc = last.toolCalls;
                const targetIdx = tc.findIndex(
                  t => !(t.output ?? t.result) && (!toolName || (t.tool || t.name) === toolName)
                );
                const fallbackIdx = targetIdx >= 0 ? targetIdx : tc.findIndex(t => !(t.output ?? t.result));
                const toUpdate = fallbackIdx >= 0 ? fallbackIdx : tc.length - 1;
                const newToolCalls = tc.map((t, i) =>
                  i === toUpdate ? { ...t, output } : t
                );
                const updated = prev.slice();
                updated[idx] = { ...last, toolCalls: newToolCalls };
                return updated;
              }
              return prev;
            });
            if (segmentToolCallsRef.current.length > 0) {
              const segIdx = segmentToolCallsRef.current.findIndex(
                t => !t.output && (!toolName || t.tool === toolName)
              );
              const segFallback = segIdx >= 0 ? segIdx : segmentToolCallsRef.current.findIndex(t => !t.output);
              const segToUpdate = segFallback >= 0 ? segFallback : segmentToolCallsRef.current.length - 1;
              segmentToolCallsRef.current = segmentToolCallsRef.current.map((t, i) =>
                i === segToUpdate ? { ...t, output } : t
              );
            }
            break;
          }

          case "new_response":
            segmentToolCallsRef.current = [];
            setMessages(prev => {
              const targetId = streamingAssistantIdRef.current || assistantMsgId;
              const idx = prev.findIndex(m => m.id === targetId);
              const last = idx >= 0 ? prev[idx] : null;
              if (idx >= 0 && last?.role === "assistant") {
                const updated = prev.slice();
                updated[idx] = { ...last, isStreaming: false };
                return updated;
              }
              return prev;
            });
            break;

          case "retrieval":
            setMessages(prev => {
              const targetId = streamingAssistantIdRef.current || assistantMsgId;
              const idx = prev.findIndex(m => m.id === targetId);
              const last = idx >= 0 ? prev[idx] : null;
              if (idx >= 0 && last?.role === "assistant") {
                const updated = prev.slice();
                updated[idx] = { ...last, retrievals: event.results || [] };
                return updated;
              }
              return prev;
            });
            break;

          case "command_response": {
            const formattedResponse = options?.formatCommandResponse
              ? options.formatCommandResponse(event.response || "")
              : (event.response || "");
            const text = (formattedResponse || "").trim();
            if (!text) break;
            setMessages(prev => {
              const idx = prev.length - 1;
              const last = prev[idx];
              const commandMsg: ChatMessage = {
                id: `command-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
                role: "command" as any,
                content: text,
                createdAt: Date.now(),
                isStreaming: false,
              };
              if (last?.role === "assistant" && !(last.content || "").trim()) {
                const updated = prev.slice();
                updated[idx] = commandMsg;
                return updated;
              }
              return [...prev, commandMsg];
            });
            break;
          }

          case "session_reset": {
            setLifecycleEvents([]);
            setLastUsage(null);
            const newAssistantId = `assistant-${Date.now()}`;
            streamingAssistantIdRef.current = newAssistantId;
            setMessages(prev => {
              const last = prev[prev.length - 1];
              if (last?.role === "command") {
                return [...prev, {
                  id: newAssistantId,
                  role: "assistant",
                  content: "",
                  createdAt: Date.now(),
                  isStreaming: true,
                }];
              }
              return prev;
            });
            break;
          }

          case "session_compacted":
            options?.onSessionCompacted?.();
            break;

          case "lifecycle":
            addLifecycleEvent(event);
            break;

          case "title":
            break;

          case "done":
            streamState.doneReceived = true;
            if (event.usage) setLastUsage(event.usage);
            if (event.context_utilization != null) setContextUtilization(event.context_utilization);
            setMessages(prev => {
              const targetId = streamingAssistantIdRef.current || assistantMsgId;
              const idx = prev.findIndex(m => m.id === targetId);
              const last = idx >= 0 ? prev[idx] : null;
              if (idx >= 0 && last && (last.role === "assistant" || last.role === "command")) {
                const finishedAt = Date.now();
                const estimatedDuration =
                  event.usage?.duration_ms && event.usage.duration_ms > 0
                    ? event.usage.duration_ms
                    : Math.max(0, finishedAt - (last.createdAt || finishedAt));
                const updated = prev.slice();
                updated[idx] = {
                  ...last,
                  isStreaming: false,
                  finishedAt,
                  streamDurationMs: estimatedDuration,
                  ...(event.usage ? { usage: event.usage } : {}),
                  ...(typeof event.content === "string" ? { content: event.content } : {}),
                };
                return updated;
              }
              return prev;
            });
            break;

          case "aborted":
            streamState.doneReceived = true;
            setMessages(prev => {
              const targetId = streamingAssistantIdRef.current || assistantMsgId;
              const idx = prev.findIndex(m => m.id === targetId);
              const last = idx >= 0 ? prev[idx] : null;
              if (idx >= 0 && last?.role === "assistant") {
                const updated = prev.slice();
                updated[idx] = {
                  ...last,
                  content:
                    typeof event.content === "string" && event.content.length > 0
                      ? event.content
                      : last.content,
                  isStreaming: false,
                  finishedAt: Date.now(),
                };
                return updated;
              }
              return prev;
            });
            break;

          case "error":
            streamState.terminalErrorReceived = true;
            streamState.doneReceived = true;
            setMessages(prev => {
              const targetId = streamingAssistantIdRef.current || assistantMsgId;
              const idx = prev.findIndex(m => m.id === targetId);
              const last = idx >= 0 ? prev[idx] : null;
              if (idx >= 0 && last?.role === "assistant") {
                const err = event.error || "";
                const friendly = err.includes("401") || err.includes("invalid") || err.includes("Authentication")
                  ? "**API Authentication Failed**: Please check the apiKey for the corresponding provider in config.json.\n\nOriginal error: " + err
                  : `**Error:** ${err}`;
                const updated = prev.slice();
                updated[idx] = { ...last, content: last.content + `\n\n${friendly}`, isStreaming: false };
                return updated;
              }
              return prev;
            });
            break;
        }
      };
    },
    [addLifecycleEvent, options, setMessages],
  );

  const finalizeStreamTurn = useCallback(
    async (
      assistantMsgId: string,
      sessionId: string | null,
      streamState: { doneReceived: boolean; terminalErrorReceived: boolean },
      dequeueLocal: boolean,
    ) => {
      currentTurnIdRef.current = null;
      const stoppedByUser = userStoppedRef.current;
      if (streamingAssistantIdRef.current === assistantMsgId) {
        streamingAssistantIdRef.current = null;
      }
      setIsStreaming(false);
      abortRef.current = null;
      userStoppedRef.current = false;
      if (!streamState.doneReceived) {
        if (!stoppedByUser && !streamState.terminalErrorReceived) {
          try {
            if (sessionId) await loadMessages(currentAgentId, sessionId);
          } catch { /* best-effort reload */ }
        }
      } else {
        setMessages(prev => {
          const targetId = streamingAssistantIdRef.current || assistantMsgId;
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
      options?.onTurnComplete?.();
      if (dequeueLocal) {
        try {
          const { dequeue } = await import("../messageQueue");
          const next = dequeue();
          if (next) {
            setTimeout(() => {
              void sendMessageRef.current?.(next.text);
            }, 50);
          }
        } catch { /* ignore */ }
      }
    },
    [currentAgentId, loadMessages, options, setMessages],
  );

  const sendMessage = useCallback(async (text: string) => {
    if (!text.trim()) return;

    if (isStreaming) {
      const { enqueue } = await import("../messageQueue");
      enqueue(text.trim());
      const now = Date.now();
      setMessages(prev => [...prev, {
        id: `user-${now}`,
        role: "user" as const,
        content: text.trim(),
        createdAt: now,
      }]);
      return;
    }

    let sessionId = currentSessionId;
    if (!sessionId) {
      try {
        const session = await api.fetchMainSession(currentAgentId);
        sessionId = session.session_id;
        setCurrentSessionId(sessionId);
        setSessionError(null);
      } catch (e: any) {
        setSessionError(e.message || "Failed to fetch session");
        return;
      }
    }

    const now = Date.now();
    const userMsg: ChatMessage = {
      id: `user-${now}`,
      role: "user",
      content: text,
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
    streamingAssistantIdRef.current = assistantMsgId;

    setMessages(prev => [...prev, userMsg, assistantMsg]);
    setIsStreaming(true);

    const controller = new AbortController();
    abortRef.current = controller;
    userStoppedRef.current = false;

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
      const sub = await api.submitChat(text, sessionId!, currentAgentId);
      currentTurnIdRef.current = sub.turn_id;
      const shouldStream = await api.waitUntilTurnRunning(sub.turn_id, controller.signal);
      if (!shouldStream) {
        streamState.doneReceived = true;
        await loadMessages(currentAgentId, sessionId!);
      } else {
        const onEvent = createStreamEventHandler(assistantMsgId, streamState);
        await api.streamTurn(sub.turn_id, onEvent, { signal: controller.signal, timeoutMs });
      }
    } catch (e: any) {
      if (e.name !== "AbortError") {
        streamState.terminalErrorReceived = true;
        streamState.doneReceived = true;
        const friendly = (e.message || "").includes("timeout")
          ? e.message
          : `**Connection error:** ${e.message}`;
        setMessages(prev => {
          const targetId = streamingAssistantIdRef.current || assistantMsgId;
          const idx = prev.findIndex(m => m.id === targetId);
          const last = idx >= 0 ? prev[idx] : null;
          if (idx >= 0 && last?.role === "assistant") {
            const updated = prev.slice();
            updated[idx] = { ...last, content: last.content + `\n\n${friendly}`, isStreaming: false };
            return updated;
          }
          return prev;
        });
      } else if (userStoppedRef.current) {
        // 用户手动停止
      }
    } finally {
      await finalizeStreamTurn(assistantMsgId, sessionId, streamState, true);
    }
  }, [
    currentAgentId,
    currentSessionId,
    isStreaming,
    createStreamEventHandler,
    finalizeStreamTurn,
    loadMessages,
    setCurrentSessionId,
  ]);

  sendMessageRef.current = sendMessage;

  /** 刷新后：若服务端仍有未完成的用户 turn，拉历史并续接 SSE */
  useEffect(() => {
    if (!currentSessionId) return;
    let cancelled = false;

    void (async () => {
      const streamState = { doneReceived: false, terminalErrorReceived: false };
      let streamAssistantId = "";
      try {
        const p = await api.fetchPendingTurn(currentSessionId, currentAgentId);
        if (cancelled) return;
        if (!p.turn_id || (p.status !== "queued" && p.status !== "running")) return;
        if (abortRef.current) return;
        if (isStreamingRef.current) return;

        const msgs = await loadMessages(currentAgentId, currentSessionId);
        if (cancelled || msgs === null) return;

        const last = msgs[msgs.length - 1];
        const resumeAssistantId = `assistant-resume-${p.turn_id}`;

        if (last?.role === "user") {
          streamAssistantId = resumeAssistantId;
          streamingAssistantIdRef.current = resumeAssistantId;
          setMessages(prev => [...prev, {
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
          streamingAssistantIdRef.current = last.id;
          setMessages(prev =>
            prev.map((m, i) =>
              i === prev.length - 1 ? { ...m, isStreaming: true } : m
            )
          );
        } else {
          streamAssistantId = resumeAssistantId;
          streamingAssistantIdRef.current = resumeAssistantId;
          setMessages(prev => [...prev, {
            id: resumeAssistantId,
            role: "assistant",
            content: "",
            createdAt: Date.now(),
            toolCalls: [],
            retrievals: [],
            isStreaming: true,
          }]);
        }

        setIsStreaming(true);
        const controller = new AbortController();
        abortRef.current = controller;
        currentTurnIdRef.current = p.turn_id;
        userStoppedRef.current = false;

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
            if (currentSessionId) await loadMessages(currentAgentId, currentSessionId);
            return;
          }
        }
        if (cancelled) return;

        const onEvent = createStreamEventHandler(streamAssistantId, streamState);
        await api.streamTurn(p.turn_id, onEvent, { signal: controller.signal, timeoutMs });
      } catch (e: any) {
        if (e.name !== "AbortError") {
          streamState.terminalErrorReceived = true;
          streamState.doneReceived = true;
          try {
            if (currentSessionId) await loadMessages(currentAgentId, currentSessionId);
          } catch { /* ignore */ }
        }
      } finally {
        if (!streamAssistantId) return;
        if (cancelled) {
          currentTurnIdRef.current = null;
          streamingAssistantIdRef.current = null;
          abortRef.current = null;
          setIsStreaming(false);
          return;
        }
        await finalizeStreamTurn(streamAssistantId, currentSessionId, streamState, false);
      }
    })();

    return () => {
      cancelled = true;
    };
    // 仅随会话变化重试恢复；避免 createStreamEventHandler 等引用变化导致反复执行
    // eslint-disable-next-line react-hooks/exhaustive-deps -- 同上
  }, [currentAgentId, currentSessionId]);

  const stopStreaming = useCallback(async () => {
    userStoppedRef.current = true;
    const sessionId = currentSessionId;
    if (sessionId) {
      try {
        await api.abortChat(currentAgentId, sessionId, {
          userInitiated: true,
          turnId: currentTurnIdRef.current ?? undefined,
        });
      } catch {
        // 后端 abort 失败时，降级为前端本地断流。
      }
    }
    abortRef.current?.abort();
    setIsStreaming(false);
    const targetAssistantId = streamingAssistantIdRef.current;
    setMessages(prev => {
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
  }, [currentAgentId, currentSessionId]);

  const clearChat = useCallback(() => {
    setMessagesByAgent(prev => {
      const next = new Map(prev);
      next.delete(currentAgentId);
      return next;
    });
    setIsStreamingByAgent(prev => {
      const next = new Map(prev);
      next.delete(currentAgentId);
      return next;
    });
    setLifecycleEvents([]);
    setLastUsage(null);
    setSessionError(null);
  }, [currentAgentId]);

  useEffect(() => {
    const currentStreaming = isStreamingByAgent.get(currentAgentId);
    if (currentStreaming) {
      if (!abortRef.current) {
        setIsStreamingByAgent(prev => {
          const next = new Map(prev);
          next.set(currentAgentId, false);
          return next;
        });
      }
    }
  }, [currentAgentId]);

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
  };
}
