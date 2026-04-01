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
  onSubagentEvent?: () => void;
  onTurnComplete?: () => void;
  formatCommandResponse?: (raw: string) => string;
}

export function useChat(
  currentAgentId: string,
  currentSessionId: string | null,
  setCurrentSessionId: (id: string | null) => void,
  options?: UseChatOptions,
) {
  // 按 Agent 存储消息状态
  const [messagesByAgent, setMessagesByAgent] = useState<Map<string, ChatMessage[]>>(new Map());
  const [isStreamingByAgent, setIsStreamingByAgent] = useState<Map<string, boolean>>(new Map());

  // 当前 Agent 的消息和状态
  const messages = messagesByAgent.get(currentAgentId) || [];
  const isStreaming = isStreamingByAgent.get(currentAgentId) || false;

  const setMessages = useCallback((updater: React.SetStateAction<ChatMessage[]>) => {
    setMessagesByAgent(prev => {
      const next = new Map(prev);
      const current = next.get(currentAgentId) || [];
      const updated = typeof updater === 'function' ? updater(current) : updater;
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

  const loadMessages = useCallback(async (agentId: string, sessionId: string) => {
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
      // 使用 agentId 直接设置消息，避免依赖闭包
      setMessagesByAgent(prev => {
        const next = new Map(prev);
        next.set(agentId, msgs);
        return next;
      });
      setSessionError(null);
    } catch {
      setMessagesByAgent(prev => {
        const next = new Map(prev);
        next.set(agentId, []);
        return next;
      });
    }
  }, []);

  const sendMessage = useCallback(async (text: string) => {
    if (!text.trim() || isStreaming) return;

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

    let segmentToolCalls: { tool: string; input: any; output: string }[] = [];
    let doneReceived = false;
    let terminalErrorReceived = false;

    let timeoutMs: number | undefined;
    if (chatTimeoutRef.current === null) {
      try {
        const cfg = await api.fetchChatTimeout();
        chatTimeoutRef.current = cfg.timeoutSeconds ?? 120;
      } catch {
        chatTimeoutRef.current = 120;
      }
    }
    if (chatTimeoutRef.current > 0) {
      timeoutMs = chatTimeoutRef.current * 1000;
    }

    try {
      await api.streamChat(text, sessionId!, currentAgentId, (event: SSEEvent) => {
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
            segmentToolCalls = [...segmentToolCalls, newTc];
            setMessages(prev => {
              const targetId = streamingAssistantIdRef.current || assistantMsgId;
              const idx = prev.findIndex(m => m.id === targetId);
              const last = idx >= 0 ? prev[idx] : null;
              if (idx >= 0 && last?.role === "assistant") {
                const updated = prev.slice();
                updated[idx] = { ...last, toolCalls: segmentToolCalls };
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
                // 按 tool 名称匹配未完成的调用，避免多个 tool_end 时错误覆盖
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
            if (segmentToolCalls.length > 0) {
              const segIdx = segmentToolCalls.findIndex(
                t => !t.output && (!toolName || t.tool === toolName)
              );
              const segFallback = segIdx >= 0 ? segIdx : segmentToolCalls.findIndex(t => !t.output);
              const segToUpdate = segFallback >= 0 ? segFallback : segmentToolCalls.length - 1;
              segmentToolCalls = segmentToolCalls.map((t, i) =>
                i === segToUpdate ? { ...t, output } : t
              );
            }
            break;
          }

          case "new_response":
            segmentToolCalls = [];
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

              // 若最后一条是用于占位的空 assistant 消息，则替换成 command。
              if (last?.role === "assistant" && !(last.content || "").trim()) {
                const updated = prev.slice();
                updated[idx] = commandMsg;
                return updated;
              }

              // 其余情况按事件逐条新增，避免多条 command_response 文案粘连。
              return [...prev, commandMsg];
            });
            break;
          }

          case "session_reset": {
            // /new or /reset: 重置本地状态，但保持当前 assistant 消息用于接收后续问候
            setLifecycleEvents([]);
            setLastUsage(null);
            // 如果 command_response 把空的 assistant 替换成了 command，需要新建 assistant 占位
            const newAssistantId = `assistant-${Date.now()}`;
            streamingAssistantIdRef.current = newAssistantId;
            setMessages(prev => {
              const last = prev[prev.length - 1];
              // 如果最后一条是 command，说明 assistant 被替换了，需要新建占位
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
            doneReceived = true;
            if (event.usage) setLastUsage(event.usage);
            if (event.context_utilization != null) setContextUtilization(event.context_utilization);
            setMessages(prev => {
              // 使用 streamingAssistantIdRef.current 获取最新的 assistant ID
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
            doneReceived = true;
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
            terminalErrorReceived = true;
            doneReceived = true;
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
      }, { signal: controller.signal, timeoutMs });
    } catch (e: any) {
      if (e.name !== "AbortError") {
        terminalErrorReceived = true;
        doneReceived = true;
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
        // 用户手动停止属于预期行为，不追加连接错误文案。
      }
    } finally {
      const stoppedByUser = userStoppedRef.current;
      if (streamingAssistantIdRef.current === assistantMsgId) {
        streamingAssistantIdRef.current = null;
      }
      setIsStreaming(false);
      abortRef.current = null;
      userStoppedRef.current = false;
      if (!doneReceived) {
        // 仅在非手动中断时回源重载，避免 stop 后突兀刷新。
        if (!stoppedByUser && !terminalErrorReceived) {
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
    }
  }, [currentAgentId, currentSessionId, isStreaming, addLifecycleEvent, setCurrentSessionId, loadMessages, options]);

  const stopStreaming = useCallback(async () => {
    userStoppedRef.current = true;
    const sessionId = currentSessionId;
    if (sessionId) {
      try {
        await api.abortChat(currentAgentId, sessionId, { userInitiated: true });
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

  // 当切换到新 Agent 时，重置 streaming 状态
  useEffect(() => {
    // 切换 Agent 时，确保当前 Agent 的 isStreaming 状态正确
    // 如果之前有过期的 streaming 状态，重置它
    const currentStreaming = isStreamingByAgent.get(currentAgentId);
    if (currentStreaming) {
      // 检查是否有正在运行的请求，如果没有则重置
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
