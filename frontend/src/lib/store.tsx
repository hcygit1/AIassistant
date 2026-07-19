"use client";

import React, { createContext, useContext, useState, useCallback, useEffect, useMemo, useRef } from "react";
import * as api from "./api";
import type { TokenUsage } from "./api";
import { useChat } from "./hooks/useChat";
import { useSubagents } from "./hooks/useSubagents";
import { useInspectorState } from "./hooks/useInspectorState";
import { useAppUiState } from "./hooks/useAppUiState";
import type { UiNotice } from "./hooks/useAppUiState";
import { formatCommandResponse as formatLocalizedCommandResponse } from "./commandResponses";
import type { Locale, Messages } from "./i18n/locales";

export type { ChatMessage, LifecycleEvent } from "./hooks/useChat";
export type { ThemeMode, EffectiveTheme } from "./hooks/useTheme";
export type { Locale, Messages } from "./i18n/locales";
export type { UiNotice } from "./hooks/useAppUiState";

interface AppState {
  // Agent
  agents: any[];
  currentAgentId: string;
  currentSessionId: string | null;
  currentModel: any | null;

  // Chat (from useChat)
  messages: ReturnType<typeof useChat>["messages"];
  isStreaming: boolean;
  lifecycleEvents: ReturnType<typeof useChat>["lifecycleEvents"];
  lastUsage: TokenUsage | null;
  contextUtilization: number | null;
  sessionError: string | null;
  sendMessage: (text: string) => Promise<void>;
  stopStreaming: () => void;

  // Config
  ragMode: boolean;
  setRagMode: (enabled: boolean) => Promise<void>;

  // Inspector
  inspectorWidth: number;
  setInspectorWidth: (w: number) => void;
  isCompactLayout: boolean;
  inspectorPanelMode: "docked" | "overlay" | "hidden";
  setInspectorPanelMode: (mode: "docked" | "overlay" | "hidden") => void;
  inspectorTab: string;
  setInspectorTab: (tab: any) => void;
  inspectorFile: { path: string; content: string } | null;
  inspectorFileLoading: boolean;
  openFile: (path: string) => Promise<void>;
  saveInspectorFile: (content: string) => Promise<void>;

  // Subagents
  subagentTree: ReturnType<typeof useSubagents>["tree"];
  subagents: ReturnType<typeof useSubagents>["flat"];
  runningSubagents: ReturnType<typeof useSubagents>["runningSubagents"];
  subagentTraceMap: ReturnType<typeof useSubagents>["traceMap"];
  subagentsLoading: boolean;
  refreshSubagents: () => Promise<void>;

  // UI
  showConfigModal: boolean;
  setShowConfigModal: (v: boolean) => void;
  showMemoryModal: boolean;
  setShowMemoryModal: (v: boolean) => void;
  theme: "system" | "light" | "dark";
  effectiveTheme: "light" | "dark";
  setTheme: (mode: "system" | "light" | "dark") => void;
  uiNotice: UiNotice | null;
  showNotice: (notice: UiNotice) => void;
  clearNotice: () => void;

  // Exec approval
  pendingApproval: { approval_id: string; tool: string; input_preview: string } | null;
  setPendingApproval: (v: { approval_id: string; tool: string; input_preview: string } | null) => void;

  // i18n
  locale: Locale;
  setLocale: (locale: Locale) => void;
  t: Messages;

  // Actions
  loadAgents: () => Promise<void>;
  switchAgent: (agentId: string) => Promise<void>;
  loadMainSession: () => Promise<void>;
  skillsRefreshTrigger: number;
  triggerSkillsRefresh: () => void;
}

const AppContext = createContext<AppState | null>(null);

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [agents, setAgents] = useState<any[]>([]);
  const [currentAgentId, setCurrentAgentId] = useState("main");
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [currentModel, setCurrentModel] = useState<any | null>(null);
  const [ragMode, setRagModeState] = useState(false);
  const [skillsRefreshTrigger, setSkillsRefreshTrigger] = useState(0);
  const [pendingApproval, setPendingApproval] = useState<{ approval_id: string; tool: string; input_preview: string } | null>(null);
  const lastLifecycleNoticeKeyRef = useRef("");
  const loadMainSessionReqRef = useRef(0);
  const currentAgentIdRef = useRef("main");

  const {
    locale,
    setLocale,
    t,
    theme,
    effectiveTheme,
    setTheme,
    showConfigModal,
    setShowConfigModal,
    showMemoryModal,
    setShowMemoryModal,
    uiNotice,
    showNotice,
    clearNotice,
  } = useAppUiState();
  const {
    inspectorWidth,
    setInspectorWidth,
    isCompactLayout,
    inspectorPanelMode,
    setInspectorPanelMode,
    inspectorTab,
    setInspectorTab,
    inspectorFile,
    inspectorFileLoading,
    openFile,
    saveInspectorFile,
    resetInspector,
  } = useInspectorState(currentAgentId);

  // 持久化 currentAgentId（Hydration 安全：初始 "main"，useEffect 恢复）
  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const saved = window.localStorage.getItem("pipixia.agent");
      if (saved && typeof saved === "string" && saved.trim()) {
        const agentId = saved.trim();
        currentAgentIdRef.current = agentId;
        setCurrentAgentId(agentId);
      }
    } catch {
      // ignore
    }
  }, []);

  const triggerSkillsRefresh = useCallback(() => setSkillsRefreshTrigger((n) => n + 1), []);
  const formatCommandResponse = useCallback(
    (raw: string) => formatLocalizedCommandResponse(raw, t),
    [t],
  );

  const loadAgents = useCallback(async () => {
    const data = await api.fetchAgents();
    setAgents(data);
  }, []);

  const loadMainSession = useCallback(async () => {
    const reqId = ++loadMainSessionReqRef.current;
    try {
      const session = await api.fetchMainSession(currentAgentId);
      if (reqId !== loadMainSessionReqRef.current) {
        return;
      }
      setCurrentSessionId(session.session_id);
      await chat.loadMessages(currentAgentId, session.session_id);
      if (reqId !== loadMainSessionReqRef.current) {
        return;
      }
      const model = await api.fetchCurrentModel(currentAgentId);
      if (reqId !== loadMainSessionReqRef.current) {
        return;
      }
      setCurrentModel(model);
    } catch {
      if (reqId !== loadMainSessionReqRef.current) {
        return;
      }
      chat.setMessages([]);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentAgentId]);

  const handleSessionCompacted = useCallback((agentId: string) => {
    if (currentAgentIdRef.current === agentId) {
      void loadMainSession();
    }
  }, [loadMainSession]);

  const chatOptions = useMemo(() => ({
    onAgentCreated: loadAgents,
    onSessionCompacted: handleSessionCompacted,
    onTurnComplete: triggerSkillsRefresh,
    formatCommandResponse,
  }), [formatCommandResponse, handleSessionCompacted, loadAgents, triggerSkillsRefresh]);

  const chat = useChat(
    currentAgentId,
    currentSessionId,
    setCurrentSessionId,
    chatOptions,
  );

  const {
    tree: subagentTree,
    flat: subagents,
    runningSubagents,
    traceMap: subagentTraceMap,
    loading: subagentsLoading,
    refreshSubagents,
  } = useSubagents(
    currentAgentId,
    currentSessionId,
    loadMainSession,
  );

  // 订阅 Agent 事件：技能热加载、危险工具执行提示、exec 确认、心跳/定时消息
  useEffect(() => {
    if (!currentAgentId) return;
    const unsub = api.subscribeAgentEvents(currentAgentId, (event) => {
      if (event.type === "lifecycle" && event.event === "skills_updated") {
        triggerSkillsRefresh();
      }
      if (event.type === "heartbeat_message") {
        loadMainSession();
      }
      if (event.type === "lifecycle" && event.event === "approval_required") {
        const ev = event as any;
        const aid = ev.approval_id || "";
        const tool = ev.tool || "exec";
        const preview = ev.input_preview || "";
        if (aid) setPendingApproval({ approval_id: aid, tool, input_preview: preview });
      }
    });
    return unsub;
  }, [currentAgentId, triggerSkillsRefresh, loadMainSession]);

  const switchAgent = useCallback(async (agentId: string) => {
    // 保存当前 Agent 的状态（不要清空，只是切换）
    currentAgentIdRef.current = agentId;
    const reqId = ++loadMainSessionReqRef.current;
    setCurrentAgentId(agentId);
    try { window.localStorage.setItem("pipixia.agent", agentId); } catch {}
    setCurrentSessionId(null);
    resetInspector();

    // 加载新 Agent 的会话和消息
    try {
      const session = await api.fetchMainSession(agentId);
      if (reqId !== loadMainSessionReqRef.current || currentAgentIdRef.current !== agentId) return;
      setCurrentSessionId(session.session_id);
      // 加载消息时会自动恢复该 Agent 的 isStreaming 状态
      await chat.loadMessages(agentId, session.session_id);
      if (reqId !== loadMainSessionReqRef.current || currentAgentIdRef.current !== agentId) return;
      const model = await api.fetchCurrentModel(agentId);
      if (reqId !== loadMainSessionReqRef.current || currentAgentIdRef.current !== agentId) return;
      setCurrentModel(model);
    } catch {
      // 如果加载失败，清空该 Agent 的消息
      if (reqId === loadMainSessionReqRef.current && currentAgentIdRef.current === agentId) {
        chat.clearAgent(agentId);
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chat]);

  const setRagMode = useCallback(async (enabled: boolean) => {
    await api.updateRagMode(enabled);
    setRagModeState(enabled);
  }, []);

  useEffect(() => {
    if (!chat.lifecycleEvents.length) return;
    const last = chat.lifecycleEvents[chat.lifecycleEvents.length - 1];
    if (!last) return;
    const data = (last.data || {}) as any;
    const key = `${last.event}|${data.session_id || ""}|${data.path || ""}|${data.reason || ""}|${last.timestamp}`;
    if (lastLifecycleNoticeKeyRef.current === key) return;
    lastLifecycleNoticeKeyRef.current = key;

    if (last.event === "session_memory_saved") {
      const path = data.path ? `（${data.path}）` : "";
      showNotice({ kind: "success", text: `长期记忆已后台保存${path}` });
      return;
    }
    if (last.event === "session_memory_failed") {
      const reason = data.reason ? `：${String(data.reason)}` : "";
      showNotice({ kind: "error", text: `长期记忆后台保存失败${reason}` });
    }
  }, [chat.lifecycleEvents, showNotice]);

  const value: AppState = {
    agents,
    currentAgentId,
    currentSessionId,
    currentModel,

    messages: chat.messages,
    isStreaming: chat.isStreaming,
    lifecycleEvents: chat.lifecycleEvents,
    lastUsage: chat.lastUsage,
    contextUtilization: chat.contextUtilization,
    sessionError: chat.sessionError,
    sendMessage: chat.sendMessage,
    stopStreaming: chat.stopStreaming,

    ragMode,
    setRagMode,

    inspectorWidth,
    setInspectorWidth,
    isCompactLayout,
    inspectorPanelMode,
    setInspectorPanelMode,
    inspectorTab,
    setInspectorTab,
    inspectorFile,
    inspectorFileLoading,
    openFile,
    saveInspectorFile,

    subagentTree,
    subagents,
    runningSubagents,
    subagentTraceMap,
    subagentsLoading,
    refreshSubagents,

    showConfigModal,
    setShowConfigModal,
    showMemoryModal,
    setShowMemoryModal,
    theme,
    effectiveTheme,
    setTheme,
    uiNotice,
    showNotice,
    clearNotice,

    pendingApproval,
    setPendingApproval,

    locale,
    setLocale,
    t,

    loadAgents,
    switchAgent,
    loadMainSession,
    skillsRefreshTrigger,
    triggerSkillsRefresh,
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp(): AppState {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useApp must be used inside AppProvider");
  return ctx;
}
