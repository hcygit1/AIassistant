"use client";

import React, { createContext, useContext, useCallback, useEffect, useRef } from "react";
import type { TokenUsage } from "./api";
import { useChat } from "./hooks/useChat";
import { useAppWorkspace } from "./hooks/useAppWorkspace";
import { useSubagents } from "./hooks/useSubagents";
import { ApprovalProvider, useApproval } from "./approvalContext";
import { UiProvider, useUi } from "./uiContext";
import { formatCommandResponse as formatLocalizedCommandResponse } from "./commandResponses";

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

  // Actions
  loadAgents: () => Promise<void>;
  switchAgent: (agentId: string) => Promise<void>;
  loadMainSession: () => Promise<void>;
  skillsRefreshTrigger: number;
  triggerSkillsRefresh: () => void;
}

const AppContext = createContext<AppState | null>(null);

function AppStateProvider({ children }: { children: React.ReactNode }) {
  const lastLifecycleNoticeKeyRef = useRef("");

  const { t, showNotice } = useUi();
  const { setPendingApproval } = useApproval();
  const formatCommandResponse = useCallback(
    (raw: string) => formatLocalizedCommandResponse(raw, t),
    [t],
  );
  const workspace = useAppWorkspace({
    formatCommandResponse,
    onApprovalRequired: setPendingApproval,
  });

  const {
    agents,
    currentAgentId,
    currentSessionId,
    currentModel,
    loadAgents,
    loadMainSession,
    switchAgent,
    chat,
    tree: subagentTree,
    flat: subagents,
    runningSubagents,
    traceMap: subagentTraceMap,
    loading: subagentsLoading,
    refreshSubagents,
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
    ragMode,
    setRagMode,
    skillsRefreshTrigger,
    triggerSkillsRefresh,
  } = workspace;

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

    loadAgents,
    switchAgent,
    loadMainSession,
    skillsRefreshTrigger,
    triggerSkillsRefresh,
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function AppProvider({ children }: { children: React.ReactNode }) {
  return (
    <UiProvider>
      <ApprovalProvider>
        <AppStateProvider>{children}</AppStateProvider>
      </ApprovalProvider>
    </UiProvider>
  );
}

export function useApp(): AppState {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useApp must be used inside AppProvider");
  return ctx;
}
