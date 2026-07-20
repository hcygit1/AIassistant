"use client";

import React, { createContext, useContext, useCallback, useMemo } from "react";
import { useAppWorkspace } from "./hooks/useAppWorkspace";
import { ApprovalProvider, useApproval } from "./approvalContext";
import { ChatProvider } from "./chatContext";
import { InspectorProvider } from "./inspectorContext";
import { SubagentProvider } from "./subagentContext";
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

  // Config
  ragMode: boolean;
  setRagMode: (enabled: boolean) => Promise<void>;

  // Actions
  loadAgents: () => Promise<void>;
  switchAgent: (agentId: string) => Promise<void>;
  loadMainSession: () => Promise<void>;
  skillsRefreshTrigger: number;
  triggerSkillsRefresh: () => void;
}

const AppContext = createContext<AppState | null>(null);

function AppStateProvider({ children }: { children: React.ReactNode }) {
  const { t } = useUi();
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
    inspector,
    loadAgents,
    loadMainSession,
    switchAgent,
    chat,
    subagents,
    ragMode,
    setRagMode,
    skillsRefreshTrigger,
    triggerSkillsRefresh,
  } = workspace;

  const value = useMemo<AppState>(() => ({
    agents,
    currentAgentId,
    currentSessionId,
    currentModel,

    ragMode,
    setRagMode,

    loadAgents,
    switchAgent,
    loadMainSession,
    skillsRefreshTrigger,
    triggerSkillsRefresh,
  }), [
    agents,
    currentAgentId,
    currentSessionId,
    currentModel,
    ragMode,
    setRagMode,
    loadAgents,
    switchAgent,
    loadMainSession,
    skillsRefreshTrigger,
    triggerSkillsRefresh,
  ]);

  return (
    <AppContext.Provider value={value}>
      <InspectorProvider inspector={inspector}>
        <ChatProvider chat={chat}>
          <SubagentProvider subagents={subagents}>{children}</SubagentProvider>
        </ChatProvider>
      </InspectorProvider>
    </AppContext.Provider>
  );
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
