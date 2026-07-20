"use client";

import React, { createContext, useContext, useCallback, useMemo } from "react";
import { useAppWorkspace } from "./hooks/useAppWorkspace";
import { AgentProvider } from "./agentContext";
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
  // Config
  ragMode: boolean;
  setRagMode: (enabled: boolean) => Promise<void>;

  // Actions
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
    agent,
    inspector,
    chat,
    subagents,
    ragMode,
    setRagMode,
    skillsRefreshTrigger,
    triggerSkillsRefresh,
  } = workspace;

  const value = useMemo<AppState>(() => ({
    ragMode,
    setRagMode,

    skillsRefreshTrigger,
    triggerSkillsRefresh,
  }), [
    ragMode,
    setRagMode,
    skillsRefreshTrigger,
    triggerSkillsRefresh,
  ]);

  return (
    <AppContext.Provider value={value}>
      <InspectorProvider inspector={inspector}>
        <ChatProvider chat={chat}>
          <SubagentProvider subagents={subagents}>
            <AgentProvider agent={agent}>{children}</AgentProvider>
          </SubagentProvider>
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
