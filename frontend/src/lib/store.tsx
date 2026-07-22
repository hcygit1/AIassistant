"use client";

import React, { useCallback } from "react";
import { useAppWorkspace } from "./hooks/useAppWorkspace";
import { AgentProvider } from "./agentContext";
import { ApprovalProvider, useApproval } from "./approvalContext";
import { ChatProvider } from "./chatContext";
import { InspectorProvider } from "./inspectorContext";
import { SkillsRefreshProvider } from "./skillsRefreshContext";
import { SubagentProvider } from "./subagentContext";
import { UiProvider, useUi } from "./uiContext";
import { formatCommandResponse as formatLocalizedCommandResponse } from "./commandResponses";

export type { ChatMessage, LifecycleEvent } from "./hooks/useChat";
export type { ThemeMode, EffectiveTheme } from "./hooks/useTheme";
export type { Locale, Messages } from "./i18n/locales";
export type { UiNotice } from "./hooks/useAppUiState";

function WorkspaceProviders({ children }: { children: React.ReactNode }) {
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
    skillsRefreshTrigger,
  } = workspace;

  return (
    <SkillsRefreshProvider version={skillsRefreshTrigger}>
      <InspectorProvider inspector={inspector}>
        <ChatProvider chat={chat}>
          <SubagentProvider subagents={subagents}>
            <AgentProvider agent={agent}>{children}</AgentProvider>
          </SubagentProvider>
        </ChatProvider>
      </InspectorProvider>
    </SkillsRefreshProvider>
  );
}

export function AppProvider({ children }: { children: React.ReactNode }) {
  return (
    <UiProvider>
      <ApprovalProvider>
        <WorkspaceProviders>{children}</WorkspaceProviders>
      </ApprovalProvider>
    </UiProvider>
  );
}
