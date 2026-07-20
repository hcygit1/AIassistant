"use client";

import { useCallback, useState } from "react";

import * as api from "../api";
import type { PendingApproval } from "../approvalContext";
import { useAgentEvents } from "./useAgentEvents";
import { useAgentWorkspace } from "./useAgentWorkspace";
import { useInspectorState } from "./useInspectorState";

interface AppWorkspaceOptions {
  formatCommandResponse: (raw: string) => string;
  onApprovalRequired: (approval: PendingApproval) => void;
}

export function useAppWorkspace({
  formatCommandResponse,
  onApprovalRequired,
}: AppWorkspaceOptions) {
  const [currentAgentId, setCurrentAgentId] = useState("main");
  const [ragMode, setRagModeState] = useState(false);
  const [skillsRefreshTrigger, setSkillsRefreshTrigger] = useState(0);
  const inspector = useInspectorState(currentAgentId);

  const triggerSkillsRefresh = useCallback(
    () => setSkillsRefreshTrigger((value) => value + 1),
    [],
  );

  const workspace = useAgentWorkspace({
    currentAgentId,
    setCurrentAgentId,
    formatCommandResponse,
    onTurnComplete: triggerSkillsRefresh,
    resetInspector: inspector.resetInspector,
  });

  useAgentEvents(currentAgentId, {
    onSkillsUpdated: triggerSkillsRefresh,
    onHeartbeatMessage: workspace.loadMainSession,
    onApprovalRequired,
  });

  const setRagMode = useCallback(async (enabled: boolean) => {
    await api.updateRagMode(enabled);
    setRagModeState(enabled);
  }, []);

  return {
    ...workspace,
    ...inspector,
    ragMode,
    setRagMode,
    skillsRefreshTrigger,
    triggerSkillsRefresh,
  };
}
