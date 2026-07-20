"use client";

import { useEffect } from "react";

import * as api from "../api";
import type { SSEEvent } from "../api";

export interface PendingApproval {
  approval_id: string;
  tool: string;
  input_preview: string;
}

interface AgentEventHandlers {
  onSkillsUpdated: () => void;
  onHeartbeatMessage: () => void;
  onApprovalRequired: (approval: PendingApproval) => void;
}

function toPendingApproval(event: SSEEvent): PendingApproval | null {
  const approvalId = event.approval_id || "";
  if (!approvalId) return null;
  return {
    approval_id: approvalId,
    tool: event.tool || "exec",
    input_preview: event.input_preview || "",
  };
}

export function useAgentEvents(
  agentId: string,
  handlers: AgentEventHandlers,
): void {
  const {
    onSkillsUpdated,
    onHeartbeatMessage,
    onApprovalRequired,
  } = handlers;

  useEffect(() => {
    if (!agentId) return;
    return api.subscribeAgentEvents(agentId, (event) => {
      if (event.type === "lifecycle" && event.event === "skills_updated") {
        onSkillsUpdated();
      }
      if (event.type === "heartbeat_message") {
        onHeartbeatMessage();
      }
      if (
        event.type === "lifecycle"
        && event.event === "approval_required"
      ) {
        const approval = toPendingApproval(event);
        if (approval) onApprovalRequired(approval);
      }
    });
  }, [agentId, onApprovalRequired, onHeartbeatMessage, onSkillsUpdated]);
}
