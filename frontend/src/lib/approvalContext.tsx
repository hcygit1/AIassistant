"use client";

import React, { createContext, useContext, useState } from "react";

export interface PendingApproval {
  approval_id: string;
  tool: string;
  input_preview: string;
}

interface ApprovalState {
  pendingApproval: PendingApproval | null;
  setPendingApproval: React.Dispatch<React.SetStateAction<PendingApproval | null>>;
}

const ApprovalContext = createContext<ApprovalState | null>(null);

export function ApprovalProvider({ children }: { children: React.ReactNode }) {
  const [pendingApproval, setPendingApproval] = useState<PendingApproval | null>(null);
  return (
    <ApprovalContext.Provider value={{ pendingApproval, setPendingApproval }}>
      {children}
    </ApprovalContext.Provider>
  );
}

export function useApproval(): ApprovalState {
  const context = useContext(ApprovalContext);
  if (!context) throw new Error("useApproval must be used inside ApprovalProvider");
  return context;
}
