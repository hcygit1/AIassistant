"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import * as api from "../api";

interface SubagentInfo {
  run_id: string;
  task: string;
  status: string;
}

export function useSubagents(
  currentAgentId: string,
  currentSessionId: string | null,
  onSubagentDone?: () => void,
) {
  const [runningSubagents, setRunningSubagents] = useState<SubagentInfo[]>([]);
  const refreshTimerRef = useRef<number | null>(null);
  const doneTimerRef = useRef<number | null>(null);

  const refreshSubagents = useCallback(async () => {
    if (!currentSessionId) return;
    try {
      const resp = await api.fetchSubagents(currentAgentId, currentSessionId);
      const data = resp.flat || [];
      const running = data
        .filter((s: any) => (s.state || s.status) === "running")
        .map((s: any) => ({
          run_id: s.run_id,
          task: s.task?.slice(0, 60) || "",
          status: s.state || s.status,
        }));
      setRunningSubagents(running);
    } catch {
      setRunningSubagents([]);
    }
  }, [currentAgentId, currentSessionId]);

  useEffect(() => {
    if (!currentSessionId) return;
    refreshSubagents();
    const id = setInterval(refreshSubagents, 2500);
    return () => clearInterval(id);
  }, [currentSessionId, refreshSubagents]);

  useEffect(() => {
    const triggerRefresh = () => {
      if (refreshTimerRef.current) window.clearTimeout(refreshTimerRef.current);
      refreshTimerRef.current = window.setTimeout(refreshSubagents, 120);
    };

    const unsubscribe = api.subscribeAgentEvents(
      currentAgentId,
      (event) => {
        const type = event.type || "";
        if (!type.startsWith("subagent_")) return;
        const announceState = String((event as any).announce_state || "").trim();
        // #region agent log
        fetch('http://127.0.0.1:7700/ingest/77c66232-605d-4df3-930c-89bbb8ebd5c2',{method:'POST',headers:{'Content-Type':'application/json','X-Debug-Session-Id':'e404f0'},body:JSON.stringify({sessionId:'e404f0',runId:'pre-fix',hypothesisId:'H2C',location:'frontend/src/lib/hooks/useSubagents.ts:eventHandler',message:'subagent_event',data:{type,runId:(event as any).run_id || ''},timestamp:Date.now()})}).catch(()=>{});
        // #endregion
        triggerRefresh();
        if (type === "subagent_announce" && (announceState === "delivered" || announceState === "dropped")) {
          // #region agent log
          fetch('http://127.0.0.1:7700/ingest/77c66232-605d-4df3-930c-89bbb8ebd5c2',{method:'POST',headers:{'Content-Type':'application/json','X-Debug-Session-Id':'e404f0'},body:JSON.stringify({sessionId:'e404f0',runId:'pre-fix',hypothesisId:'H2D',location:'frontend/src/lib/hooks/useSubagents.ts:eventHandler',message:'onSubagentDone_call',data:{type,announceState,runId:(event as any).run_id || ''},timestamp:Date.now()})}).catch(()=>{});
          // #endregion
          if (doneTimerRef.current) window.clearTimeout(doneTimerRef.current);
          doneTimerRef.current = window.setTimeout(() => {
            onSubagentDone?.();
          }, announceState === "delivered" ? 250 : 100);
        }
      },
      () => { },
    );

    return () => {
      unsubscribe();
      if (refreshTimerRef.current) {
        window.clearTimeout(refreshTimerRef.current);
        refreshTimerRef.current = null;
      }
      if (doneTimerRef.current) {
        window.clearTimeout(doneTimerRef.current);
        doneTimerRef.current = null;
      }
    };
  }, [currentAgentId, currentSessionId, onSubagentDone, refreshSubagents]);

  return { runningSubagents };
}
