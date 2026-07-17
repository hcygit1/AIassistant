"use client";

import { useEffect, useRef, useCallback, useMemo, useReducer } from "react";
import * as api from "../api";
import {
  createSubagentState,
  deriveSubagentViews,
  mapSubagentEvent,
  subagentStateReducer,
} from "../subagentState";

export function useSubagents(
  currentAgentId: string,
  currentSessionId: string | null,
  onSubagentDone?: () => void,
) {
  const scopeKey = `${currentAgentId}\u0000${currentSessionId || ""}`;
  const [state, dispatch] = useReducer(
    subagentStateReducer,
    scopeKey,
    createSubagentState,
  );
  const refreshTimerRef = useRef<number | null>(null);
  const doneTimerRef = useRef<number | null>(null);

  const refreshSubagents = useCallback(async () => {
    if (!currentSessionId) return;
    dispatch({ type: "loading", scopeKey });
    try {
      const resp = await api.fetchSubagents(currentAgentId, currentSessionId);
      const tree = resp.tree || [];
      const flat = resp.flat || [];
      dispatch({ type: "success", scopeKey, tree, flat });
    } catch {
      dispatch({ type: "failure", scopeKey });
    }
  }, [currentAgentId, currentSessionId, scopeKey]);

  useEffect(() => {
    if (!currentSessionId) return;
    const initial = window.setTimeout(() => void refreshSubagents(), 0);
    const id = setInterval(refreshSubagents, 2500);
    return () => {
      window.clearTimeout(initial);
      clearInterval(id);
    };
  }, [currentSessionId, refreshSubagents]);

  useEffect(() => {
    const triggerRefresh = () => {
      if (refreshTimerRef.current) window.clearTimeout(refreshTimerRef.current);
      refreshTimerRef.current = window.setTimeout(refreshSubagents, 120);
    };

    const unsubscribe = api.subscribeAgentEvents(
      currentAgentId,
      (event) => {
        const mapped = mapSubagentEvent(event, Date.now());
        if (!mapped) return;
        dispatch({
          type: "trace",
          scopeKey,
          runId: mapped.runId,
          trace: mapped.trace,
        });
        if (mapped.shouldRefresh) triggerRefresh();
        if (mapped.doneDelayMs !== null) {
          if (doneTimerRef.current) window.clearTimeout(doneTimerRef.current);
          doneTimerRef.current = window.setTimeout(() => {
            onSubagentDone?.();
          }, mapped.doneDelayMs);
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
  }, [currentAgentId, onSubagentDone, refreshSubagents, scopeKey]);

  const visibleState = state.scopeKey === scopeKey
    ? state
    : { ...createSubagentState(scopeKey), loading: Boolean(currentSessionId) };
  const { runningSubagents } = useMemo(
    () => deriveSubagentViews(visibleState.flat),
    [visibleState.flat],
  );

  return {
    tree: visibleState.tree,
    flat: visibleState.flat,
    runningSubagents,
    traceMap: visibleState.traceMap,
    loading: visibleState.loading,
    refreshSubagents,
  };
}
