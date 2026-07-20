"use client";

import { useEffect, useRef } from "react";

import type { LifecycleEvent } from "../chatState";
import type { UiNotice } from "./useAppUiState";

type ShowNotice = (notice: UiNotice) => void;

export function resolveLifecycleNotice(event: LifecycleEvent): UiNotice | null {
  const data = event.data || {};

  if (event.event === "session_memory_saved") {
    const path = data.path ? `（${data.path}）` : "";
    return { kind: "success", text: `长期记忆已后台保存${path}` };
  }
  if (event.event === "session_memory_failed") {
    const reason = data.reason ? `：${String(data.reason)}` : "";
    return { kind: "error", text: `长期记忆后台保存失败${reason}` };
  }
  return null;
}

export function useLifecycleNotices(
  events: LifecycleEvent[],
  showNotice: ShowNotice,
): void {
  const lastNoticeKeyRef = useRef("");

  useEffect(() => {
    const event = events[events.length - 1];
    if (!event) return;

    const data = event.data || {};
    const key = `${event.event}|${data.session_id || ""}|${data.path || ""}|${data.reason || ""}|${event.timestamp}`;
    if (lastNoticeKeyRef.current === key) return;
    lastNoticeKeyRef.current = key;

    const notice = resolveLifecycleNotice(event);
    if (notice) showNotice(notice);
  }, [events, showNotice]);
}
