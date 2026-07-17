"use client";

import { useCallback, useEffect, useState } from "react";

import * as api from "../api";

export type InspectorPanelMode = "docked" | "overlay" | "hidden";

export interface InspectorFile {
  path: string;
  content: string;
}

export function useInspectorState(currentAgentId: string) {
  const [inspectorWidth, setInspectorWidth] = useState(380);
  const [inspectorPanelMode, setInspectorPanelMode] = useState<InspectorPanelMode>("docked");
  const [inspectorTab, setInspectorTab] = useState<string>("files");
  const [inspectorFile, setInspectorFile] = useState<InspectorFile | null>(null);
  const [inspectorFileLoading, setInspectorFileLoading] = useState(false);

  useEffect(() => {
    try {
      const savedMode = window.localStorage.getItem("pipixia.inspector.mode");
      if (savedMode === "docked" || savedMode === "overlay" || savedMode === "hidden") {
        setInspectorPanelMode(savedMode);
      }
      const savedWidth = Number(window.localStorage.getItem("pipixia.inspector.width") || "");
      if (Number.isFinite(savedWidth) && savedWidth >= 280 && savedWidth <= 720) {
        setInspectorWidth(savedWidth);
      }
    } catch {
      // Ignore unavailable local storage.
    }
  }, []);

  useEffect(() => {
    try {
      window.localStorage.setItem("pipixia.inspector.mode", inspectorPanelMode);
    } catch {
      // Ignore unavailable local storage.
    }
  }, [inspectorPanelMode]);

  useEffect(() => {
    try {
      window.localStorage.setItem("pipixia.inspector.width", String(inspectorWidth));
    } catch {
      // Ignore unavailable local storage.
    }
  }, [inspectorWidth]);

  const openFile = useCallback(async (path: string) => {
    setInspectorFileLoading(true);
    setInspectorTab("files");
    try {
      setInspectorFile(await api.readFile(currentAgentId, path));
    } catch {
      setInspectorFile({ path, content: "（无法读取文件）" });
    } finally {
      setInspectorFileLoading(false);
    }
  }, [currentAgentId]);

  const saveInspectorFile = useCallback(async (content: string) => {
    if (!inspectorFile) return;
    await api.saveFile(currentAgentId, inspectorFile.path, content);
    setInspectorFile({ ...inspectorFile, content });
  }, [currentAgentId, inspectorFile]);

  const resetInspector = useCallback(() => {
    setInspectorFile(null);
    setInspectorFileLoading(false);
  }, []);

  return {
    inspectorWidth,
    setInspectorWidth,
    inspectorPanelMode,
    setInspectorPanelMode,
    inspectorTab,
    setInspectorTab,
    inspectorFile,
    inspectorFileLoading,
    openFile,
    saveInspectorFile,
    resetInspector,
  };
}
