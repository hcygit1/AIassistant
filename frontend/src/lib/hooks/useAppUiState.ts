"use client";

import { useCallback, useEffect, useState } from "react";

import { getMessages } from "../i18n/locales";
import type { Locale } from "../i18n/locales";
import { useTheme } from "./useTheme";

export interface UiNotice {
  kind: "success" | "error" | "info";
  text: string;
}

export function useAppUiState() {
  const [locale, setLocaleState] = useState<Locale>("zh-CN");
  const [showConfigModal, setShowConfigModal] = useState(false);
  const [showMemoryModal, setShowMemoryModal] = useState(false);
  const [uiNotice, setUiNotice] = useState<UiNotice | null>(null);
  const { theme, effectiveTheme, setTheme } = useTheme();
  const t = getMessages(locale);

  useEffect(() => {
    let timer: number | undefined;
    try {
      const saved = window.localStorage.getItem("pipixia.locale");
      if (saved === "zh-CN" || saved === "en-US") {
        timer = window.setTimeout(() => setLocaleState(saved), 0);
      }
    } catch {
      // Ignore unavailable local storage.
    }
    return () => {
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, []);

  const setLocale = useCallback((nextLocale: Locale) => {
    setLocaleState(nextLocale);
    try {
      window.localStorage.setItem("pipixia.locale", nextLocale);
    } catch {
      // Ignore unavailable local storage.
    }
  }, []);

  const showNotice = useCallback((notice: UiNotice) => setUiNotice(notice), []);
  const clearNotice = useCallback(() => setUiNotice(null), []);

  return {
    locale,
    setLocale,
    t,
    theme,
    effectiveTheme,
    setTheme,
    showConfigModal,
    setShowConfigModal,
    showMemoryModal,
    setShowMemoryModal,
    uiNotice,
    showNotice,
    clearNotice,
  };
}
