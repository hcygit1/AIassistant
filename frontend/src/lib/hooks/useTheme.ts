"use client";

import { useState, useCallback, useEffect } from "react";

export type ThemeMode = "system" | "light" | "dark";
export type EffectiveTheme = "light" | "dark";

const STORAGE_KEY = "pipixia-theme";

export function useTheme() {
  const [theme, setThemeState] = useState<ThemeMode>("system");
  const [systemTheme, setSystemTheme] = useState<EffectiveTheme>("light");

  const computeEffective = useCallback((mode: ThemeMode): EffectiveTheme => {
    if (mode === "dark") return "dark";
    if (mode === "light") return "light";
    if (typeof window !== "undefined" && window.matchMedia?.("(prefers-color-scheme: dark)")?.matches) return "dark";
    return "light";
  }, []);

  const applyTheme = useCallback((mode: ThemeMode) => {
    if (typeof document !== "undefined") {
      document.documentElement.dataset.theme = mode;
    }
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const stored = window.localStorage.getItem(STORAGE_KEY);
    const initial: ThemeMode = stored === "light" || stored === "dark" || stored === "system"
      ? stored
      : "system";
    applyTheme(initial);
    const timer = window.setTimeout(() => {
      setThemeState(initial);
      setSystemTheme(computeEffective("system"));
    }, 0);
    return () => window.clearTimeout(timer);
  }, [applyTheme, computeEffective]);

  useEffect(() => {
    if (theme === "system" && typeof window !== "undefined") {
      const mql = window.matchMedia?.("(prefers-color-scheme: dark)");
      const handler = () => setSystemTheme(computeEffective("system"));
      mql?.addEventListener?.("change", handler);
      return () => mql?.removeEventListener?.("change", handler);
    }
  }, [theme, computeEffective]);

  const setTheme = useCallback((mode: ThemeMode) => {
    setThemeState(mode);
    if (typeof window !== "undefined") {
      window.localStorage.setItem(STORAGE_KEY, mode);
    }
    if (mode === "system") setSystemTheme(computeEffective("system"));
    applyTheme(mode);
  }, [applyTheme, computeEffective]);

  const effectiveTheme: EffectiveTheme = theme === "system" ? systemTheme : theme;

  return { theme, effectiveTheme, setTheme };
}
