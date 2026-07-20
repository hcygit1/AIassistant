"use client";

import React, { createContext, useContext } from "react";

import { useAppUiState } from "./hooks/useAppUiState";

export type UiState = ReturnType<typeof useAppUiState>;

const UiContext = createContext<UiState | null>(null);

export function UiProvider({ children }: { children: React.ReactNode }) {
  const value = useAppUiState();
  return <UiContext.Provider value={value}>{children}</UiContext.Provider>;
}

export function useUi(): UiState {
  const context = useContext(UiContext);
  if (!context) throw new Error("useUi must be used inside UiProvider");
  return context;
}
