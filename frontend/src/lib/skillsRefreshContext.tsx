"use client";

import React, { createContext, useContext } from "react";


const SkillsRefreshContext = createContext<number | null>(null);

export function SkillsRefreshProvider({
  version,
  children,
}: {
  version: number;
  children: React.ReactNode;
}) {
  return (
    <SkillsRefreshContext.Provider value={version}>
      {children}
    </SkillsRefreshContext.Provider>
  );
}

export function useSkillsRefresh(): number {
  const version = useContext(SkillsRefreshContext);
  if (version === null) {
    throw new Error(
      "useSkillsRefresh must be used inside SkillsRefreshProvider",
    );
  }
  return version;
}
