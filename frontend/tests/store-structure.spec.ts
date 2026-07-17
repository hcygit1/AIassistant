import { expect, test } from "@playwright/test";
import { existsSync, readFileSync } from "node:fs";

test("keeps inspector persistence and file operations outside AppProvider", () => {
  const hookPath = "src/lib/hooks/useInspectorState.ts";
  expect(existsSync(hookPath)).toBe(true);
  if (!existsSync(hookPath)) return;

  const store = readFileSync("src/lib/store.tsx", "utf8");
  const inspector = readFileSync(hookPath, "utf8");

  expect(store).not.toContain("pipixia.inspector.");
  expect(store).not.toContain("api.readFile(");
  expect(store).not.toContain("api.saveFile(");
  expect(inspector).toContain("pipixia.inspector.mode");
  expect(inspector).toContain("api.readFile(");
  expect(inspector).toContain("api.saveFile(");
});

test("keeps locale, theme and modal state outside AppProvider", () => {
  const hookPath = "src/lib/hooks/useAppUiState.ts";
  expect(existsSync(hookPath)).toBe(true);
  if (!existsSync(hookPath)) return;

  const store = readFileSync("src/lib/store.tsx", "utf8");
  const uiState = readFileSync(hookPath, "utf8");

  expect(store).not.toContain("pipixia.locale");
  expect(store).not.toContain("useTheme(");
  expect(uiState).toContain("pipixia.locale");
  expect(uiState).toContain("useTheme(");
  expect(uiState).toContain("showConfigModal");
  expect(uiState).toContain("showMemoryModal");
});
