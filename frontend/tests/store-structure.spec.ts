import { expect, test } from "@playwright/test";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

function listTsxFiles(root: string): string[] {
  return readdirSync(root, { withFileTypes: true }).flatMap((entry) => {
    const path = join(root, entry.name);
    if (entry.isDirectory()) return listTsxFiles(path);
    return entry.isFile() && path.endsWith(".tsx") ? [path] : [];
  });
}

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

test("exposes UI state through a dedicated context", () => {
  const contextPath = "src/lib/uiContext.tsx";
  expect(existsSync(contextPath)).toBe(true);
  if (!existsSync(contextPath)) return;

  const context = readFileSync(contextPath, "utf8");
  const store = readFileSync("src/lib/store.tsx", "utf8");

  expect(context).toContain("UiProvider");
  expect(context).toContain("useUi");
  expect(store).not.toContain("useAppUiState()");
});

test("keeps agent event subscription outside AppProvider", () => {
  const hookPath = "src/lib/hooks/useAgentEvents.ts";
  expect(existsSync(hookPath)).toBe(true);
  if (!existsSync(hookPath)) return;

  const store = readFileSync("src/lib/store.tsx", "utf8");
  const hook = readFileSync(hookPath, "utf8");

  expect(store).not.toContain("subscribeAgentEvents");
  expect(store).not.toContain("approval_required");
  expect(hook).toContain("subscribeAgentEvents");
  expect(hook).toContain("approval_required");
});

test("keeps approval state outside AppStateProvider", () => {
  const contextPath = "src/lib/approvalContext.tsx";
  expect(existsSync(contextPath)).toBe(true);
  if (!existsSync(contextPath)) return;

  const context = readFileSync(contextPath, "utf8");
  const store = readFileSync("src/lib/store.tsx", "utf8");
  const modal = readFileSync("src/components/layout/ApprovalModal.tsx", "utf8");

  expect(context).toContain("ApprovalProvider");
  expect(context).toContain("useApproval");
  expect(store).not.toContain("const [pendingApproval");
  expect(store).not.toContain("pendingApproval,");
  expect(modal).toContain("useApproval");
  expect(modal).not.toContain("pendingApproval } = useApp()");
});

test("keeps UI-only fields out of useApp consumers", () => {
  const uiFields = new Set([
    "showConfigModal",
    "setShowConfigModal",
    "showMemoryModal",
    "setShowMemoryModal",
    "theme",
    "effectiveTheme",
    "setTheme",
    "uiNotice",
    "showNotice",
    "clearNotice",
    "locale",
    "setLocale",
    "t",
  ]);
  const files = [
    ...listTsxFiles("src/app"),
    ...listTsxFiles("src/components"),
  ];

  for (const file of files) {
    const source = readFileSync(file, "utf8");
    const appSelections = source.matchAll(
      /const\s*\{((?:(?!useUi\(\);)[\s\S])*?)\}\s*=\s*useApp\(\);/g,
    );
    for (const selection of appSelections) {
      const selectedNames = selection[1]
        .split(",")
        .map((name) => name.trim().split(":")[0].trim())
        .filter(Boolean);
      expect(
        selectedNames.filter((name) => uiFields.has(name)),
        `${file} should read UI-only fields from useUi()`,
      ).toEqual([]);
    }
  }
});

test("keeps toast notifications clear of the workspace rail", () => {
  const styles = readFileSync("src/app/globals.css", "utf8");

  expect(styles).toContain("left: calc(4rem + 24px);");
  expect(styles).toContain("width: min(380px, calc(100vw - 112px));");
  expect(styles).toContain("min-width: 0;");
});

test("uses the config drawer as the incomplete initialization notice", () => {
  const page = readFileSync("src/app/page.tsx", "utf8");

  expect(page).not.toContain("检测到尚未完成初始化");
  expect(page).toContain("setShowConfigModal(true)");
});
