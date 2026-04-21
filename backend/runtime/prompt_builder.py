"""System Prompt 构建器

静态指令全部在 AGENTS.md 中维护，本模块只负责：
  1. 按序加载静态文件（AGENTS.md → IDENTITY.md → USER.md）—— KV cache 前缀可复用
  2. 拼接动态 section（工具列表、技能快照、心跳配置、时间、工作区、运行时）
  3. 生成 PromptReport 供调试
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from config import (
    get_heartbeat_config,
    resolve_agent_config,
    resolve_agent_dir,
    resolve_agent_workspace,
)
from runtime.context_budget import resolve_budget
from runtime.source_sink_guard import build_trust_boundary_policy


# ---------------------------------------------------------------------------
# Prompt Report
# ---------------------------------------------------------------------------

@dataclass
class PromptFileEntry:
    label: str
    chars: int
    truncated: bool


@dataclass
class PromptReport:
    mode: str
    total_chars: int
    sections: list[str]
    injected_files: list[PromptFileEntry]
    tool_count: int
    tool_names: list[str]
    truncation_events: int

    def summary(self) -> str:
        lines = [
            f"[PromptReport] mode={self.mode} total_chars={self.total_chars} "
            f"sections={len(self.sections)} tools={self.tool_count} "
            f"files={len(self.injected_files)} truncations={self.truncation_events}",
        ]
        for f in self.injected_files:
            tag = " [TRUNCATED]" if f.truncated else ""
            lines.append(f"  file: {f.label} ({f.chars} chars){tag}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt Params
# ---------------------------------------------------------------------------

@dataclass
class PromptParams:
    agent_id: str
    mode: Literal["full", "minimal", "none"] = "full"
    available_tools: list[str] | None = None
    extra_system_prompt: str | None = None
    default_think_level: str = "off"
    locale: str = "zh-CN"
    heartbeat_prompt: str | None = None


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------

class PromptBuilder:

    MINIMAL_AGENTS_SECTIONS = [
        "工具调用风格", "安全", "技能", "大型工具输出",
    ]

    def build_system_prompt(
        self,
        agent_id: str,
        mode: Literal["full", "minimal", "none"] = "full",
        available_tools: list[str] | None = None,
        *,
        params: PromptParams | None = None,
    ) -> str:
        """Build system prompt. Accepts either positional args (legacy) or PromptParams."""
        if params is None:
            params = PromptParams(
                agent_id=agent_id,
                mode=mode,
                available_tools=available_tools,
            )

        prompt, _ = self._build_with_report(params)
        return prompt

    def build_system_prompt_with_report(
        self,
        params: PromptParams,
    ) -> tuple[str, PromptReport]:
        return self._build_with_report(params)

    def _build_with_report(self, params: PromptParams) -> tuple[str, PromptReport]:
        agent_id = params.agent_id
        mode = params.mode

        if mode == "none":
            prompt = "你是一个运行在 ClawChain 中的个人助手。"
            report = PromptReport(
                mode="none",
                total_chars=len(prompt),
                sections=["identity"],
                injected_files=[],
                tool_count=0,
                tool_names=[],
                truncation_events=0,
            )
            return prompt, report

        collected_sections: list[str] = []
        section_names: list[str] = []

        def _add(name: str, content: str) -> None:
            if content:
                collected_sections.append(content)
                section_names.append(name)

        # ── 静态文件（KV cache 前缀可复用）──
        context_text, file_entries, truncation_events = self._build_project_context_with_report(
            agent_id,
            mode=mode,
        )
        if context_text:
            _add("project_context", context_text)

        _add("trust_boundary", build_trust_boundary_policy())

        # ── 动态 section ──
        if mode == "full":
            _add("tooling", self._build_tooling(params.available_tools))
            _add("skills", self._build_skills(agent_id))
            _add("heartbeat_config", self._build_heartbeat_config(params))
            _add("time", self._build_time(agent_id))
            _add("workspace", self._build_workspace(agent_id))
            _add("runtime", self._build_runtime(agent_id))
        elif mode == "minimal":
            _add("tooling", self._build_tooling(params.available_tools))
            _add("skills", self._build_skills(agent_id))
            _add("time", self._build_time(agent_id))
            _add("workspace", self._build_workspace(agent_id))
            _add("runtime", self._build_runtime(agent_id))

        if params.extra_system_prompt:
            header = "## 子 Agent 上下文" if mode == "minimal" else "## 额外上下文"
            _add("extra_context", f"{header}\n{params.extra_system_prompt}")

        prompt = "\n\n".join(collected_sections)

        report = PromptReport(
            mode=mode,
            total_chars=len(prompt),
            sections=section_names,
            injected_files=file_entries,
            tool_count=len(params.available_tools) if params.available_tools else 0,
            tool_names=list(params.available_tools) if params.available_tools else [],
            truncation_events=truncation_events,
        )

        return prompt, report

    # ------------------------------------------------------------------
    # 结构化会话摘要 → Agent 可读文本
    # ------------------------------------------------------------------

    @staticmethod
    def format_session_summary(summary) -> str:
        """将 SessionSummary（或 dict）渲染为 Agent 可读的注入文本。"""
        import json as _json

        def _field(obj, key: str, fallback: str = "") -> str:
            val = getattr(obj, key, None) if not isinstance(obj, dict) else obj.get(key)
            return val or fallback

        def _list_field(obj, key: str) -> list:
            val = _field(obj, key, "[]")
            if isinstance(val, list):
                return val
            try:
                parsed = _json.loads(val)
                return parsed if isinstance(parsed, list) else []
            except Exception:
                return [val] if val and val != "[]" else []

        goal = _field(obj=summary, key="goal")
        progress = _field(obj=summary, key="progress")
        decisions = _list_field(obj=summary, key="decisions")
        open_items = _list_field(obj=summary, key="open_items")
        entities = _list_field(obj=summary, key="entities")
        user_prefs = _list_field(obj=summary, key="user_preferences")
        raw = _field(obj=summary, key="raw_summary")

        parts: list[str] = ["[会话摘要 — 压缩上下文]"]

        if goal:
            parts.append(f"\n🎯 会话目标：{goal}")
        if progress:
            parts.append(f"\n📋 当前进展：{progress}")
        if decisions:
            parts.append("\n💡 关键决策：")
            for d in decisions:
                parts.append(f"  - {d}")
        if open_items:
            parts.append("\n📌 待办事项：")
            for item in open_items:
                parts.append(f"  - {item}")
        if entities:
            parts.append(f"\n🏷️ 关键实体：{', '.join(entities)}")
        if user_prefs:
            parts.append(f"\n👤 用户偏好：{', '.join(user_prefs)}")

        if len(parts) == 1 and raw:
            parts.append(f"\n{raw}")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_section(markdown: str, heading: str) -> str:
        lines = markdown.splitlines()
        in_section = False
        result = []
        for line in lines:
            if line.startswith("## ") and heading in line:
                in_section = True
                result.append(line)
                continue
            if in_section:
                if line.startswith("## "):
                    break
                result.append(line)
        return "\n".join(result).strip()

    # ------------------------------------------------------------------
    # Dynamic section builders (仅运行时数据，静态指令在 AGENTS.md)
    # ------------------------------------------------------------------

    TOOL_DOCS: dict[str, str] = {
        "read": "读取文件内容（支持行号范围）",
        "write": "创建或覆盖文件",
        "edit": "精确编辑文件（查找替换）",
        "apply_patch": "应用多文件补丁",
        "grep": "搜索文件内容（正则表达式）",
        "find": "按模式查找文件",
        "ls": "列出目录内容",
        "exec": "执行 Shell 命令（沙箱环境）",
        "python_repl": "执行 Python 代码",
        "process_list": "列出活跃进程",
        "process_kill": "终止指定进程",
        "web_search": "搜索网络",
        "web_fetch": "获取并提取网页内容",
        "agents_list": "列出可用的 Agent ID",
        "sessions_list": "列出会话（含子 Agent）",
        "sessions_history": "获取其他会话的历史记录",
        "sessions_send": "向其他会话/子 Agent 发送消息",
        "sessions_spawn": "生成独立的子 Agent",
        "subagents": "管理子 Agent（list/kill/steer）",
        "session_status": "显示会话状态卡片",
        "memory_search": "语义 + 关键词混合搜索记忆（FTS5 + ANN）",
        "memory_get": "按 chunk_id 读取完整记忆内容",
        "search_knowledge_base": "搜索知识库文档",
        "cron": "管理定时任务与提醒（list/add/update/remove/run/wake）",
    }

    TOOL_CATEGORIES: dict[str, str] = {
        "read": "core", "write": "core", "edit": "core", "apply_patch": "core",
        "grep": "core", "find": "core", "ls": "core", "exec": "core",
        "python_repl": "core", "process_list": "core", "process_kill": "core",
        "web_search": "external", "web_fetch": "external",
        "agents_list": "core", "sessions_list": "core", "sessions_history": "core",
        "sessions_send": "core", "sessions_spawn": "core", "subagents": "core",
        "session_status": "core", "memory_search": "core", "memory_get": "core",
        "search_knowledge_base": "core",
        "cron": "core",
    }

    @classmethod
    def _build_tooling(cls, available_tools: list[str] | None = None) -> str:
        lines = [
            "## 可用工具",
            "",
            "工具名称区分大小写，请严格按照以下名称调用：",
            "",
        ]

        tools_to_show = available_tools or list(cls.TOOL_DOCS.keys())
        for name in tools_to_show:
            desc = cls.TOOL_DOCS.get(name, "")
            cat = cls.TOOL_CATEGORIES.get(name) or ("skill" if name not in cls.TOOL_DOCS else "core")
            tag = f" [{cat}]" if cat else ""
            lines.append(f"- {name}{tag}: {desc}")

        return "\n".join(lines)

    def _build_skills(self, agent_id: str) -> str:
        agent_dir = resolve_agent_dir(agent_id)
        snapshot_path = agent_dir / "SKILLS_SNAPSHOT.md"
        if not snapshot_path.exists():
            return ""
        content = snapshot_path.read_text(encoding="utf-8")
        if not content.strip():
            return ""
        return f"## 技能快照\n\n{content}"

    @staticmethod
    def _build_heartbeat_config(params: PromptParams) -> str:
        if params.heartbeat_prompt is not None:
            hb_prompt = params.heartbeat_prompt
        else:
            hb_prompt = get_heartbeat_config(params.agent_id).get("prompt", "")
        if hb_prompt:
            return f"## 心跳配置\n\n心跳 prompt：{hb_prompt}"
        return ""

    @staticmethod
    def _build_time(agent_id: str) -> str:
        cfg = resolve_agent_config(agent_id)
        tz = cfg.get("user_timezone", "Asia/Shanghai")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"## 当前时间\n\n时区: {tz}\n当前时间: {now}"

    @staticmethod
    def _build_workspace(agent_id: str) -> str:
        workspace = resolve_agent_workspace(agent_id)
        return (
            "## 工作区\n\n"
            f"你的工作目录是: {workspace}\n"
            "除非另有明确指示，所有文件操作都在此目录内进行。"
        )

    @staticmethod
    def _build_runtime(agent_id: str) -> str:
        cfg = resolve_agent_config(agent_id)
        os_info = f"{platform.system()} {platform.release()} ({platform.machine()})"
        model = cfg.get("model", "deepseek-chat")
        thinking = cfg.get("thinkingDefault", "off")
        return (
            "## 运行时信息\n\n"
            f"Runtime: agent={agent_id} | 系统={os_info} | "
            f"模型={model} | 通道=webchat | thinking={thinking}"
        )

    # ------------------------------------------------------------------
    # Project Context (with report)
    # ------------------------------------------------------------------

    def _build_project_context_with_report(
        self,
        agent_id: str,
        mode: str = "full",
    ) -> tuple[str, list[PromptFileEntry], int]:
        workspace = resolve_agent_workspace(agent_id)

        if mode == "minimal":
            context_files = [
                ("AGENTS.md", workspace / "AGENTS.md"),
            ]
        else:
            context_files = [
                ("AGENTS.md", workspace / "AGENTS.md"),
                ("IDENTITY.md", workspace / "IDENTITY.md"),
                ("USER.md", workspace / "USER.md"),
            ]

        if mode == "minimal":
            lines: list[str] = []
        else:
            lines = [
                "---",
                "",
                "# 项目上下文",
                "",
                "以下项目上下文文件已加载：",
                "如果存在 IDENTITY.md，请体现其中定义的人格和语气。",
            ]

        file_entries: list[PromptFileEntry] = []
        truncation_events = 0

        for label, path in context_files:
            if not path.exists():
                lines.append(f"\n## {label}")
                lines.append("[MISSING] — 文件不存在，Agent 可通过文件工具创建。")
                file_entries.append(PromptFileEntry(label=label, chars=0, truncated=False))
                continue

            try:
                content = path.read_text(encoding="utf-8")
            except Exception:
                lines.append(f"\n## {label}")
                lines.append("[ERROR] — 文件读取失败。")
                file_entries.append(PromptFileEntry(label=label, chars=0, truncated=False))
                continue

            if not content.strip():
                lines.append(f"\n## {label}")
                lines.append("[EMPTY] — 文件为空。")
                file_entries.append(PromptFileEntry(label=label, chars=0, truncated=False))
                continue

            if mode == "minimal" and label == "AGENTS.md":
                parts = []
                for heading in self.MINIMAL_AGENTS_SECTIONS:
                    section = self._extract_section(content, heading)
                    if section:
                        parts.append(section)
                content = "\n\n".join(parts) if parts else content

            file_entries.append(PromptFileEntry(label=label, chars=len(content), truncated=False))
            lines.append(f"\n## {label}")
            lines.append(content)

        return "\n".join(lines), file_entries, truncation_events


prompt_builder = PromptBuilder()
