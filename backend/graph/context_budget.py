"""全局上下文预算模型

从 contextTokens + 比率 派生所有上下文分配值，
作为 prompt_builder / token_counter / session_pruning 等模块的唯一参数来源。
"""

from __future__ import annotations

from dataclasses import dataclass

CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class ContextBudget:
    context_tokens: int = 200_000

    # 一级切分
    thinking_reserve: float = 0.20
    active_ratio: float = 0.80

    # 二级切分 (active 内部)
    session_summary_ratio: float = 0.05
    # 历史工具截断阈值
    jit_tool_output_ratio: float = 0.036

    # 压缩触发 (基于 active)
    sliding_ratio: float = 0.80
    forced_ratio: float = 0.95

    # 单文件截断上限 (chars)
    max_file_chars: int = 20_000

    # --- 派生属性 ---

    @property
    def active_tokens(self) -> int:
        return int(self.context_tokens * self.active_ratio)

    @property
    def session_summary_tokens(self) -> int:
        return int(self.active_tokens * self.session_summary_ratio)

    @property
    def session_summary_chars(self) -> int:
        return self.session_summary_tokens * CHARS_PER_TOKEN

    @property
    def jit_tool_output_tokens(self) -> int:
        return int(self.active_tokens * self.jit_tool_output_ratio)

    @property
    def jit_tool_output_chars(self) -> int:
        return self.jit_tool_output_tokens * CHARS_PER_TOKEN

    @property
    def sliding_threshold(self) -> int:
        return int(self.active_tokens * self.sliding_ratio)

    @property
    def forced_threshold(self) -> int:
        return int(self.active_tokens * self.forced_ratio)


_DEFAULT_BUDGET = ContextBudget()


def resolve_budget(agent_id: str | None = None) -> ContextBudget:
    """从 agent 配置读取 contextTokens + contextBudget 比率，返回 ContextBudget。"""
    if not agent_id:
        return _DEFAULT_BUDGET
    try:
        from config import resolve_agent_config
        cfg = resolve_agent_config(agent_id)
        budget_cfg: dict = cfg.get("contextBudget", {})
        context_tokens = cfg.get("contextTokens", 200_000)

        known_fields = {f.name for f in ContextBudget.__dataclass_fields__.values()}
        filtered = {k: v for k, v in budget_cfg.items() if k in known_fields}
        return ContextBudget(context_tokens=context_tokens, **filtered)
    except Exception:
        return _DEFAULT_BUDGET
