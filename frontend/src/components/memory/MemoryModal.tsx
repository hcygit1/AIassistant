"use client";

import { useState, useEffect, useCallback } from "react";
import { useApp } from "@/lib/store";
import * as api from "@/lib/api";
import {
  X, BrainCircuit, BarChart3, ListTodo, Sparkles, Search,
  ChevronRight, ChevronLeft, Loader2, Database, Clock,
  FileText, MessageSquare, Bot, User, Filter, RefreshCw,
} from "lucide-react";

/* ================================================================
   Types
   ================================================================ */

interface MemStats {
  totalChunks: number;
  totalTasks: number;
  completedTasks: number;
  totalSkills: number;
  totalSessions: number;
  roleBreakdown: Record<string, number>;
  dedupBreakdown: Record<string, number>;
  timeRange: { earliest: string | null; latest: string | null };
}

interface TaskItem {
  id: string;
  sessionKey: string;
  title: string;
  summary: string;
  status: string;
  startedAt: number;
  endedAt: number | null;
  chunkCount: number;
}

interface TaskDetail {
  id: string;
  title: string;
  summary: string;
  status: string;
  startedAt: number;
  endedAt: number | null;
  chunks: { id: string; role: string; content: string; summary: string; createdAt: number }[];
}

interface SkillItem {
  id: string;
  name: string;
  description: string;
  version: number;
  status: string;
  qualityScore: number | null;
  createdAt: number;
  updatedAt: number;
}

interface MemoryItem {
  id: string;
  sessionKey: string;
  role: string;
  summary: string;
  excerpt: string;
  taskId: string | null;
  createdAt: number;
}

interface SearchResult {
  id: string;
  score: number;
  role: string;
  summary: string;
  excerpt: string;
  sessionKey: string;
  taskId: string | null;
  createdAt: number;
}

/* ================================================================
   Helpers
   ================================================================ */

function formatTs(ts: number | null) {
  if (!ts) return "—";
  const d = new Date(ts * 1000);
  return d.toLocaleString("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
}

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    active: "var(--success)", completed: "var(--accent)", merged: "var(--warning)",
    deprecated: "var(--text-tertiary)", superseded: "var(--text-tertiary)",
  };
  const c = colors[status] || "var(--text-secondary)";
  return (
    <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium"
      style={{ background: `color-mix(in srgb, ${c} 15%, transparent)`, color: c }}>
      <span className="w-1.5 h-1.5 rounded-full" style={{ background: c }} />
      {status}
    </span>
  );
}

function RoleIcon({ role }: { role: string }) {
  if (role === "assistant") return <Bot className="w-3 h-3" style={{ color: "var(--accent)" }} />;
  if (role === "user") return <User className="w-3 h-3" style={{ color: "var(--success)" }} />;
  return <MessageSquare className="w-3 h-3" style={{ color: "var(--text-secondary)" }} />;
}

function StatCard({ label, value, icon }: { label: string; value: string | number; icon: React.ReactNode }) {
  return (
    <div className="glass-card p-3 flex items-center gap-3">
      <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
        style={{ background: "var(--accent-bg)" }}>
        {icon}
      </div>
      <div>
        <div className="text-lg font-bold tabular-nums" style={{ color: "var(--text)" }}>{value}</div>
        <div className="text-[10px]" style={{ color: "var(--text-secondary)" }}>{label}</div>
      </div>
    </div>
  );
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 gap-3" style={{ color: "var(--text-tertiary)" }}>
      <Database className="w-8 h-8" />
      <p className="text-xs">{message}</p>
    </div>
  );
}

/* ================================================================
   Tab: 概览 (Overview)
   ================================================================ */

function OverviewTab() {
  const [stats, setStats] = useState<MemStats | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const r = await api.memStats();
      if (r.ok) setStats(r);
    } catch { /* ignore */ }
    setLoading(false);
  }, []);

  useEffect(() => { load(); }, [load]);

  if (loading) return <div className="flex items-center justify-center py-20"><Loader2 className="w-5 h-5 animate-spin" style={{ color: "var(--accent)" }} /></div>;
  if (!stats) return <EmptyState message="记忆系统尚未初始化" />;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold" style={{ color: "var(--text-secondary)" }}>数据概览</h3>
        <button onClick={load} className="btn-ghost p-1" type="button"><RefreshCw className="w-3.5 h-3.5" /></button>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <StatCard label="记忆片段" value={stats.totalChunks} icon={<FileText className="w-4 h-4" style={{ color: "var(--accent)" }} />} />
        <StatCard label="任务" value={stats.totalTasks} icon={<ListTodo className="w-4 h-4" style={{ color: "var(--accent)" }} />} />
        <StatCard label="技能" value={stats.totalSkills} icon={<Sparkles className="w-4 h-4" style={{ color: "var(--accent)" }} />} />
        <StatCard label="会话" value={stats.totalSessions} icon={<MessageSquare className="w-4 h-4" style={{ color: "var(--accent)" }} />} />
      </div>

      {/* Role Breakdown */}
      <div className="glass-card p-3 space-y-2">
        <div className="text-[11px] font-semibold" style={{ color: "var(--text)" }}>角色分布</div>
        <div className="flex gap-2 flex-wrap">
          {Object.entries(stats.roleBreakdown).map(([role, count]) => (
            <div key={role} className="flex items-center gap-1.5 px-2 py-1 rounded-md text-[11px]"
              style={{ background: "var(--bg-inset)" }}>
              <RoleIcon role={role} />
              <span style={{ color: "var(--text)" }}>{role}</span>
              <span className="font-bold tabular-nums" style={{ color: "var(--accent)" }}>{count}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Dedup Breakdown */}
      <div className="glass-card p-3 space-y-2">
        <div className="text-[11px] font-semibold" style={{ color: "var(--text)" }}>去重状态</div>
        <div className="flex gap-2 flex-wrap">
          {Object.entries(stats.dedupBreakdown).map(([status, count]) => (
            <div key={status} className="flex items-center gap-1.5 px-2 py-1 rounded-md text-[11px]"
              style={{ background: "var(--bg-inset)" }}>
              <StatusBadge status={status} />
              <span className="font-bold tabular-nums" style={{ color: "var(--text)" }}>{count}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Time Range */}
      <div className="glass-card p-3 space-y-1">
        <div className="text-[11px] font-semibold" style={{ color: "var(--text)" }}>时间范围</div>
        <div className="flex items-center gap-2 text-[11px]" style={{ color: "var(--text-secondary)" }}>
          <Clock className="w-3.5 h-3.5" />
          <span>{stats.timeRange.earliest ? formatTs(Number(stats.timeRange.earliest)) : "—"}</span>
          <span>→</span>
          <span>{stats.timeRange.latest ? formatTs(Number(stats.timeRange.latest)) : "—"}</span>
        </div>
      </div>

      {/* Task completion */}
      {stats.totalTasks > 0 && (
        <div className="glass-card p-3 space-y-2">
          <div className="text-[11px] font-semibold" style={{ color: "var(--text)" }}>任务完成率</div>
          <div className="h-2 rounded-full overflow-hidden" style={{ background: "var(--bg-inset)" }}>
            <div className="h-full rounded-full transition-all" style={{
              width: `${Math.round((stats.completedTasks / stats.totalTasks) * 100)}%`,
              background: "var(--accent)",
            }} />
          </div>
          <div className="text-[10px] tabular-nums" style={{ color: "var(--text-secondary)" }}>
            {stats.completedTasks} / {stats.totalTasks} 已完成 ({Math.round((stats.completedTasks / stats.totalTasks) * 100)}%)
          </div>
        </div>
      )}
    </div>
  );
}

/* ================================================================
   Tab: 任务 (Tasks)
   ================================================================ */

function TasksTab() {
  const [tasks, setTasks] = useState<TaskItem[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(0);
  const [statusFilter, setStatusFilter] = useState("");
  const [detail, setDetail] = useState<TaskDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const PAGE_SIZE = 20;

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const r = await api.memTasks({ status: statusFilter || undefined, limit: PAGE_SIZE, offset: page * PAGE_SIZE });
      if (r.ok) { setTasks(r.tasks); setTotal(r.total); }
    } catch { /* ignore */ }
    setLoading(false);
  }, [page, statusFilter]);

  useEffect(() => { load(); }, [load]);

  const openDetail = async (id: string) => {
    setDetailLoading(true);
    try {
      const r = await api.memTaskDetail(id);
      if (r.ok) setDetail(r);
    } catch { /* ignore */ }
    setDetailLoading(false);
  };

  if (detail) {
    return (
      <div className="space-y-3">
        <button onClick={() => setDetail(null)} className="flex items-center gap-1 text-[11px] btn-ghost px-2 py-1" type="button">
          <ChevronLeft className="w-3.5 h-3.5" /> 返回列表
        </button>
        <div className="glass-card p-4 space-y-3">
          <div className="flex items-start justify-between">
            <div>
              <h3 className="text-sm font-semibold" style={{ color: "var(--text)" }}>{detail.title || "未命名任务"}</h3>
              <p className="text-[10px] font-mono mt-0.5" style={{ color: "var(--text-tertiary)" }}>{detail.id}</p>
            </div>
            <StatusBadge status={detail.status} />
          </div>
          {detail.summary && (
            <p className="text-[11px] leading-relaxed" style={{ color: "var(--text-secondary)" }}>{detail.summary}</p>
          )}
          <div className="flex gap-3 text-[10px]" style={{ color: "var(--text-tertiary)" }}>
            <span>开始: {formatTs(detail.startedAt)}</span>
            {detail.endedAt && <span>结束: {formatTs(detail.endedAt)}</span>}
          </div>
        </div>

        <div className="text-[11px] font-semibold px-1" style={{ color: "var(--text-secondary)" }}>
          关联片段 ({detail.chunks.length})
        </div>
        <div className="space-y-1.5 max-h-[60vh] overflow-y-auto">
          {detail.chunks.map((c) => (
            <div key={c.id} className="glass-card p-2.5 space-y-1">
              <div className="flex items-center gap-1.5">
                <RoleIcon role={c.role} />
                <span className="text-[10px] font-medium" style={{ color: "var(--text)" }}>{c.role}</span>
                <span className="text-[9px] ml-auto" style={{ color: "var(--text-tertiary)" }}>{formatTs(c.createdAt)}</span>
              </div>
              {c.summary && <p className="text-[10px]" style={{ color: "var(--accent)" }}>{c.summary}</p>}
              <p className="text-[10px] leading-relaxed whitespace-pre-wrap" style={{ color: "var(--text-secondary)" }}>{c.content}</p>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold" style={{ color: "var(--text-secondary)" }}>任务列表</h3>
        <div className="flex items-center gap-2">
          <select
            value={statusFilter}
            onChange={(e) => { setStatusFilter(e.target.value); setPage(0); }}
            className="input text-[10px] py-0.5 px-1.5 w-24"
          >
            <option value="">全部状态</option>
            <option value="active">active</option>
            <option value="completed">completed</option>
          </select>
          <button onClick={load} className="btn-ghost p-1" type="button"><RefreshCw className="w-3.5 h-3.5" /></button>
        </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-16"><Loader2 className="w-5 h-5 animate-spin" style={{ color: "var(--accent)" }} /></div>
      ) : tasks.length === 0 ? (
        <EmptyState message="暂无任务" />
      ) : (
        <>
          <div className="space-y-1.5">
            {tasks.map((t) => (
              <button key={t.id} type="button" onClick={() => openDetail(t.id)}
                className="w-full text-left glass-card p-3 space-y-1.5 transition-all"
                onMouseEnter={e => (e.currentTarget.style.background = "var(--hover)")}
                onMouseLeave={e => (e.currentTarget.style.background = "")}>
                <div className="flex items-start justify-between gap-2">
                  <span className="text-[12px] font-medium truncate" style={{ color: "var(--text)" }}>{t.title || "未命名"}</span>
                  <StatusBadge status={t.status} />
                </div>
                {t.summary && <p className="text-[10px] line-clamp-2" style={{ color: "var(--text-secondary)" }}>{t.summary}</p>}
                <div className="flex items-center gap-3 text-[9px]" style={{ color: "var(--text-tertiary)" }}>
                  <span>{formatTs(t.startedAt)}</span>
                  <span>{t.chunkCount} 片段</span>
                </div>
              </button>
            ))}
          </div>

          {/* Pagination */}
          {total > PAGE_SIZE && (
            <div className="flex items-center justify-center gap-2 pt-2">
              <button disabled={page === 0} onClick={() => setPage(p => p - 1)} className="btn-ghost p-1.5 disabled:opacity-30" type="button">
                <ChevronLeft className="w-3.5 h-3.5" />
              </button>
              <span className="text-[10px] tabular-nums" style={{ color: "var(--text-secondary)" }}>
                {page + 1} / {Math.ceil(total / PAGE_SIZE)}
              </span>
              <button disabled={(page + 1) * PAGE_SIZE >= total} onClick={() => setPage(p => p + 1)} className="btn-ghost p-1.5 disabled:opacity-30" type="button">
                <ChevronRight className="w-3.5 h-3.5" />
              </button>
            </div>
          )}
        </>
      )}

      {detailLoading && (
        <div className="fixed inset-0 z-[70] flex items-center justify-center" style={{ background: "rgba(0,0,0,0.2)" }}>
          <Loader2 className="w-6 h-6 animate-spin" style={{ color: "var(--accent)" }} />
        </div>
      )}
    </div>
  );
}

/* ================================================================
   Tab: 技能 (Skills)
   ================================================================ */

function SkillsTab() {
  const [skills, setSkills] = useState<SkillItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [statusFilter, setStatusFilter] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const r = await api.memSkills({ status: statusFilter || undefined });
      if (r.ok) setSkills(r.skills);
    } catch { /* ignore */ }
    setLoading(false);
  }, [statusFilter]);

  useEffect(() => { load(); }, [load]);

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold" style={{ color: "var(--text-secondary)" }}>技能列表</h3>
        <div className="flex items-center gap-2">
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="input text-[10px] py-0.5 px-1.5 w-24"
          >
            <option value="">全部</option>
            <option value="active">active</option>
            <option value="deprecated">deprecated</option>
          </select>
          <button onClick={load} className="btn-ghost p-1" type="button"><RefreshCw className="w-3.5 h-3.5" /></button>
        </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-16"><Loader2 className="w-5 h-5 animate-spin" style={{ color: "var(--accent)" }} /></div>
      ) : skills.length === 0 ? (
        <EmptyState message="暂无技能" />
      ) : (
        <div className="space-y-1.5">
          {skills.map((s) => (
            <div key={s.id} className="glass-card p-3 space-y-1.5">
              <div className="flex items-start justify-between gap-2">
                <div className="flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5 flex-shrink-0" style={{ color: "var(--accent)" }} />
                  <span className="text-[12px] font-medium" style={{ color: "var(--text)" }}>{s.name}</span>
                  <span className="text-[9px] px-1 py-0.5 rounded" style={{ background: "var(--bg-inset)", color: "var(--text-tertiary)" }}>v{s.version}</span>
                </div>
                <StatusBadge status={s.status} />
              </div>
              {s.description && <p className="text-[10px] line-clamp-2" style={{ color: "var(--text-secondary)" }}>{s.description}</p>}
              <div className="flex items-center gap-3 text-[9px]" style={{ color: "var(--text-tertiary)" }}>
                {s.qualityScore != null && <span>质量: {(s.qualityScore * 100).toFixed(0)}%</span>}
                <span>更新: {formatTs(s.updatedAt)}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ================================================================
   Tab: 记忆搜索 (Memory Search)
   ================================================================ */

function SearchTab() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);

  const [memories, setMemories] = useState<MemoryItem[]>([]);
  const [memTotal, setMemTotal] = useState(0);
  const [memPage, setMemPage] = useState(1);
  const [memLoading, setMemLoading] = useState(true);
  const [roleFilter, setRoleFilter] = useState("");

  const loadMemories = useCallback(async () => {
    setMemLoading(true);
    try {
      const r = await api.memMemories({ page: memPage, limit: 30, role: roleFilter || undefined });
      if (r.ok) { setMemories(r.memories); setMemTotal(r.totalPages); }
    } catch { /* ignore */ }
    setMemLoading(false);
  }, [memPage, roleFilter]);

  useEffect(() => { loadMemories(); }, [loadMemories]);

  const doSearch = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setSearched(true);
    try {
      const r = await api.memSearch(query.trim());
      if (r.ok) setResults(r.results || []);
    } catch { /* ignore */ }
    setLoading(false);
  };

  return (
    <div className="space-y-4">
      {/* Search bar */}
      <div className="flex gap-2">
        <div className="flex-1 relative">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5" style={{ color: "var(--text-tertiary)" }} />
          <input
            className="input text-xs pl-8 pr-3 py-2 w-full"
            placeholder="搜索记忆片段..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && doSearch()}
          />
        </div>
        <button onClick={doSearch} disabled={loading || !query.trim()} className="btn-ghost px-3 py-1.5 text-[11px] font-medium disabled:opacity-40" type="button"
          style={{ background: "var(--accent-bg)", color: "var(--accent)" }}>
          {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : "搜索"}
        </button>
      </div>

      {/* Search Results */}
      {searched && (
        <div className="space-y-2">
          <div className="text-[11px] font-semibold" style={{ color: "var(--text-secondary)" }}>
            搜索结果 ({results.length})
          </div>
          {results.length === 0 ? (
            <p className="text-[10px] py-4 text-center" style={{ color: "var(--text-tertiary)" }}>未找到相关记忆</p>
          ) : (
            <div className="space-y-1.5 max-h-[40vh] overflow-y-auto">
              {results.map((r) => (
                <div key={r.id} className="glass-card p-2.5 space-y-1">
                  <div className="flex items-center gap-1.5">
                    <RoleIcon role={r.role} />
                    <span className="text-[10px] font-medium" style={{ color: "var(--text)" }}>{r.role}</span>
                    <span className="text-[9px] px-1 rounded tabular-nums" style={{ background: "var(--accent-bg)", color: "var(--accent)" }}>
                      {r.score.toFixed(2)}
                    </span>
                    <span className="text-[9px] ml-auto" style={{ color: "var(--text-tertiary)" }}>{formatTs(r.createdAt)}</span>
                  </div>
                  {r.summary && <p className="text-[10px]" style={{ color: "var(--accent)" }}>{r.summary}</p>}
                  <p className="text-[10px] leading-relaxed line-clamp-3" style={{ color: "var(--text-secondary)" }}>{r.excerpt}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Browse all memories */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <div className="text-[11px] font-semibold" style={{ color: "var(--text-secondary)" }}>全部记忆</div>
          <div className="flex items-center gap-2">
            <select value={roleFilter} onChange={(e) => { setRoleFilter(e.target.value); setMemPage(1); }}
              className="input text-[10px] py-0.5 px-1.5 w-24">
              <option value="">全部角色</option>
              <option value="user">user</option>
              <option value="assistant">assistant</option>
              <option value="tool">tool</option>
            </select>
            <button onClick={loadMemories} className="btn-ghost p-1" type="button"><RefreshCw className="w-3.5 h-3.5" /></button>
          </div>
        </div>

        {memLoading ? (
          <div className="flex items-center justify-center py-8"><Loader2 className="w-4 h-4 animate-spin" style={{ color: "var(--accent)" }} /></div>
        ) : memories.length === 0 ? (
          <EmptyState message="暂无记忆片段" />
        ) : (
          <>
            <div className="space-y-1 max-h-[40vh] overflow-y-auto">
              {memories.map((m) => (
                <div key={m.id} className="glass-card p-2.5 space-y-1">
                  <div className="flex items-center gap-1.5">
                    <RoleIcon role={m.role} />
                    <span className="text-[10px] font-medium" style={{ color: "var(--text)" }}>{m.role}</span>
                    {m.taskId && (
                      <span className="text-[9px] px-1 rounded truncate max-w-[100px]"
                        style={{ background: "var(--bg-inset)", color: "var(--text-tertiary)" }}
                        title={m.taskId}>
                        任务关联
                      </span>
                    )}
                    <span className="text-[9px] ml-auto flex-shrink-0" style={{ color: "var(--text-tertiary)" }}>{formatTs(m.createdAt)}</span>
                  </div>
                  {m.summary && <p className="text-[10px]" style={{ color: "var(--accent)" }}>{m.summary}</p>}
                  <p className="text-[10px] leading-relaxed line-clamp-2" style={{ color: "var(--text-secondary)" }}>{m.excerpt}</p>
                </div>
              ))}
            </div>

            {memTotal > 1 && (
              <div className="flex items-center justify-center gap-2 pt-1">
                <button disabled={memPage <= 1} onClick={() => setMemPage(p => p - 1)} className="btn-ghost p-1.5 disabled:opacity-30" type="button">
                  <ChevronLeft className="w-3.5 h-3.5" />
                </button>
                <span className="text-[10px] tabular-nums" style={{ color: "var(--text-secondary)" }}>
                  {memPage} / {memTotal}
                </span>
                <button disabled={memPage >= memTotal} onClick={() => setMemPage(p => p + 1)} className="btn-ghost p-1.5 disabled:opacity-30" type="button">
                  <ChevronRight className="w-3.5 h-3.5" />
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

/* ================================================================
   Main Modal
   ================================================================ */

const TABS = [
  { key: "overview", label: "概览", icon: BarChart3 },
  { key: "tasks", label: "任务", icon: ListTodo },
  { key: "skills", label: "技能", icon: Sparkles },
  { key: "search", label: "记忆", icon: Search },
] as const;

type TabKey = (typeof TABS)[number]["key"];

export default function MemoryModal() {
  const { showMemoryModal, setShowMemoryModal } = useApp();
  const [tab, setTab] = useState<TabKey>("overview");

  useEffect(() => {
    if (!showMemoryModal) return;
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape") setShowMemoryModal(false); };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [showMemoryModal, setShowMemoryModal]);

  if (!showMemoryModal) return null;

  return (
    <>
      {/* Backdrop */}
      <div className="fixed inset-0 z-[60] transition-opacity"
        style={{ background: "rgba(0,0,0,0.35)", backdropFilter: "blur(6px)" }}
        onClick={() => setShowMemoryModal(false)} aria-hidden />

      {/* Modal */}
      <div className="fixed inset-4 sm:inset-8 z-[61] flex flex-col rounded-2xl overflow-hidden animate-scale-in"
        style={{
          background: "var(--glass-heavy)",
          backdropFilter: "blur(var(--blur-heavy)) saturate(1.8)",
          WebkitBackdropFilter: "blur(var(--blur-heavy)) saturate(1.8)",
          border: "1px solid var(--glass-border)",
          boxShadow: "var(--shadow-xl)",
        }}>

        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 flex-shrink-0"
          style={{ borderBottom: "1px solid var(--border)" }}>
          <div className="flex items-center gap-2.5">
            <BrainCircuit className="w-4.5 h-4.5" style={{ color: "var(--accent)" }} />
            <h2 className="text-sm font-semibold" style={{ color: "var(--text)" }}>记忆看板</h2>
          </div>
          <button onClick={() => setShowMemoryModal(false)} className="btn-ghost p-1.5" type="button">
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Tab Bar */}
        <div className="flex items-center px-5 gap-1 flex-shrink-0"
          style={{ borderBottom: "1px solid var(--border)" }}>
          {TABS.map((t) => (
            <button key={t.key} type="button" onClick={() => setTab(t.key)}
              className="flex items-center gap-1.5 px-3 py-2.5 text-[11px] font-medium transition-all relative"
              style={{ color: tab === t.key ? "var(--accent)" : "var(--text-secondary)" }}>
              <t.icon className="w-3.5 h-3.5" />
              <span>{t.label}</span>
              {tab === t.key && (
                <div className="absolute bottom-0 left-1 right-1 h-[2px] rounded-full" style={{ background: "var(--accent)" }} />
              )}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-5">
          {tab === "overview" && <OverviewTab />}
          {tab === "tasks" && <TasksTab />}
          {tab === "skills" && <SkillsTab />}
          {tab === "search" && <SearchTab />}
        </div>
      </div>
    </>
  );
}
