"""PIPIXIA 后端入口 — FastAPI + Uvicorn"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app_lifecycle import ApplicationLifecycle
from config import load_config, DATA_DIR, list_agents
from tools.skills_scanner import scan_all_skills
from tools.skills_watcher import skills_watcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8002",
    "http://127.0.0.1:8002",
)


def configure_session_work_recovery(
    *,
    resolver=None,
    cron_recovery_callbacks=None,
) -> None:
    if resolver is None:
        from sessions.session_work_recovery_resolver import (
            session_work_recovery_resolver,
        )

        resolver = session_work_recovery_resolver
    def resolve_cron_callbacks(record):
        if not record.run_id:
            return {}
        callbacks = cron_recovery_callbacks
        if callbacks is None:
            from scheduler.cron_service import cron_service

            callbacks = cron_service.recovery_callbacks
        return callbacks(record.run_id, record.id)

    resolver.bind("cron", resolve_cron_callbacks)


def resolve_cors_settings(
    environ: Mapping[str, str],
) -> tuple[list[str], str | None]:
    configured_origins = environ.get("PIPIXIA_CORS_ORIGINS", "").strip()
    if configured_origins:
        origins = [
            origin.strip()
            for origin in configured_origins.split(",")
            if origin.strip()
        ]
    else:
        origins = list(DEFAULT_CORS_ORIGINS)

    origin_regex = environ.get("PIPIXIA_CORS_ORIGIN_REGEX", "").strip() or None
    return origins, origin_regex


def _setup_logging_from_config() -> None:
    """根据配置设置日志文件轮转"""
    from config import get_config
    from logging.handlers import RotatingFileHandler

    cfg = get_config()
    app_cfg = cfg.get("app", {})

    # 设置日志级别
    log_level = app_cfg.get("logLevel", "info").upper()
    logging.getLogger().setLevel(getattr(logging, log_level, logging.INFO))

    # 配置文件日志
    log_file_cfg = app_cfg.get("logFile", {})
    if log_file_cfg.get("enabled", True):
        max_bytes = log_file_cfg.get("maxBytes", 10 * 1024 * 1024)  # 默认10MB
        backup_count = log_file_cfg.get("backupCount", 5)

        log_dir = DATA_DIR / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "app.log"

        # 检查是否已有文件处理器
        root_logger = logging.getLogger()
        has_file_handler = any(
            isinstance(h, RotatingFileHandler) for h in root_logger.handlers
        )
        if not has_file_handler:
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
            )
            root_logger.addHandler(file_handler)
            logger.info(f"File logging enabled: {log_file} (maxBytes={max_bytes}, backupCount={backup_count})")


def _get_session_work_delivery():
    from sessions.session_work_delivery import session_work_delivery

    return session_work_delivery


def _get_config() -> dict:
    from config import get_config

    return get_config()


def _create_cron_scheduler():
    from scheduler.cron_scheduler import CronScheduler
    from sessions.session_work_runtime import session_work_runtime

    return CronScheduler(runtime=session_work_runtime)


async def _resume_subagent_runs() -> None:
    from subagents.subagent_resume import resume_subagent_runs

    await resume_subagent_runs()


@asynccontextmanager
async def lifespan(application: FastAPI):
    """启动时初始化：配置、技能扫描、Agent 引擎、Heartbeat、技能热加载"""
    from runtime.agent import agent_manager
    from system_messages.heartbeat import heartbeat_runner
    from subagents.subagent_archive import (
        start_subagent_archive,
        stop_subagent_archive,
    )

    lifecycle = ApplicationLifecycle(
        load_config=load_config,
        setup_logging=_setup_logging_from_config,
        scan_skills=scan_all_skills,
        agent_manager=agent_manager,
        skills_watcher=skills_watcher,
        configure_work_recovery=configure_session_work_recovery,
        work_delivery_provider=_get_session_work_delivery,
        list_agents=list_agents,
        heartbeat_runner=heartbeat_runner,
        start_subagent_archive=start_subagent_archive,
        stop_subagent_archive=stop_subagent_archive,
        get_config=_get_config,
        cron_scheduler_factory=_create_cron_scheduler,
        resume_subagent_runs=_resume_subagent_runs,
        data_dir=DATA_DIR,
        log=logger,
    )
    try:
        await lifecycle.start(application)
        yield
    finally:
        await lifecycle.stop(application)
        from tools.rag_mcp_client import close_rag_mcp_client
        await close_rag_mcp_client()


app = FastAPI(title="PIPIXIA", version="0.2.0", lifespan=lifespan)

_cors_origins, _cors_origin_regex = resolve_cors_settings(os.environ)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_origin_regex=_cors_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
from api.chat import router as chat_router          # noqa: E402
from api.agents import router as agents_router      # noqa: E402
from api.sessions import router as sessions_router  # noqa: E402
from api.files import router as files_router        # noqa: E402
from api.compress import router as compress_router  # noqa: E402
from api.config_api import router as config_router  # noqa: E402
from api.events import router as events_router      # noqa: E402
from api.cron_api import router as cron_router      # noqa: E402
from api.approvals import router as approvals_router  # noqa: E402
from api.mem_api import router as mem_router          # noqa: E402

app.include_router(chat_router, prefix="/api")
app.include_router(agents_router, prefix="/api")
app.include_router(sessions_router, prefix="/api")
app.include_router(files_router, prefix="/api")
app.include_router(compress_router, prefix="/api")
app.include_router(config_router, prefix="/api")
app.include_router(events_router, prefix="/api")
app.include_router(cron_router, prefix="/api")
app.include_router(approvals_router, prefix="/api")
app.include_router(mem_router, prefix="/api")


@app.get("/api/health")
async def health() -> dict[str, str]:
    """Service liveness/readiness probe."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8002, reload=True)
