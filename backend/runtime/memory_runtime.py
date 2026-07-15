"""Memory component lifecycle and ingestion for Agent runtime."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from config import resolve_mem_config


logger = logging.getLogger("runtime.agent")


class MemoryRuntime:
    def __init__(self) -> None:
        self.stores: dict[str, Any] = {}
        self.embedders: dict[str, Any] = {}
        self.workers: dict[str, Any] = {}
        self.recalls: dict[str, Any] = {}

    @staticmethod
    def _close_store(agent_id: str, store: Any) -> None:
        try:
            store.close()
        except Exception as e:
            logger.warning("Failed to close mem store for %s: %s", agent_id, e)

    def _remove_agent(self, agent_id: str) -> None:
        store = self.stores.pop(agent_id, None)
        self.embedders.pop(agent_id, None)
        self.workers.pop(agent_id, None)
        self.recalls.pop(agent_id, None)
        if store is not None:
            self._close_store(agent_id, store)

    def initialize_agent(self, agent_id: str) -> None:
        try:
            mem_cfg = resolve_mem_config()
        except Exception as e:
            logger.error("Failed to initialize mem system for %s: %s", agent_id, e)
            return

        if not mem_cfg.get("enabled", True):
            self._remove_agent(agent_id)
            logger.info("Mem system disabled for %s", agent_id)
            return

        store: Any | None = None
        try:
            from mem.embedder import MemEmbedder
            from mem.recall import MemRecall
            from mem.skill_evolver import MemSkillEvolver
            from mem.store import MemStore
            from mem.task_processor import MemTaskProcessor
            from mem.worker import MemWorker

            store = MemStore(
                db_path=mem_cfg["storage"]["db_path"],
                dimensions=mem_cfg.get("embedding", {}).get("dimensions", 1536),
            )

            embedder = MemEmbedder.from_config(mem_cfg.get("embedding", {}))

            skill_store_dir = str(
                Path(mem_cfg["storage"]["db_path"]).parent / "skills-store"
            )
            skill_evolver = MemSkillEvolver.from_config(
                mem_cfg,
                store=store,
                embedder=embedder,
                skill_store_dir=skill_store_dir,
            )

            async def _on_task_completed(task: Any) -> None:
                await skill_evolver.on_task_completed(task)

            task_processor = MemTaskProcessor.from_config(
                mem_cfg,
                store=store,
                embedder=embedder,
                on_task_completed=_on_task_completed,
            )

            async def _on_chunks_ingested(
                session_key: str,
                session_end: bool,
            ) -> None:
                await task_processor.on_chunks_ingested(
                    session_key,
                    session_end,
                    owner=agent_id,
                )

            worker = MemWorker.from_config(
                mem_cfg,
                store=store,
                embedder=embedder,
                on_chunks_ingested=_on_chunks_ingested,
            )

            recall = MemRecall.from_config(
                mem_cfg,
                store=store,
                embedder=embedder,
                agent_id=agent_id,
            )
        except Exception as e:
            if store is not None:
                self._close_store(agent_id, store)
            logger.error("Failed to initialize mem system for %s: %s", agent_id, e)
            return

        previous_store = self.stores.get(agent_id)
        self.stores[agent_id] = store
        self.embedders[agent_id] = embedder
        self.workers[agent_id] = worker
        self.recalls[agent_id] = recall
        if previous_store is not None and previous_store is not store:
            self._close_store(agent_id, previous_store)
        logger.info("Mem system initialized for agent %s", agent_id)

    async def ingest_turn(
        self,
        agent_id: str,
        session_id: str,
        user_content: str,
        assistant_content: str,
    ) -> None:
        worker = self.workers.get(agent_id)
        if not worker:
            return
        try:
            from mem.worker import IngestMessage

            turn_id = str(uuid.uuid4())
            batch: list[IngestMessage] = []
            if user_content.strip():
                batch.append(
                    IngestMessage(
                        role="user",
                        content=user_content.strip(),
                        session_key=session_id,
                        turn_id=turn_id,
                        owner=agent_id,
                    )
                )
            if assistant_content.strip():
                batch.append(
                    IngestMessage(
                        role="assistant",
                        content=assistant_content.strip(),
                        session_key=session_id,
                        turn_id=turn_id,
                        owner=agent_id,
                    )
                )
            if batch:
                await worker.enqueue(batch, session_end=False)
        except Exception as e:
            logger.warning("incremental_ingest failed for %s: %s", agent_id, e)

    async def ingest_messages(
        self,
        agent_id: str,
        session_id: str,
        messages: list[dict[str, Any]],
        session_end: bool = False,
    ) -> None:
        worker = self.workers.get(agent_id)
        if not worker or not messages:
            return
        try:
            from mem.worker import IngestMessage

            batch: list[IngestMessage] = []
            for message in messages:
                content = message.get("content", "").strip()
                if not content:
                    continue
                role = message.get("role", "user")
                if role == "system":
                    continue
                batch.append(
                    IngestMessage(
                        role=role,
                        content=content,
                        session_key=session_id,
                        turn_id=str(uuid.uuid4()),
                        owner=agent_id,
                    )
                )
            if batch:
                await worker.enqueue(batch, session_end=session_end)
        except Exception as e:
            logger.warning("batch_ingest failed for %s: %s", agent_id, e)

    def close(self) -> None:
        for agent_id, store in list(self.stores.items()):
            self._close_store(agent_id, store)

        self.stores.clear()
        self.embedders.clear()
        self.workers.clear()
        self.recalls.clear()
