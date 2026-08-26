"""Optional Cross-Encoder reranking for memory candidates."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class MemReranker:
    """Cross-Encoder applied only to the small post-RRF candidate set."""

    def __init__(self, model: str, *, device: str = "cpu", max_length: int = 512):
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except ImportError as exc:
            raise RuntimeError("memory reranking requires sentence-transformers") from exc
        self.model_name = model
        self._model: Any = CrossEncoder(model, device=device, max_length=max_length)
        logger.info("Memory reranker loaded: %s", model)

    def score(self, query: str, candidates: list[tuple[str, str]]) -> list[float]:
        if not candidates:
            return []
        scores = self._model.predict(
            [(query, text) for _, text in candidates], show_progress_bar=False
        )
        return [float(score) for score in scores]
