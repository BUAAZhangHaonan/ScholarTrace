"""Pipeline timing instrumentation for ScholarTrace."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Generator

logger = logging.getLogger(__name__)


class PipelineTimer:
    """Hierarchical timing for pipeline stages.

    Usage::

        timer = PipelineTimer("query_pipeline")
        with timer.stage("retrieval"):
            ...
            with timer.stage("connector: arxiv"):
                ...
        timer.log_summary()
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._start = time.monotonic()
        self._depth = 0
        self._records: list[tuple[int, str, float]] = []

    @contextmanager
    def stage(self, name: str) -> Generator[None, None, None]:
        """Time a named stage, logging on entry and exit."""
        indent = "  " * self._depth
        logger.info("[TIMING] %s%s ...", indent, name)
        stage_start = time.monotonic()
        self._depth += 1
        try:
            yield
        finally:
            self._depth -= 1
            elapsed = time.monotonic() - stage_start
            self._records.append((self._depth, name, elapsed))
            logger.info("[TIMING] %s%s took %.2fs", indent, name, elapsed)

    def total_elapsed(self) -> float:
        return time.monotonic() - self._start

    def log_summary(self) -> None:
        total = self.total_elapsed()
        logger.info("[TIMING] === %s total: %.2fs ===", self._name, total)
        for depth, name, elapsed in self._records:
            indent = "  " * depth
            pct = (elapsed / total * 100) if total > 0 else 0
            logger.info("[TIMING] %s%s: %.2fs (%.1f%%)", indent, name, elapsed, pct)
