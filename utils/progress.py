from __future__ import annotations

import logging
from dataclasses import dataclass


@dataclass
class PipelineProgress:
    steps: list[str]
    logger: logging.Logger

    def __post_init__(self) -> None:
        self.total = len(self.steps)
        self.current = 0

    def advance(self, message: str | None = None) -> None:
        step_label = self.steps[self.current] if self.current < self.total else "post-processing"
        self.current += 1
        suffix = f" - {message}" if message else ""
        self.logger.info("Progress %d/%d: %s%s", self.current, self.total, step_label, suffix)
