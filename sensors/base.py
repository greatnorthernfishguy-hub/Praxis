"""
Praxis Sensor Base Class — Interface for the Three Pheromones

Abstract base class for all Praxis sensors. Each sensor observes one
dimension of the developer's workflow and produces WorkflowSignal objects.

Canonical source: https://github.com/greatnorthernfishguy-hub/Praxis
License: AGPL-3.0

# ---- Changelog ----
# [2026-03-18] Claude (Opus 4.6) — Initial creation (Phase 1).
#   What: SensorBase ABC defining the sensor contract.
#   Why:  PRD §4 — three pheromone sensors share a common interface.
#         Follows Immunis sensor architecture pattern (Immunis PRD §5).
#   How:  ABC with collect_signals() and get_stats() methods.
# -------------------
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from core.signals import WorkflowSignal


class SensorBase(ABC):
    """Abstract base class for Praxis pheromone sensors.

    Each sensor type implements:
      - SENSOR_TYPE: str class attribute identifying the pheromone
      - collect_signals(): gather new workflow signals
      - get_stats(): return sensor-specific telemetry
    """

    SENSOR_TYPE: str = ""

    def __init__(self, config: Dict[str, Any]) -> None:
        if not self.SENSOR_TYPE:
            raise ValueError("Subclass must set SENSOR_TYPE")
        self._config = config

    @abstractmethod
    def collect_signals(self) -> List[WorkflowSignal]:
        """Collect new workflow signals since last poll.

        Returns:
            List of WorkflowSignal objects ready for substrate ingestion.
            Embeddings are NOT included — the hook handles embedding.
        """
        ...

    def get_stats(self) -> Dict[str, Any]:
        """Return sensor-specific telemetry. Override for richer stats."""
        return {"sensor_type": self.SENSOR_TYPE, "enabled": True}

    def shutdown(self) -> None:
        """Graceful shutdown. Override if cleanup needed."""
        pass
