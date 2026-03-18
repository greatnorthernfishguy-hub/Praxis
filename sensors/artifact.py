"""
Praxis Artifact Sensor — Pheromone 2

Monitors document lifecycle events (create, modify, reference, delete).
This is a stub for Phase 1 — full implementation in Phase 5 (v0.5).

Canonical source: https://github.com/greatnorthernfishguy-hub/Praxis
License: AGPL-3.0

# ---- Changelog ----
# [2026-03-18] Claude (Opus 4.6) — Initial creation (Phase 1).
#   What: ArtifactSensor stub with interface definition.
#   Why:  PRD §14 Phase 1 — sensor interface definitions.
#         Full implementation deferred to Phase 5.
#   How:  Extends SensorBase. collect_signals() returns empty list.
# -------------------
"""

from __future__ import annotations

from typing import Any, Dict, List

from core.signals import WorkflowSignal
from sensors.base import SensorBase


class ArtifactSensor(SensorBase):
    """Pheromone 2: Artifact Stream (PRD §4.2).

    Observes documents, code files, PRDs, configuration files — any
    persistent artifact in the developer's workflow. Not just content,
    but lifecycle: creation, modification, reference, deletion.

    Phase 1: Stub. Full implementation in Phase 5.
    """

    SENSOR_TYPE = "artifact"

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

    def collect_signals(self) -> List[WorkflowSignal]:
        """Phase 1 stub — no autonomous artifact monitoring yet."""
        return []
