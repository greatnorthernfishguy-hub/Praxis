"""
Praxis Outcome Sensor — Pheromone 3

Captures success/failure results and produces reward signals.
This is a stub for Phase 1 — full implementation in Phase 6 (v0.6).

Canonical source: https://github.com/greatnorthernfishguy-hub/Praxis
License: AGPL-3.0

# ---- Changelog ----
# [2026-03-18] Claude (Opus 4.6) — Initial creation (Phase 1).
#   What: OutcomeSensor stub with interface definition.
#   Why:  PRD §14 Phase 1 — sensor interface definitions.
#         Full implementation deferred to Phase 6.
#   How:  Extends SensorBase. collect_signals() returns empty list.
# -------------------
"""

from __future__ import annotations

from typing import Any, Dict, List

from core.signals import WorkflowSignal
from sensors.base import SensorBase


class OutcomeSensor(SensorBase):
    """Pheromone 3: Outcome Stream (PRD §4.3).

    Observes results: code that compiled, tests that passed, PRD sections
    that survived review, deployments that succeeded, bugs that appeared.
    Produces three-factor reward signals via inject_reward().

    Phase 1: Stub. Full implementation in Phase 6.
    """

    SENSOR_TYPE = "outcome"

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

    def collect_signals(self) -> List[WorkflowSignal]:
        """Phase 1 stub — no autonomous outcome monitoring yet."""
        return []
