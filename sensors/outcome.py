"""
Praxis Outcome Sensor — Pheromone 3

Captures success/failure results and produces three-factor reward signals
via inject_reward(). Outcomes trace back to originating intent signals
through the substrate's causal topology.

Canonical source: https://github.com/greatnorthernfishguy-hub/Praxis
License: AGPL-3.0

# ---- Changelog ----
# [2026-03-18] Claude (Opus 4.6) — Initial creation (Phase 1).
#   What: OutcomeSensor stub with interface definition.
#   Why:  PRD §14 Phase 1.
#   How:  Extends SensorBase. collect_signals() returns empty list.
# [2026-03-18] Claude (Opus 4.6) — Phase 6 full implementation.
#   What: Outcome recording with reward signal generation, intent
#         tracing via related_intent_ids, configurable reward strengths,
#         CPS integration for OUTCOME entries.
#   Why:  PRD §4.3 — outcome signals are the three-factor reward for
#         the substrate. PRD §14 Phase 6.
#   How:  record_outcome() creates WorkflowSignal with OutcomeMeta,
#         generates reward parameters (strength, success) for the hook
#         to pass to ecosystem.record_outcome(). Intent tracing links
#         outcomes to originating signals via related_intent_ids.
# -------------------
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.signals import OutcomeMeta, WorkflowSignal
from sensors.base import SensorBase

logger = logging.getLogger("praxis.outcome")


class OutcomeSensor(SensorBase):
    """Pheromone 3: Outcome Stream (PRD §4.3).

    Observes results and produces reward signals:
    - Success (build passed, tests green, review survived) → positive reward
    - Failure (build broke, tests red, review rejected) → negative reward

    The sensor produces WorkflowSignals annotated with reward parameters.
    The hook uses these to call ecosystem.record_outcome() with appropriate
    strength, creating the three-factor learning signal in the substrate.

    Outcome signals can arrive long after the intent that produced them.
    The related_intent_ids field traces back through the substrate's
    causal topology to the originating intent expression.
    """

    SENSOR_TYPE = "outcome"

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

        self._positive_strength: float = config.get(
            "positive_reward_strength", 0.7
        )
        self._negative_strength: float = config.get(
            "negative_reward_strength", -0.5
        )

        # Signal buffer
        self._buffer: List[WorkflowSignal] = []

        # Counters
        self._total_outcomes = 0
        self._total_successes = 0
        self._total_failures = 0

        # Log path (set by hook)
        self._log_path: Optional[Path] = None

    def record_outcome(
        self,
        embedding: np.ndarray,
        outcome_type: str,
        success: bool,
        severity: float = 0.5,
        related_intent_ids: Optional[List[str]] = None,
        session_id: str = "",
        layer_depth: str = "implementation",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[WorkflowSignal, float]:
        """Record an outcome event.

        Args:
            embedding: Semantic embedding of the outcome context
            outcome_type: 'build' | 'test' | 'review' | 'deploy' | 'bug'
            success: Whether the outcome was positive
            severity: 0.0 to 1.0, scales reward strength
            related_intent_ids: Signal IDs of originating intents
            session_id: Current session
            layer_depth: Intent layer
            metadata: Additional context

        Returns:
            (WorkflowSignal, reward_strength) — the signal and the
            computed reward strength for the hook to use with
            ecosystem.record_outcome().
        """
        self._total_outcomes += 1
        if success:
            self._total_successes += 1
            base_strength = self._positive_strength
        else:
            self._total_failures += 1
            base_strength = self._negative_strength

        # Scale reward by severity
        reward_strength = base_strength * severity

        meta = OutcomeMeta(
            outcome_type=outcome_type,
            success=success,
            severity=severity,
            related_intent_ids=related_intent_ids or [],
        )

        extra_meta = dict(metadata or {})
        extra_meta.update(meta.__dict__)

        signal = WorkflowSignal(
            pheromone="outcome",
            timestamp=time.time(),
            embedding=embedding,
            metadata=extra_meta,
            session_id=session_id,
            layer_depth=layer_depth,
        )

        self._buffer.append(signal)
        self._log_signal(signal)
        return signal, reward_strength

    def collect_signals(self) -> List[WorkflowSignal]:
        """Return buffered signals and clear."""
        signals = self._buffer[:]
        self._buffer.clear()
        return signals

    def get_stats(self) -> Dict[str, Any]:
        return {
            "sensor_type": self.SENSOR_TYPE,
            "enabled": True,
            "total_outcomes": self._total_outcomes,
            "total_successes": self._total_successes,
            "total_failures": self._total_failures,
            "success_rate": (
                round(self._total_successes / self._total_outcomes, 4)
                if self._total_outcomes > 0 else 0.0
            ),
            "positive_strength": self._positive_strength,
            "negative_strength": self._negative_strength,
            "buffer_size": len(self._buffer),
        }

    def set_log_path(self, path: Path) -> None:
        self._log_path = path

    # -----------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------

    def _log_signal(self, signal: WorkflowSignal) -> None:
        if self._log_path is None:
            return
        try:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(signal.to_dict()) + "\n")
        except Exception:
            pass
