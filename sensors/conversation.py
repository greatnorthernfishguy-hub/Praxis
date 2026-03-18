"""
Praxis Conversation Sensor — Pheromone 1

Captures conversation turns (human and AI) as WorkflowSignal objects.
This is a stub for Phase 1 — full implementation in Phase 2 (v0.2).

Canonical source: https://github.com/greatnorthernfishguy-hub/Praxis
License: AGPL-3.0

# ---- Changelog ----
# [2026-03-18] Claude (Opus 4.6) — Initial creation (Phase 1).
#   What: ConversationSensor stub with interface definition.
#   Why:  PRD §14 Phase 1 — sensor interface definitions.
#         Full implementation deferred to Phase 2.
#   How:  Extends SensorBase. collect_signals() returns empty list.
#         feed() method defined for Phase 2 implementation.
# -------------------
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from core.signals import ConversationMeta, WorkflowSignal
from sensors.base import SensorBase


class ConversationSensor(SensorBase):
    """Pheromone 1: Conversation Stream (PRD §4.1).

    Observes every message in both directions — what the human says
    AND what the AI responds. The pair (human utterance + AI response +
    temporal position) is the signal unit.

    Phase 1: Interface defined. Signals created via feed() calls from
    the hook. Full autonomous capture in Phase 2.
    """

    SENSOR_TYPE = "conversation"

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._buffer: List[WorkflowSignal] = []
        self._total_captured = 0

    def feed(
        self,
        text: str,
        direction: str = "human",
        session_id: str = "",
        speaker_id: str = "default",
        modality: str = "text",
        layer_depth: str = "implementation",
    ) -> WorkflowSignal:
        """Create a conversation signal from a message.

        Called by the hook on each on_message(). Embedding is handled
        by the hook — this sensor only builds the signal structure.

        Args:
            text: The message content
            direction: 'human' or 'ai'
            session_id: Current session identifier
            speaker_id: Speaker identifier for multi-speaker
            modality: 'text' or 'voice'
            layer_depth: Intent decomposition layer

        Returns:
            WorkflowSignal ready for embedding and substrate ingestion.
        """
        self._total_captured += 1

        meta = ConversationMeta(
            direction=direction,
            speaker_id=speaker_id,
            modality=modality,
            turn_index=self._total_captured,
            message_length=len(text),
        )

        signal = WorkflowSignal(
            pheromone="conversation",
            timestamp=time.time(),
            metadata=meta.__dict__,
            session_id=session_id,
            layer_depth=layer_depth,
        )

        self._buffer.append(signal)
        return signal

    def collect_signals(self) -> List[WorkflowSignal]:
        """Return buffered signals and clear the buffer."""
        signals = self._buffer[:]
        self._buffer.clear()
        return signals

    def get_stats(self) -> Dict[str, Any]:
        return {
            "sensor_type": self.SENSOR_TYPE,
            "enabled": True,
            "total_captured": self._total_captured,
            "buffer_size": len(self._buffer),
        }
