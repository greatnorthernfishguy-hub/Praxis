"""
Praxis Conversation Sensor — Pheromone 1

Captures conversation turns (human and AI) as WorkflowSignal objects,
embeds them, and manages the temporal signal window for substrate
binding.

The sensor maintains a ring buffer of recent signals with their
embeddings. When a new signal arrives, temporal synapses are created
to all signals within the configurable temporal window (default 300s).
This is the "retinal preprocessing" — the substrate handles all
intelligence from there.

Canonical source: https://github.com/greatnorthernfishguy-hub/Praxis
License: AGPL-3.0

# ---- Changelog ----
# [2026-03-18] Claude (Opus 4.6) — Initial creation (Phase 1).
#   What: ConversationSensor stub with interface definition.
#   Why:  PRD §14 Phase 1 — sensor interface definitions.
#   How:  Extends SensorBase. feed() creates WorkflowSignal.
# [2026-03-18] Claude (Opus 4.6) — Phase 2 full implementation.
#   What: Temporal signal window, substrate node creation, temporal
#         synapse binding between signals within the temporal window.
#   Why:  PRD §14 Phase 2 — full Conversation Stream sensor.
#         PRD §4.1 specifies capture of both directions.
#         PRD §6.4 specifies temporal synapse creation.
#   How:  Ring buffer (_recent_signals) holds signals within the
#         temporal window. On each feed(), the signal is recorded to
#         the substrate via record_outcome(), then temporal bindings
#         are created to recent signals via _bind_temporal(). Expired
#         signals are pruned from the window on each call.
# -------------------
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from core.signals import ConversationMeta, WorkflowSignal
from sensors.base import SensorBase

logger = logging.getLogger("praxis.conversation")


@dataclass
class TemporalSignal:
    """A signal with its embedding, stored in the temporal window."""
    signal: WorkflowSignal
    embedding: np.ndarray
    node_target_id: str  # The target_id used when recording to substrate


class ConversationSensor(SensorBase):
    """Pheromone 1: Conversation Stream (PRD §4.1).

    Observes every message in both directions — what the human says
    AND what the AI responds. The pair (human utterance + AI response +
    temporal position) is the signal unit.

    The sensor:
    1. Creates WorkflowSignal with ConversationMeta on each feed()
    2. Records the signal to the substrate (raw embedding — Law 7)
    3. Creates temporal synapses to recent signals within the window
    4. Prunes expired signals from the temporal window
    5. Logs signals to JSONL for session replay / cold start

    The temporal window creates associative structure: messages that
    occur close together in time develop synaptic connections. The
    substrate learns which message patterns co-occur and strengthens
    their associations via Hebbian learning on each step().
    """

    SENSOR_TYPE = "conversation"

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

        # Temporal window (PRD §13: default 300 seconds)
        self._temporal_window: float = config.get(
            "temporal_window_seconds", 300.0
        )
        self._capture_both: bool = config.get(
            "capture_both_directions", True
        )
        self._min_length: int = config.get("min_message_length", 10)

        # Recent signals within the temporal window
        self._recent: Deque[TemporalSignal] = deque()

        # Pending signals not yet collected by the hook
        self._buffer: List[WorkflowSignal] = []

        # Counters
        self._total_captured = 0
        self._total_bindings = 0
        self._total_skipped = 0

        # Signal log path (set by hook after init)
        self._log_path: Optional[Path] = None

    def feed(
        self,
        text: str,
        embedding: np.ndarray,
        direction: str = "human",
        session_id: str = "",
        speaker_id: str = "default",
        modality: str = "text",
        layer_depth: str = "implementation",
    ) -> Optional[WorkflowSignal]:
        """Capture a conversation turn as a WorkflowSignal.

        This is the primary ingestion method. Called by the hook on each
        on_message(). The embedding is provided by the hook (via _embed).

        Args:
            text: The message content
            embedding: 384d normalized embedding from the hook
            direction: 'human' or 'ai'
            session_id: Current session identifier
            speaker_id: Speaker identifier for multi-speaker
            modality: 'text' or 'voice'
            layer_depth: Intent decomposition layer

        Returns:
            WorkflowSignal if captured, None if skipped.
        """
        # Direction gate
        if direction == "ai" and not self._capture_both:
            self._total_skipped += 1
            return None

        self._total_captured += 1

        # Build metadata
        meta = ConversationMeta(
            direction=direction,
            speaker_id=speaker_id,
            modality=modality,
            turn_index=self._total_captured,
            message_length=len(text),
        )

        # Create signal
        signal = WorkflowSignal(
            pheromone="conversation",
            timestamp=time.time(),
            embedding=embedding,
            metadata=meta.__dict__,
            session_id=session_id,
            layer_depth=layer_depth,
        )

        # Target ID for substrate node — unique per signal
        node_target_id = f"conv:{signal.signal_id}"

        # Prune expired signals from temporal window
        self._prune_window(signal.timestamp)

        # Store in temporal window for future bindings
        ts = TemporalSignal(
            signal=signal,
            embedding=embedding,
            node_target_id=node_target_id,
        )
        self._recent.append(ts)

        # Buffer for collect_signals()
        self._buffer.append(signal)

        # Log to JSONL if path is set
        self._log_signal(signal)

        return signal

    def get_temporal_neighbors(
        self, current_timestamp: float
    ) -> List[TemporalSignal]:
        """Return all signals within the temporal window of the given time.

        Used by the hook to create temporal synapses between the new
        signal and its recent neighbors.
        """
        self._prune_window(current_timestamp)
        return list(self._recent)

    def get_recent_target_ids(
        self, exclude_signal_id: str, current_timestamp: float
    ) -> List[Tuple[str, float]]:
        """Return (target_id, age_seconds) for recent signals in the window.

        Excludes the signal with the given ID (the new one).
        Used by the hook to create temporal bindings.
        """
        self._prune_window(current_timestamp)
        result = []
        for ts in self._recent:
            if ts.signal.signal_id == exclude_signal_id:
                continue
            age = current_timestamp - ts.signal.timestamp
            result.append((ts.node_target_id, age))
        return result

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
            "total_bindings": self._total_bindings,
            "total_skipped": self._total_skipped,
            "temporal_window_size": len(self._recent),
            "buffer_size": len(self._buffer),
            "temporal_window_seconds": self._temporal_window,
        }

    def set_log_path(self, path: Path) -> None:
        """Set the JSONL log file path for signal persistence."""
        self._log_path = path

    def increment_bindings(self, count: int = 1) -> None:
        """Track temporal bindings created by the hook."""
        self._total_bindings += count

    # -----------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------

    def _prune_window(self, current_time: float) -> None:
        """Remove signals older than the temporal window."""
        cutoff = current_time - self._temporal_window
        while self._recent and self._recent[0].signal.timestamp < cutoff:
            self._recent.popleft()

    def _log_signal(self, signal: WorkflowSignal) -> None:
        """Append signal to JSONL log for session replay / cold start."""
        if self._log_path is None:
            return
        try:
            entry = signal.to_dict()
            with open(self._log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as exc:
            logger.debug("Signal log write failed: %s", exc)
