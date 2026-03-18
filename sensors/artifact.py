"""
Praxis Artifact Sensor — Pheromone 2

Monitors document lifecycle events (create, modify, reference, delete)
and tracks artifact salience through reference counting. Artifacts that
are frequently referenced develop strong substrate connections. Artifacts
that go unreferenced decay naturally through weight pruning.

Canonical source: https://github.com/greatnorthernfishguy-hub/Praxis
License: AGPL-3.0

# ---- Changelog ----
# [2026-03-18] Claude (Opus 4.6) — Initial creation (Phase 1).
#   What: ArtifactSensor stub with interface definition.
#   Why:  PRD §14 Phase 1.
#   How:  Extends SensorBase. collect_signals() returns empty list.
# [2026-03-18] Claude (Opus 4.6) — Phase 5 full implementation.
#   What: Artifact lifecycle tracking with register/reference/modify/
#         delete events. Reference counting for salience. Stale artifact
#         detection. Substrate recording via record_artifact().
#   Why:  PRD §4.2 — artifacts carry lifecycle signal (not just content).
#         PRD §14 Phase 5.
#   How:  In-memory artifact registry keyed by artifact_id. Each lifecycle
#         event creates a WorkflowSignal with ArtifactMeta. Reference
#         counting drives salience — high-reference artifacts surface
#         more readily. Stale detection flags unreferenced artifacts.
# -------------------
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from core.signals import ArtifactMeta, WorkflowSignal
from sensors.base import SensorBase

logger = logging.getLogger("praxis.artifact")


@dataclass
class TrackedArtifact:
    """Internal record for a tracked artifact."""
    artifact_id: str
    artifact_type: str = "other"       # 'prd' | 'code' | 'config' | 'summary' | 'other'
    content_hash: str = ""
    reference_count: int = 0
    last_referenced: float = 0.0
    created_at: float = 0.0
    last_modified: float = 0.0
    last_embedding: Optional[np.ndarray] = field(default=None, repr=False)


class ArtifactSensor(SensorBase):
    """Pheromone 2: Artifact Stream (PRD §4.2).

    Tracks persistent artifacts (documents, code, PRDs, configs) through
    their lifecycle. Each event produces a WorkflowSignal:

    - create:    New artifact registered
    - modify:    Content changed (detected via content hash)
    - reference: Artifact mentioned or used in conversation
    - delete:    Artifact removed from tracking

    Reference counting drives salience — frequently-referenced artifacts
    develop stronger substrate connections. The sensor does not embed
    content itself; embeddings are provided by the caller (hook).
    """

    SENSOR_TYPE = "artifact"

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

        self._temporal_window: float = config.get(
            "temporal_window_seconds", 3600.0
        )
        self._stale_threshold_days: int = config.get(
            "stale_threshold_days", 30
        )

        # Artifact registry
        self._artifacts: Dict[str, TrackedArtifact] = {}

        # Signal buffer
        self._buffer: List[WorkflowSignal] = []

        # Counters
        self._total_events = 0
        self._events_by_type: Dict[str, int] = {}

        # Log path (set by hook)
        self._log_path: Optional[Path] = None

    def register_artifact(
        self,
        artifact_id: str,
        embedding: np.ndarray,
        artifact_type: str = "other",
        content_hash: str = "",
        session_id: str = "",
        layer_depth: str = "implementation",
    ) -> WorkflowSignal:
        """Register a new artifact or update an existing one.

        If the artifact already exists and content_hash differs,
        this becomes a 'modify' event. Otherwise it's a 'create'.
        """
        now = time.time()
        existing = self._artifacts.get(artifact_id)

        if existing is not None:
            if content_hash and content_hash != existing.content_hash:
                event_type = "modify"
                existing.content_hash = content_hash
                existing.last_modified = now
                existing.last_embedding = embedding
            else:
                # No change — treat as reference
                return self.record_reference(
                    artifact_id, embedding, session_id, layer_depth
                )
        else:
            event_type = "create"
            self._artifacts[artifact_id] = TrackedArtifact(
                artifact_id=artifact_id,
                artifact_type=artifact_type,
                content_hash=content_hash,
                created_at=now,
                last_modified=now,
                last_referenced=now,
                last_embedding=embedding,
            )

        return self._create_signal(
            artifact_id=artifact_id,
            event_type=event_type,
            embedding=embedding,
            artifact_type=artifact_type,
            content_hash=content_hash,
            session_id=session_id,
            layer_depth=layer_depth,
        )

    def record_reference(
        self,
        artifact_id: str,
        embedding: np.ndarray,
        session_id: str = "",
        layer_depth: str = "implementation",
    ) -> WorkflowSignal:
        """Record that an artifact was referenced in conversation."""
        now = time.time()
        artifact = self._artifacts.get(artifact_id)

        if artifact is not None:
            artifact.reference_count += 1
            artifact.last_referenced = now
            artifact.last_embedding = embedding
            ref_count = artifact.reference_count
            artifact_type = artifact.artifact_type
            content_hash = artifact.content_hash
        else:
            # Reference to unknown artifact — register it
            self._artifacts[artifact_id] = TrackedArtifact(
                artifact_id=artifact_id,
                reference_count=1,
                last_referenced=now,
                created_at=now,
                last_embedding=embedding,
            )
            ref_count = 1
            artifact_type = "other"
            content_hash = ""

        return self._create_signal(
            artifact_id=artifact_id,
            event_type="reference",
            embedding=embedding,
            artifact_type=artifact_type,
            content_hash=content_hash,
            reference_count=ref_count,
            session_id=session_id,
            layer_depth=layer_depth,
        )

    def record_delete(
        self,
        artifact_id: str,
        embedding: np.ndarray,
        session_id: str = "",
    ) -> Optional[WorkflowSignal]:
        """Record artifact deletion."""
        artifact = self._artifacts.pop(artifact_id, None)
        if artifact is None:
            return None

        return self._create_signal(
            artifact_id=artifact_id,
            event_type="delete",
            embedding=embedding,
            artifact_type=artifact.artifact_type,
            content_hash=artifact.content_hash,
            reference_count=artifact.reference_count,
            session_id=session_id,
        )

    def get_stale_artifacts(self) -> List[str]:
        """Return artifact_ids unreferenced for longer than stale threshold."""
        cutoff = time.time() - (self._stale_threshold_days * 86400)
        return [
            a.artifact_id for a in self._artifacts.values()
            if a.last_referenced < cutoff
        ]

    def get_artifact(self, artifact_id: str) -> Optional[TrackedArtifact]:
        """Get tracking info for an artifact."""
        return self._artifacts.get(artifact_id)

    def collect_signals(self) -> List[WorkflowSignal]:
        """Return buffered signals and clear."""
        signals = self._buffer[:]
        self._buffer.clear()
        return signals

    def get_stats(self) -> Dict[str, Any]:
        return {
            "sensor_type": self.SENSOR_TYPE,
            "enabled": True,
            "tracked_artifacts": len(self._artifacts),
            "total_events": self._total_events,
            "events_by_type": dict(self._events_by_type),
            "buffer_size": len(self._buffer),
        }

    def set_log_path(self, path: Path) -> None:
        self._log_path = path

    # -----------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------

    def _create_signal(
        self,
        artifact_id: str,
        event_type: str,
        embedding: np.ndarray,
        artifact_type: str = "other",
        content_hash: str = "",
        reference_count: int = 0,
        session_id: str = "",
        layer_depth: str = "implementation",
    ) -> WorkflowSignal:
        """Create a WorkflowSignal for an artifact lifecycle event."""
        self._total_events += 1
        self._events_by_type[event_type] = (
            self._events_by_type.get(event_type, 0) + 1
        )

        meta = ArtifactMeta(
            artifact_id=artifact_id,
            event_type=event_type,
            content_hash=content_hash,
            artifact_type=artifact_type,
            reference_count=reference_count,
        )

        signal = WorkflowSignal(
            pheromone="artifact",
            timestamp=time.time(),
            embedding=embedding,
            metadata=meta.__dict__,
            session_id=session_id,
            layer_depth=layer_depth,
        )

        self._buffer.append(signal)
        self._log_signal(signal)
        return signal

    def _log_signal(self, signal: WorkflowSignal) -> None:
        if self._log_path is None:
            return
        try:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(signal.to_dict()) + "\n")
        except Exception:
            pass
