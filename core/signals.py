"""
Praxis Signal Dataclasses — The Three Pheromones

Defines the data structures for the three sensor streams (conversation,
artifact, outcome) and the base WorkflowSignal that all sensors produce.

These are thin preprocessing structures — the retina that transforms raw
workflow observations into substrate-compatible signals. The intelligence
is in the substrate, not here.

Canonical source: https://github.com/greatnorthernfishguy-hub/Praxis
License: AGPL-3.0

# ---- Changelog ----
# [2026-03-18] Claude (Opus 4.6) — Initial creation (Phase 1).
#   What: WorkflowSignal, ConversationMeta, ArtifactMeta, OutcomeMeta
#         dataclasses per PRD §6.2.
#   Why:  PRD §6.2 specifies these as the signal format for all three
#         pheromone sensors.
#   How:  Dataclasses with fair starting values. UUID generation for
#         signal IDs. Numpy array for 384d embeddings.
# -------------------
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ConversationMeta:
    """Metadata for conversation stream signals (PRD §4.1)."""
    direction: str = "human"          # 'human' | 'ai'
    speaker_id: str = "default"       # For multi-speaker scenarios
    modality: str = "text"            # 'text' | 'voice'
    turn_index: int = 0               # Position in conversation
    message_length: int = 0           # Character count


@dataclass
class ArtifactMeta:
    """Metadata for artifact stream signals (PRD §4.2)."""
    artifact_id: str = ""             # Stable identifier for the artifact
    event_type: str = "reference"     # 'create' | 'modify' | 'reference' | 'delete'
    content_hash: str = ""            # For change detection
    artifact_type: str = "other"      # 'prd' | 'code' | 'config' | 'summary' | 'other'
    reference_count: int = 0          # Cumulative references to this artifact


@dataclass
class OutcomeMeta:
    """Metadata for outcome stream signals (PRD §4.3)."""
    outcome_type: str = "build"       # 'build' | 'test' | 'review' | 'deploy' | 'bug'
    success: bool = True              # Binary success/failure
    severity: float = 0.5             # 0.0 to 1.0
    related_intent_ids: List[str] = field(default_factory=list)


@dataclass
class WorkflowSignal:
    """Raw signal from any of the three pheromone sensors (PRD §6.2).

    This is the universal signal format that all sensors produce.
    The embedding enters the substrate with fair starting values —
    no pre-programmed classifications.
    """
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pheromone: str = ""               # 'conversation' | 'artifact' | 'outcome'
    timestamp: float = 0.0            # Unix timestamp with microsecond precision
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    metadata: Dict[str, Any] = field(default_factory=dict)
    session_id: str = ""              # Which session produced this signal
    layer_depth: str = "implementation"  # 'vision' | 'architecture' | 'design' | 'implementation'

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSONL logging (excludes embedding)."""
        return {
            "signal_id": self.signal_id,
            "pheromone": self.pheromone,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "layer_depth": self.layer_depth,
            "metadata": self.metadata,
        }
