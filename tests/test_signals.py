"""Tests for Praxis signal dataclasses."""

import sys
import os

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vendored = os.path.join(repo_root, "vendored")
for p in [repo_root, vendored]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from core.signals import (
    WorkflowSignal,
    ConversationMeta,
    ArtifactMeta,
    OutcomeMeta,
)


def test_workflow_signal_defaults():
    """WorkflowSignal has valid defaults."""
    sig = WorkflowSignal()
    assert sig.signal_id  # UUID generated
    assert sig.pheromone == ""
    assert sig.embedding is None
    assert sig.layer_depth == "implementation"


def test_workflow_signal_unique_ids():
    """Each signal gets a unique ID."""
    s1 = WorkflowSignal()
    s2 = WorkflowSignal()
    assert s1.signal_id != s2.signal_id


def test_workflow_signal_to_dict():
    """to_dict() serializes without embedding."""
    sig = WorkflowSignal(
        pheromone="conversation",
        timestamp=1234567890.0,
        session_id="test-session",
        embedding=np.zeros(384, dtype=np.float32),
    )
    d = sig.to_dict()
    assert d["pheromone"] == "conversation"
    assert d["timestamp"] == 1234567890.0
    assert d["session_id"] == "test-session"
    assert "embedding" not in d


def test_conversation_meta():
    """ConversationMeta has correct defaults."""
    meta = ConversationMeta()
    assert meta.direction == "human"
    assert meta.modality == "text"
    assert meta.turn_index == 0


def test_artifact_meta():
    """ArtifactMeta has correct defaults."""
    meta = ArtifactMeta()
    assert meta.event_type == "reference"
    assert meta.artifact_type == "other"


def test_outcome_meta():
    """OutcomeMeta has correct defaults."""
    meta = OutcomeMeta()
    assert meta.success is True
    assert meta.severity == 0.5
    assert meta.related_intent_ids == []


if __name__ == "__main__":
    test_workflow_signal_defaults()
    print("test_workflow_signal_defaults PASSED")

    test_workflow_signal_unique_ids()
    print("test_workflow_signal_unique_ids PASSED")

    test_workflow_signal_to_dict()
    print("test_workflow_signal_to_dict PASSED")

    test_conversation_meta()
    print("test_conversation_meta PASSED")

    test_artifact_meta()
    print("test_artifact_meta PASSED")

    test_outcome_meta()
    print("test_outcome_meta PASSED")

    print("\nAll signal tests passed.")
