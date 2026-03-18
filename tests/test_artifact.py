"""Tests for Praxis Artifact Sensor — Phase 5."""

import sys
import os
import time

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vendored = os.path.join(repo_root, "vendored")
for p in [repo_root, vendored]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from sensors.artifact import ArtifactSensor


def _emb(seed=42):
    rng = np.random.RandomState(seed)
    v = rng.randn(384).astype(np.float32)
    return v / np.linalg.norm(v)


def test_register_creates_artifact():
    sensor = ArtifactSensor(config={})
    signal = sensor.register_artifact("doc-1", _emb(1), "prd", "abc123", "s1")
    assert signal.pheromone == "artifact"
    assert signal.metadata["event_type"] == "create"
    assert signal.metadata["artifact_id"] == "doc-1"
    assert sensor.get_artifact("doc-1") is not None


def test_register_existing_same_hash_is_reference():
    sensor = ArtifactSensor(config={})
    sensor.register_artifact("doc-1", _emb(1), "prd", "abc", "s1")
    signal = sensor.register_artifact("doc-1", _emb(2), "prd", "abc", "s1")
    assert signal.metadata["event_type"] == "reference"
    assert sensor.get_artifact("doc-1").reference_count == 1


def test_register_existing_new_hash_is_modify():
    sensor = ArtifactSensor(config={})
    sensor.register_artifact("doc-1", _emb(1), "prd", "abc", "s1")
    signal = sensor.register_artifact("doc-1", _emb(2), "prd", "def", "s1")
    assert signal.metadata["event_type"] == "modify"


def test_record_reference():
    sensor = ArtifactSensor(config={})
    sensor.register_artifact("doc-1", _emb(1), "code", "abc", "s1")
    signal = sensor.record_reference("doc-1", _emb(2), "s1")
    assert signal.metadata["event_type"] == "reference"
    assert sensor.get_artifact("doc-1").reference_count == 1


def test_reference_unknown_creates():
    sensor = ArtifactSensor(config={})
    signal = sensor.record_reference("doc-new", _emb(1), "s1")
    assert signal.metadata["event_type"] == "reference"
    assert sensor.get_artifact("doc-new") is not None


def test_record_delete():
    sensor = ArtifactSensor(config={})
    sensor.register_artifact("doc-1", _emb(1), "config", "abc", "s1")
    signal = sensor.record_delete("doc-1", _emb(2), "s1")
    assert signal is not None
    assert signal.metadata["event_type"] == "delete"
    assert sensor.get_artifact("doc-1") is None


def test_delete_unknown_returns_none():
    sensor = ArtifactSensor(config={})
    assert sensor.record_delete("nope", _emb(1)) is None


def test_stale_detection():
    sensor = ArtifactSensor(config={"stale_threshold_days": 1})
    sensor.register_artifact("old-doc", _emb(1), "prd", "abc", "s1")
    # Backdate the reference time
    sensor._artifacts["old-doc"].last_referenced = time.time() - 200000
    stale = sensor.get_stale_artifacts()
    assert "old-doc" in stale


def test_collect_signals():
    sensor = ArtifactSensor(config={})
    sensor.register_artifact("d1", _emb(1), "code", "a", "s1")
    sensor.record_reference("d1", _emb(2), "s1")
    signals = sensor.collect_signals()
    assert len(signals) == 2
    assert sensor.collect_signals() == []


def test_stats():
    sensor = ArtifactSensor(config={})
    sensor.register_artifact("d1", _emb(1), "code", "a", "s1")
    sensor.record_reference("d1", _emb(2), "s1")
    stats = sensor.get_stats()
    assert stats["tracked_artifacts"] == 1
    assert stats["total_events"] == 2
    assert stats["events_by_type"]["create"] == 1
    assert stats["events_by_type"]["reference"] == 1


if __name__ == "__main__":
    tests = [
        test_register_creates_artifact,
        test_register_existing_same_hash_is_reference,
        test_register_existing_new_hash_is_modify,
        test_record_reference,
        test_reference_unknown_creates,
        test_record_delete,
        test_delete_unknown_returns_none,
        test_stale_detection,
        test_collect_signals,
        test_stats,
    ]
    for t in tests:
        t()
        print(f"{t.__name__} PASSED")
    print(f"\nAll {len(tests)} artifact sensor tests passed.")
