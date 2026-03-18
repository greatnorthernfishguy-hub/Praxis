"""Tests for Praxis sensors — base class and all three pheromones."""

import sys
import os

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vendored = os.path.join(repo_root, "vendored")
for p in [repo_root, vendored]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from sensors.conversation import ConversationSensor
from sensors.artifact import ArtifactSensor
from sensors.outcome import OutcomeSensor


def _emb(seed=42):
    rng = np.random.RandomState(seed)
    v = rng.randn(384).astype(np.float32)
    return v / np.linalg.norm(v)


def test_conversation_sensor_basic():
    sensor = ConversationSensor(config={})
    signal = sensor.feed(text="hello world", embedding=_emb(1), session_id="s1")
    assert signal is not None
    assert sensor.collect_signals() != []


def test_artifact_sensor_basic():
    sensor = ArtifactSensor(config={})
    signal = sensor.register_artifact("doc", _emb(1), "prd", "hash", "s1")
    assert signal.pheromone == "artifact"
    assert len(sensor.collect_signals()) == 1


def test_outcome_sensor_basic():
    sensor = OutcomeSensor(config={})
    signal, strength = sensor.record_outcome(_emb(1), "build", True, session_id="s1")
    assert signal.pheromone == "outcome"
    assert strength > 0


def test_all_sensor_types():
    """All three sensor types have correct SENSOR_TYPE."""
    assert ConversationSensor(config={}).SENSOR_TYPE == "conversation"
    assert ArtifactSensor(config={}).SENSOR_TYPE == "artifact"
    assert OutcomeSensor(config={}).SENSOR_TYPE == "outcome"


if __name__ == "__main__":
    tests = [
        test_conversation_sensor_basic,
        test_artifact_sensor_basic,
        test_outcome_sensor_basic,
        test_all_sensor_types,
    ]
    for t in tests:
        t()
        print(f"{t.__name__} PASSED")
    print(f"\nAll {len(tests)} sensor integration tests passed.")
