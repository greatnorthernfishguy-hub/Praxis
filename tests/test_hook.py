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


def _make_embedding(seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    vec = rng.randn(384).astype(np.float32)
    return vec / np.linalg.norm(vec)


def test_conversation_sensor_feed():
    """ConversationSensor.feed() creates a signal with embedding."""
    sensor = ConversationSensor(config={})
    emb = _make_embedding(1)
    signal = sensor.feed(
        text="Hello, I want to build a module",
        embedding=emb,
        direction="human",
        session_id="test-session",
    )
    assert signal is not None
    assert signal.pheromone == "conversation"
    assert signal.session_id == "test-session"
    assert signal.metadata["direction"] == "human"
    assert signal.metadata["message_length"] == len("Hello, I want to build a module")


def test_conversation_sensor_collect():
    """collect_signals() returns buffered signals and clears."""
    sensor = ConversationSensor(config={})
    sensor.feed(text="message 1", embedding=_make_embedding(1), session_id="s1")
    sensor.feed(text="message 2", embedding=_make_embedding(2), session_id="s1")

    signals = sensor.collect_signals()
    assert len(signals) == 2

    signals2 = sensor.collect_signals()
    assert len(signals2) == 0


def test_conversation_sensor_stats():
    """Stats track total captured."""
    sensor = ConversationSensor(config={})
    sensor.feed(text="message 1", embedding=_make_embedding(1), session_id="s1")
    sensor.feed(text="message 2", embedding=_make_embedding(2), session_id="s1")

    stats = sensor.get_stats()
    assert stats["total_captured"] == 2
    assert stats["sensor_type"] == "conversation"


def test_artifact_sensor_stub():
    """ArtifactSensor stub returns empty."""
    sensor = ArtifactSensor(config={})
    assert sensor.collect_signals() == []
    assert sensor.SENSOR_TYPE == "artifact"


def test_outcome_sensor_stub():
    """OutcomeSensor stub returns empty."""
    sensor = OutcomeSensor(config={})
    assert sensor.collect_signals() == []
    assert sensor.SENSOR_TYPE == "outcome"


if __name__ == "__main__":
    test_conversation_sensor_feed()
    print("test_conversation_sensor_feed PASSED")

    test_conversation_sensor_collect()
    print("test_conversation_sensor_collect PASSED")

    test_conversation_sensor_stats()
    print("test_conversation_sensor_stats PASSED")

    test_artifact_sensor_stub()
    print("test_artifact_sensor_stub PASSED")

    test_outcome_sensor_stub()
    print("test_outcome_sensor_stub PASSED")

    print("\nAll hook/sensor tests passed.")
