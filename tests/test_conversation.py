"""Tests for Praxis Conversation Sensor — Phase 2."""

import sys
import os
import time

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vendored = os.path.join(repo_root, "vendored")
for p in [repo_root, vendored]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from sensors.conversation import ConversationSensor, TemporalSignal


def _make_embedding(seed: int = 42) -> np.ndarray:
    """Create a deterministic 384d normalized embedding."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(384).astype(np.float32)
    return vec / np.linalg.norm(vec)


def test_feed_with_embedding():
    """feed() creates signal with embedding stored."""
    sensor = ConversationSensor(config={
        "temporal_window_seconds": 300.0,
        "capture_both_directions": True,
        "min_message_length": 5,
    })
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
    assert signal.embedding is emb
    assert signal.metadata["direction"] == "human"
    assert signal.metadata["message_length"] == len("Hello, I want to build a module")


def test_feed_ai_direction_captured():
    """AI messages are captured when capture_both_directions is True."""
    sensor = ConversationSensor(config={
        "capture_both_directions": True,
    })
    emb = _make_embedding(2)
    signal = sensor.feed(
        text="I can help with that module",
        embedding=emb,
        direction="ai",
        session_id="test-session",
    )
    assert signal is not None
    assert signal.metadata["direction"] == "ai"


def test_feed_ai_direction_filtered():
    """AI messages are filtered when capture_both_directions is False."""
    sensor = ConversationSensor(config={
        "capture_both_directions": False,
    })
    emb = _make_embedding(3)
    signal = sensor.feed(
        text="I can help with that module",
        embedding=emb,
        direction="ai",
        session_id="test-session",
    )
    assert signal is None


def test_temporal_window():
    """Signals within the temporal window are accessible."""
    sensor = ConversationSensor(config={
        "temporal_window_seconds": 300.0,
    })

    emb1 = _make_embedding(10)
    emb2 = _make_embedding(20)
    emb3 = _make_embedding(30)

    s1 = sensor.feed(text="first message", embedding=emb1, session_id="s1")
    s2 = sensor.feed(text="second message", embedding=emb2, session_id="s1")
    s3 = sensor.feed(text="third message", embedding=emb3, session_id="s1")

    # All three should be in the window
    neighbors = sensor.get_temporal_neighbors(time.time())
    assert len(neighbors) == 3


def test_temporal_window_expiry():
    """Expired signals are pruned from the temporal window."""
    sensor = ConversationSensor(config={
        "temporal_window_seconds": 10.0,  # Very short window for testing
    })

    # Create a signal "20 seconds ago"
    emb1 = _make_embedding(10)
    s1 = sensor.feed(text="old message", embedding=emb1, session_id="s1")
    # Manually backdate it
    sensor._recent[0].signal.timestamp = time.time() - 20.0

    # Create a current signal
    emb2 = _make_embedding(20)
    s2 = sensor.feed(text="new message", embedding=emb2, session_id="s1")

    # Only the new signal should be in the window
    neighbors = sensor.get_temporal_neighbors(time.time())
    assert len(neighbors) == 1
    assert neighbors[0].signal.signal_id == s2.signal_id


def test_get_recent_target_ids():
    """get_recent_target_ids returns IDs excluding the specified signal."""
    sensor = ConversationSensor(config={
        "temporal_window_seconds": 300.0,
    })

    emb1 = _make_embedding(10)
    emb2 = _make_embedding(20)

    s1 = sensor.feed(text="first message", embedding=emb1, session_id="s1")
    s2 = sensor.feed(text="second message", embedding=emb2, session_id="s1")

    # Get neighbors excluding s2
    neighbors = sensor.get_recent_target_ids(
        exclude_signal_id=s2.signal_id,
        current_timestamp=time.time(),
    )
    assert len(neighbors) == 1
    target_id, age = neighbors[0]
    assert target_id == f"conv:{s1.signal_id}"
    assert age >= 0


def test_collect_signals():
    """collect_signals() returns buffered signals and clears."""
    sensor = ConversationSensor(config={})
    sensor.feed(text="message 1", embedding=_make_embedding(1), session_id="s1")
    sensor.feed(text="message 2", embedding=_make_embedding(2), session_id="s1")

    signals = sensor.collect_signals()
    assert len(signals) == 2

    # Buffer empty after collect
    signals2 = sensor.collect_signals()
    assert len(signals2) == 0


def test_stats():
    """Stats track captured, bindings, window size."""
    sensor = ConversationSensor(config={
        "temporal_window_seconds": 300.0,
    })
    sensor.feed(text="message 1", embedding=_make_embedding(1), session_id="s1")
    sensor.feed(text="message 2", embedding=_make_embedding(2), session_id="s1")
    sensor.increment_bindings(3)

    stats = sensor.get_stats()
    assert stats["total_captured"] == 2
    assert stats["total_bindings"] == 3
    assert stats["temporal_window_size"] == 2
    assert stats["sensor_type"] == "conversation"


def test_signal_logging(tmp_path):
    """Signals are logged to JSONL."""
    sensor = ConversationSensor(config={})
    log_file = tmp_path / "conversation.jsonl"
    sensor.set_log_path(log_file)

    sensor.feed(text="logged message", embedding=_make_embedding(1), session_id="s1")

    assert log_file.exists()
    content = log_file.read_text().strip()
    assert "logged message" not in content  # to_dict excludes raw text
    assert "conversation" in content


def test_multiple_speakers():
    """Different speaker IDs are preserved in metadata."""
    sensor = ConversationSensor(config={
        "capture_both_directions": True,
    })

    s1 = sensor.feed(
        text="idea from alice",
        embedding=_make_embedding(1),
        speaker_id="alice",
        session_id="brainstorm",
    )
    s2 = sensor.feed(
        text="response from bob",
        embedding=_make_embedding(2),
        speaker_id="bob",
        session_id="brainstorm",
    )

    assert s1.metadata["speaker_id"] == "alice"
    assert s2.metadata["speaker_id"] == "bob"


if __name__ == "__main__":
    tests = [
        test_feed_with_embedding,
        test_feed_ai_direction_captured,
        test_feed_ai_direction_filtered,
        test_temporal_window,
        test_temporal_window_expiry,
        test_get_recent_target_ids,
        test_collect_signals,
        test_stats,
        test_multiple_speakers,
    ]
    for t in tests:
        t()
        print(f"{t.__name__} PASSED")

    # Tests needing tmp_path
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        test_signal_logging(Path(td))
    print("test_signal_logging PASSED")

    print(f"\nAll {len(tests) + 1} conversation sensor tests passed.")
