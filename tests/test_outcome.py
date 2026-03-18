"""Tests for Praxis Outcome Sensor — Phase 6."""

import sys
import os

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vendored = os.path.join(repo_root, "vendored")
for p in [repo_root, vendored]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from sensors.outcome import OutcomeSensor


def _emb(seed=42):
    rng = np.random.RandomState(seed)
    v = rng.randn(384).astype(np.float32)
    return v / np.linalg.norm(v)


def test_record_success():
    sensor = OutcomeSensor(config={
        "positive_reward_strength": 0.7,
        "negative_reward_strength": -0.5,
    })
    signal, strength = sensor.record_outcome(
        embedding=_emb(1),
        outcome_type="build",
        success=True,
        severity=1.0,
        session_id="s1",
    )
    assert signal.pheromone == "outcome"
    assert signal.metadata["success"] is True
    assert signal.metadata["outcome_type"] == "build"
    assert strength == 0.7  # 0.7 * 1.0


def test_record_failure():
    sensor = OutcomeSensor(config={
        "positive_reward_strength": 0.7,
        "negative_reward_strength": -0.5,
    })
    signal, strength = sensor.record_outcome(
        embedding=_emb(1),
        outcome_type="test",
        success=False,
        severity=0.8,
        session_id="s1",
    )
    assert signal.metadata["success"] is False
    assert abs(strength - (-0.4)) < 0.001  # -0.5 * 0.8


def test_severity_scales_reward():
    sensor = OutcomeSensor(config={
        "positive_reward_strength": 1.0,
        "negative_reward_strength": -1.0,
    })
    _, strength_high = sensor.record_outcome(
        _emb(1), "build", True, severity=1.0, session_id="s1"
    )
    _, strength_low = sensor.record_outcome(
        _emb(2), "build", True, severity=0.2, session_id="s1"
    )
    assert strength_high == 1.0
    assert abs(strength_low - 0.2) < 0.001


def test_related_intent_ids():
    sensor = OutcomeSensor(config={})
    signal, _ = sensor.record_outcome(
        _emb(1), "test", True,
        related_intent_ids=["intent-abc", "intent-def"],
        session_id="s1",
    )
    assert signal.metadata["related_intent_ids"] == ["intent-abc", "intent-def"]


def test_collect_signals():
    sensor = OutcomeSensor(config={})
    sensor.record_outcome(_emb(1), "build", True, session_id="s1")
    sensor.record_outcome(_emb(2), "test", False, session_id="s1")
    signals = sensor.collect_signals()
    assert len(signals) == 2
    assert sensor.collect_signals() == []


def test_stats():
    sensor = OutcomeSensor(config={
        "positive_reward_strength": 0.7,
        "negative_reward_strength": -0.5,
    })
    sensor.record_outcome(_emb(1), "build", True, session_id="s1")
    sensor.record_outcome(_emb(2), "test", True, session_id="s1")
    sensor.record_outcome(_emb(3), "test", False, session_id="s1")
    stats = sensor.get_stats()
    assert stats["total_outcomes"] == 3
    assert stats["total_successes"] == 2
    assert stats["total_failures"] == 1
    assert abs(stats["success_rate"] - 0.6667) < 0.001


def test_default_strengths():
    sensor = OutcomeSensor(config={})
    assert sensor._positive_strength == 0.7
    assert sensor._negative_strength == -0.5


if __name__ == "__main__":
    tests = [
        test_record_success,
        test_record_failure,
        test_severity_scales_reward,
        test_related_intent_ids,
        test_collect_signals,
        test_stats,
        test_default_strengths,
    ]
    for t in tests:
        t()
        print(f"{t.__name__} PASSED")
    print(f"\nAll {len(tests)} outcome sensor tests passed.")
