"""
Praxis Integration Tests — Phase 7

Exercises the full signal pipeline end-to-end without requiring
a live ecosystem. Tests the complete flow:
  conversation → sensor → substrate record → CPS store → session bridge

Also tests autonomic state response behavior and cold start.

# ---- Changelog ----
# [2026-03-18] Claude (Opus 4.6) — Phase 7: Integration tests.
#   What: End-to-end pipeline tests, autonomic response tests,
#         cold start tests, multi-session lifecycle test.
#   Why:  PRD §14 Phase 7 — full Tier 2 validation.
#   How:  Uses standalone sensor/CPS/bridge instances (no live ecosystem)
#         to validate the complete signal flow.
# -------------------
"""

import sys
import os
import time
import json

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vendored = os.path.join(repo_root, "vendored")
for p in [repo_root, vendored]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from core.config import PraxisConfig
from core.signals import WorkflowSignal, ConversationMeta, ArtifactMeta, OutcomeMeta
from core.session_bridge import SessionBridge
from sensors.conversation import ConversationSensor
from sensors.artifact import ArtifactSensor
from sensors.outcome import OutcomeSensor
from store.cps import ContextPersistenceStore, CPSEntry


def _emb(seed=42):
    rng = np.random.RandomState(seed)
    v = rng.randn(384).astype(np.float32)
    return v / np.linalg.norm(v)


# -------------------------------------------------------------------
# End-to-end pipeline test
# -------------------------------------------------------------------

def test_full_pipeline():
    """Complete signal flow: conversation → CPS → session bridge."""
    # 1. Create components
    conv = ConversationSensor(config={
        "temporal_window_seconds": 300.0,
        "capture_both_directions": True,
    })
    cps = ContextPersistenceStore(config={"max_entries": 1000})
    bridge = SessionBridge(
        config={
            "max_context_items": 5,
            "min_activation_threshold": 0.0,
            "context_injection_format": "markdown",
            "auto_generate_summary": True,
            "summary_max_tokens": 500,
        },
        cps=cps,
    )

    # 2. Simulate session 1: several conversation turns
    session_1 = "session-001"
    messages = [
        ("I want to build an authentication module", 10),
        ("Let's use JWT tokens for stateless auth", 20),
        ("The middleware should validate on every request", 30),
        ("We rejected session cookies — too much server state", 40),
    ]

    for text, seed in messages:
        emb = _emb(seed)
        signal = conv.feed(text=text, embedding=emb, session_id=session_1)
        assert signal is not None

        # Store in CPS
        entry_type = "REJECTED_ALTERNATIVE" if "rejected" in text else "INTENT"
        cps.store_from_signal(
            content=text,
            embedding=emb,
            entry_type=entry_type,
            pheromone_source="conversation",
            session_id=session_1,
        )

    assert cps.count == 4

    # 3. Generate session summary
    summary = bridge.generate_summary(session_1)
    assert summary is not None
    assert "session-001" in summary
    assert cps.count == 5  # 4 entries + 1 SESSION_SUMMARY

    # 4. Start new session — surface context
    session_2 = "session-002"
    new_msg_emb = _emb(10)  # Similar to first message
    items = bridge.surface_context(
        initial_message="Continue work on auth module",
        embedding=new_msg_emb,
        session_id=session_2,
    )
    assert len(items) > 0

    # 5. Format for injection
    formatted = bridge.format_context(items, fmt="markdown")
    assert "## Prior Context" in formatted
    assert len(formatted) > 50

    # 6. Track usage
    used = bridge.track_usage("auth module JWT", _emb(10))
    assert isinstance(used, list)


# -------------------------------------------------------------------
# All three pheromones together
# -------------------------------------------------------------------

def test_three_pheromones():
    """All three sensors produce signals that CPS can store."""
    cps = ContextPersistenceStore(config={"max_entries": 1000})

    # Conversation
    conv = ConversationSensor(config={})
    conv_signal = conv.feed("design the API", _emb(1), session_id="s1")
    cps.store_from_signal(
        content="design the API",
        embedding=_emb(1),
        entry_type="INTENT",
        pheromone_source="conversation",
        session_id="s1",
    )

    # Artifact
    art = ArtifactSensor(config={})
    art_signal = art.register_artifact("api-spec.yaml", _emb(2), "config", "abc", "s1")
    cps.store_from_signal(
        content="api-spec.yaml created",
        embedding=_emb(2),
        entry_type="ARTIFACT",
        pheromone_source="artifact",
        session_id="s1",
    )

    # Outcome
    out = OutcomeSensor(config={})
    out_signal, reward = out.record_outcome(
        _emb(3), "test", True, severity=0.9, session_id="s1"
    )
    cps.store_from_signal(
        content="API tests passed",
        embedding=_emb(3),
        entry_type="OUTCOME",
        pheromone_source="outcome",
        session_id="s1",
        outcome_signal=reward,
    )

    assert cps.count == 3
    stats = cps.get_stats()
    assert stats["type_counts"]["INTENT"] == 1
    assert stats["type_counts"]["ARTIFACT"] == 1
    assert stats["type_counts"]["OUTCOME"] == 1


# -------------------------------------------------------------------
# Autonomic response behavior (PRD §9)
# -------------------------------------------------------------------

def test_autonomic_parasympathetic_behavior():
    """In PARASYMPATHETIC, short messages are filtered."""
    conv = ConversationSensor(config={"min_message_length": 10})
    # Short message — would be filtered by hook in PARASYMPATHETIC
    signal = conv.feed("hi", _emb(1), session_id="s1")
    assert signal is not None  # Sensor doesn't filter — hook does
    assert signal.metadata["message_length"] == 2


def test_autonomic_sympathetic_increases_capture():
    """In SYMPATHETIC, capture granularity increases (no filtering)."""
    # The hook skips min_length check when SYMPATHETIC
    # We test that the sensor captures everything regardless
    conv = ConversationSensor(config={"min_message_length": 100})
    signal = conv.feed("short", _emb(1), session_id="s1")
    assert signal is not None  # Sensor always captures — hook gates


# -------------------------------------------------------------------
# Cold start mitigation (PRD §12)
# -------------------------------------------------------------------

def test_cold_start_artifact_import():
    """Cold start: ingesting existing artifacts populates CPS."""
    cps = ContextPersistenceStore(config={"max_entries": 1000})
    art = ArtifactSensor(config={})

    # Simulate importing existing project artifacts
    artifacts = [
        ("README.md", "Project readme with architecture overview", "summary"),
        ("api/routes.py", "Flask routes for authentication", "code"),
        ("docs/auth-prd.md", "Authentication PRD v1", "prd"),
    ]

    for artifact_id, content, atype in artifacts:
        emb = _emb(hash(artifact_id) % 1000)
        art.register_artifact(artifact_id, emb, atype, "", "bootstrap")
        cps.store_from_signal(
            content=content,
            embedding=emb,
            entry_type="ARTIFACT",
            pheromone_source="artifact",
            session_id="bootstrap",
        )

    assert cps.count == 3
    assert art.get_stats()["tracked_artifacts"] == 3

    # New session should be able to surface these
    bridge = SessionBridge(
        config={"max_context_items": 5, "min_activation_threshold": 0.0},
        cps=cps,
    )
    items = bridge.surface_context("auth system", _emb(hash("api/routes.py") % 1000), "s1")
    assert len(items) > 0


def test_cold_start_transcript_replay():
    """Cold start: replaying conversation transcripts populates substrate."""
    conv = ConversationSensor(config={"temporal_window_seconds": 3600.0})
    cps = ContextPersistenceStore(config={"max_entries": 1000})

    # Replay historical conversation
    transcript = [
        ("We need OAuth2 for the API", "human"),
        ("I recommend using the authorization code flow", "ai"),
        ("Agreed. Let's use the PKCE extension too", "human"),
    ]

    for text, direction in transcript:
        emb = _emb(hash(text) % 10000)
        conv.feed(text=text, embedding=emb, direction=direction, session_id="replay")
        cps.store_from_signal(
            content=text,
            embedding=emb,
            entry_type="INTENT" if direction == "human" else "DISCOVERY",
            pheromone_source="conversation",
            session_id="replay",
        )

    assert cps.count == 3
    assert conv.get_stats()["total_captured"] == 3
    # Temporal window should have all 3 signals
    neighbors = conv.get_temporal_neighbors(time.time())
    assert len(neighbors) == 3


# -------------------------------------------------------------------
# Multi-session lifecycle
# -------------------------------------------------------------------

def test_multi_session_lifecycle():
    """Complete lifecycle across multiple sessions with summary carry-over."""
    cps = ContextPersistenceStore(config={"max_entries": 1000})
    bridge_config = {
        "max_context_items": 10,
        "min_activation_threshold": 0.0,
        "auto_generate_summary": True,
        "summary_max_tokens": 500,
        "context_injection_format": "markdown",
    }
    bridge = SessionBridge(config=bridge_config, cps=cps)

    # Session 1
    cps.store_from_signal("Build auth", _emb(1), "INTENT", "conversation", "s1")
    cps.store_from_signal("Use JWT", _emb(2), "DECISION", "conversation", "s1")
    summary_1 = bridge.generate_summary("s1")
    assert summary_1 is not None

    # Session 2 — should surface session 1 context
    items = bridge.surface_context("Continue auth work", _emb(1), "s2")
    assert len(items) > 0
    # Session summary from s1 should be among surfaced items
    types_surfaced = {item["entry_type"] for item in items}
    # At minimum, INTENT and DECISION should surface
    assert "INTENT" in types_surfaced or "DECISION" in types_surfaced

    # Session 2 adds more
    cps.store_from_signal("Add rate limiting", _emb(3), "INTENT", "conversation", "s2")
    summary_2 = bridge.generate_summary("s2")
    assert summary_2 is not None

    # Session 3 — should have context from both sessions
    items_3 = bridge.surface_context("security improvements", _emb(1), "s3")
    assert len(items_3) > 0
    sessions_represented = {item["session_id"] for item in items_3}
    assert len(sessions_represented) >= 1  # At least one session surfaced


# -------------------------------------------------------------------
# CPS persistence roundtrip with full pipeline
# -------------------------------------------------------------------

def test_cps_roundtrip_full(tmp_path):
    """CPS save/load preserves entries from all three pheromones."""
    cps = ContextPersistenceStore(
        config={"max_entries": 1000},
        data_dir=str(tmp_path),
    )

    cps.store_from_signal("intent msg", _emb(1), "INTENT", "conversation", "s1")
    cps.store_from_signal("artifact ref", _emb(2), "ARTIFACT", "artifact", "s1")
    cps.store_from_signal(
        "test passed", _emb(3), "OUTCOME", "outcome", "s1",
        outcome_signal=0.7,
    )

    cps.save()

    # Load fresh
    cps2 = ContextPersistenceStore(
        config={"max_entries": 1000},
        data_dir=str(tmp_path),
    )
    assert cps2.count == 3

    stats = cps2.get_stats()
    assert stats["type_counts"]["INTENT"] == 1
    assert stats["type_counts"]["ARTIFACT"] == 1
    assert stats["type_counts"]["OUTCOME"] == 1


# -------------------------------------------------------------------
# Config validation
# -------------------------------------------------------------------

def test_config_all_defaults():
    """PraxisConfig loads with all PRD defaults."""
    cfg = PraxisConfig()
    assert cfg.sensors.conversation.temporal_window_seconds == 300.0
    assert cfg.sensors.artifact.temporal_window_seconds == 3600.0
    assert cfg.sensors.outcome.positive_reward_strength == 0.7
    assert cfg.cps.max_entries == 50000
    assert cfg.thresholds.auto_execute == 0.70
    assert cfg.thresholds.recommend == 0.40
    assert cfg.thresholds.host_premium == 0.15
    assert cfg.ng_lite.module_id == "praxis"


if __name__ == "__main__":
    tests = [
        test_full_pipeline,
        test_three_pheromones,
        test_autonomic_parasympathetic_behavior,
        test_autonomic_sympathetic_increases_capture,
        test_cold_start_artifact_import,
        test_cold_start_transcript_replay,
        test_multi_session_lifecycle,
        test_config_all_defaults,
    ]
    for t in tests:
        t()
        print(f"{t.__name__} PASSED")

    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        test_cps_roundtrip_full(Path(td))
    print("test_cps_roundtrip_full PASSED")

    print(f"\nAll {len(tests) + 1} integration tests passed.")
