"""
Live ingestion test — verify Praxis can digest real content end-to-end.

This test simulates the Converter's future workflow: feeding real content
through all three pheromone sensors and verifying the substrate learns.

Not a unit test — this exercises the full pipeline with real embeddings
(or hash fallback if no model available).

# ---- Changelog ----
# [2026-04-19] Claude Code (Sonnet 4.6) — Update to record_conversation() API.
#   What: Replace _module_on_message() calls with record_conversation().
#   Why:  _module_on_message is now a no-op (River-based arch). record_conversation()
#         is the new public standalone API — symmetric with record_artifact/outcome.
#   How:  s/_module_on_message(msg, hook._embed(msg))/record_conversation(msg)/
# [2026-03-18] Claude (Opus 4.6) — Initial creation.
#   What: End-to-end ingestion test with real content through all sensors.
#   Why:  Verify Praxis pipeline before building Converter.
#   How:  Instantiate PraxisHook, feed conversation, artifact, and outcome
#         signals, verify substrate records and CPS entries.
# -------------------
"""

import os
import sys
import tempfile

# Ensure vendored and project root are on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "vendored"))


def test_live_conversation_ingestion():
    """Feed a multi-turn conversation through the pipeline."""
    from core.praxis_hook import PraxisHook

    hook = PraxisHook()

    # Simulate a user describing software they're dropping in
    messages = [
        "I'm bringing in Mousepad — it's XFCE's lightweight text editor.",
        "I mostly use it for quick edits, config files, and reading logs.",
        "It has syntax highlighting, search and replace, and tabs.",
        "The source is about 15,000 lines of C.",
    ]

    results = []
    for msg in messages:
        result = hook.record_conversation(msg)
        results.append(result)

    # All messages should be captured
    captured = [r for r in results if r.get("status") == "captured"]
    assert len(captured) >= 3, f"Expected at least 3 captured, got {len(captured)}"

    # Temporal bindings should exist (messages within 300s window)
    total_bindings = sum(r.get("temporal_bindings", 0) for r in captured)
    assert total_bindings > 0, "Expected temporal bindings between conversation turns"

    # CPS should have entries
    assert hook._cps.count > 0, "CPS should have entries after conversation"

    # Signal count should match
    assert hook._signal_count >= len(captured), "Signal count should track captures"

    print(f"  Conversation: {len(captured)} signals, {total_bindings} temporal bindings, {hook._cps.count} CPS entries")


def test_live_artifact_ingestion():
    """Feed artifact lifecycle events through the pipeline."""
    from core.praxis_hook import PraxisHook

    hook = PraxisHook()

    # Register a source file as an artifact
    result_create = hook.record_artifact(
        artifact_id="mousepad/src/mousepad-window.c",
        content="Main window implementation for Mousepad text editor. Handles tab management, menu construction, toolbar, and the GTK application window lifecycle.",
        artifact_type="code",
        event_type="create",
        layer_depth="architecture",
    )
    assert result_create["status"] == "recorded"

    # Reference it from another context
    result_ref = hook.record_artifact(
        artifact_id="mousepad/src/mousepad-window.c",
        content="The window module handles tab management — each tab is a MousepadDocument with its own GtkTextBuffer.",
        artifact_type="code",
        event_type="reference",
        layer_depth="design",
    )
    assert result_ref["status"] == "recorded"

    # Register another file
    result_create2 = hook.record_artifact(
        artifact_id="mousepad/src/mousepad-document.c",
        content="Document buffer implementation. Wraps GtkTextBuffer with undo/redo, syntax highlighting via GtkSourceView, and encoding detection.",
        artifact_type="code",
        event_type="create",
        layer_depth="architecture",
    )
    assert result_create2["status"] == "recorded"

    assert hook._cps.count >= 3, "CPS should have entries for all artifact events"

    print(f"  Artifacts: 3 events recorded, {hook._cps.count} CPS entries")


def test_live_outcome_ingestion():
    """Feed outcome signals through the pipeline."""
    from core.praxis_hook import PraxisHook

    hook = PraxisHook()

    # Positive outcome — successful capability identification
    result_pos = hook.record_outcome(
        context="Successfully identified text editing capabilities: syntax highlighting, search/replace, tab management, encoding detection.",
        outcome_type="review",
        success=True,
        severity=0.7,
        layer_depth="architecture",
    )
    assert result_pos["status"] == "recorded"
    assert result_pos["reward_strength"] > 0, "Positive outcome should have positive reward"

    # Negative outcome — missed a capability
    result_neg = hook.record_outcome(
        context="Missed the print functionality — Mousepad has print support via GtkPrintOperation.",
        outcome_type="review",
        success=False,
        severity=0.4,
        layer_depth="design",
    )
    assert result_neg["status"] == "recorded"
    assert result_neg["reward_strength"] < 0, "Negative outcome should have negative reward"

    print(f"  Outcomes: positive reward={result_pos['reward_strength']:.4f}, negative reward={result_neg['reward_strength']:.4f}")


def test_live_full_digestion_simulation():
    """Simulate a complete software digestion through all three pheromones.

    This is what the Converter will do: conversation about the software,
    artifact registration of key files, and outcome validation of
    identified capabilities.
    """
    from core.praxis_hook import PraxisHook

    hook = PraxisHook()

    # --- Phase 0: User Introduction (Conversation Stream) ---
    phase0_msgs = [
        "I'm bringing in Mousepad, XFCE's text editor.",
        "It's lightweight — about 15K lines of C. Good for quick edits.",
    ]
    for msg in phase0_msgs:
        hook.record_conversation(msg)

    # --- Phase 1: Structural Survey (Artifact Stream) ---
    artifacts = [
        ("mousepad/Makefile.am", "Build system — autotools, GTK3 dependency, GtkSourceView optional", "config"),
        ("mousepad/src/main.c", "Entry point — GTK application init, command line parsing, single instance check", "code"),
        ("mousepad/src/mousepad-window.c", "Main window — tab bar, menu, toolbar, drag and drop support", "code"),
        ("mousepad/src/mousepad-document.c", "Document model — GtkTextBuffer wrapper, undo/redo, encoding", "code"),
        ("mousepad/src/mousepad-search-bar.c", "Search and replace — incremental search, regex support, highlight matches", "code"),
    ]
    for art_id, content, art_type in artifacts:
        hook.record_artifact(
            artifact_id=art_id,
            content=content,
            artifact_type=art_type,
            event_type="create",
            layer_depth="architecture",
        )

    # --- Phase 3: Capability Validation (Outcome Stream) ---
    capabilities = [
        ("Text editing with syntax highlighting via GtkSourceView", True, 0.8),
        ("Search and replace with regex support", True, 0.7),
        ("Multi-tab document management", True, 0.7),
        ("Encoding detection and conversion", True, 0.6),
        ("Print support via GtkPrintOperation", True, 0.5),
    ]
    for cap_desc, success, severity in capabilities:
        hook.record_outcome(
            context=f"Capability identified: {cap_desc}",
            outcome_type="review",
            success=success,
            severity=severity,
            layer_depth="design",
        )

    # --- Verify the substrate has learned ---
    stats = hook._module_stats()
    cps_count = hook._cps.count

    conv_count = stats["conversation_sensor"]["total_captured"]
    art_count = stats["artifact_sensor"]["total_events"]
    out_count = stats["outcome_sensor"]["total_outcomes"]

    print(f"\n  Full digestion simulation:")
    print(f"    Conversation signals: {conv_count}")
    print(f"    Artifact signals: {art_count}")
    print(f"    Outcome signals: {out_count}")
    print(f"    CPS entries: {cps_count}")
    print(f"    Total substrate signals: {stats['signal_count']}")

    # Verify all pheromones fired
    assert conv_count >= 2
    assert art_count >= 5
    assert out_count >= 5
    assert cps_count >= 12, f"Expected 12+ CPS entries, got {cps_count}"

    # --- Test recall: can the substrate find Mousepad-related content? ---
    query_embedding = hook._embed("text editor with syntax highlighting")
    retrieved = hook._cps.retrieve(query_embedding, top_k=5)
    assert len(retrieved) > 0, "Should retrieve related content from CPS"

    # Check that retrieved entries are actually related (retrieve returns (CPSEntry, score) tuples)
    found_relevant = False
    for entry, score in retrieved:
        if any(kw in entry.content.lower() for kw in ["mousepad", "text", "syntax", "editor", "highlight"]):
            found_relevant = True
            break

    assert found_relevant, f"Retrieved entries should be relevant to query. Got: {[(e.content[:50], s) for e, s in retrieved]}"

    print(f"    Recall test: queried 'text editor with syntax highlighting' → {len(retrieved)} results, relevance confirmed")
    for entry, score in retrieved[:3]:
        print(f"      [{score:.3f}] {entry.content[:80]}")


def test_grow_returns_correct_shape():
    """grow() returns dict with all required keys when Morphogenesis succeeds."""
    from unittest.mock import MagicMock, patch
    from core.praxis_hook import PraxisHook

    hook = PraxisHook()

    fake_result = MagicMock()
    fake_result.morpho_path = "/tmp/test_organism_42.morpho"
    fake_result.name = "test_organism"
    fake_result.behaviors = ["filter"]
    fake_result.fitness = 0.75
    fake_result.alive = True
    fake_result.fingerprint = "abc123"
    fake_result.calibrated = False

    fake_bridge = MagicMock()
    fake_bridge.grow.return_value = fake_result

    with patch("morphogenesis.creation_bridge.CreationBridge", return_value=fake_bridge):
        result = hook.grow("filter noise from sensor stream", seed=42)

    assert result["status"] == "grown"
    assert result["morpho_path"] == "/tmp/test_organism_42.morpho"
    assert result["name"] == "test_organism"
    assert result["behaviors"] == ["filter"]
    assert result["fitness"] == 0.75
    assert result["alive"] is True
    assert "fingerprint" in result
    assert "calibrated" in result


def test_grow_records_all_three_sensors():
    """grow() feeds conversation, artifact, and outcome sensors."""
    from unittest.mock import MagicMock, patch
    from core.praxis_hook import PraxisHook

    hook = PraxisHook()
    signal_before = hook._signal_count

    fake_result = MagicMock()
    fake_result.morpho_path = "/tmp/organism.morpho"
    fake_result.name = "org"
    fake_result.behaviors = ["passthrough"]
    fake_result.fitness = 0.5
    fake_result.alive = True
    fake_result.fingerprint = "fp"
    fake_result.calibrated = False

    fake_bridge = MagicMock()
    fake_bridge.grow.return_value = fake_result

    with patch("morphogenesis.creation_bridge.CreationBridge", return_value=fake_bridge):
        hook.grow("passthrough signal processor", seed=1)

    # conversation + artifact + outcome = at least 2 substrate signals (outcome may not
    # increment signal_count if short; artifact does; conversation does if long enough)
    assert hook._signal_count > signal_before


def test_grow_handles_import_error_gracefully():
    """grow() returns error dict when morphogenesis is not importable."""
    from unittest.mock import patch
    from core.praxis_hook import PraxisHook
    import builtins

    hook = PraxisHook()
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "morphogenesis.creation_bridge":
            raise ImportError("morphogenesis not installed")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        result = hook.grow("should fail gracefully")

    assert result["status"] == "error"
    assert "error" in result


def test_grow_handles_creation_error():
    """grow() returns error dict and records failure outcome when grow() raises."""
    from unittest.mock import MagicMock, patch
    from core.praxis_hook import PraxisHook

    hook = PraxisHook()

    fake_bridge = MagicMock()
    fake_bridge.grow.side_effect = ValueError("organism died during growth")

    with patch("morphogenesis.creation_bridge.CreationBridge", return_value=fake_bridge):
        result = hook.grow("an ambiguous description that fails")

    assert result["status"] == "error"
    assert "died during growth" in result["error"]

    # Failure outcome should have been recorded (outcome sensor gets a signal)
    stats = hook._module_stats()
    assert stats["outcome_sensor"]["total_outcomes"] >= 1


if __name__ == "__main__":
    print("Testing live ingestion pipeline...\n")
    test_live_conversation_ingestion()
    test_live_artifact_ingestion()
    test_live_outcome_ingestion()
    test_live_full_digestion_simulation()
    print("\nAll live ingestion tests passed.")
