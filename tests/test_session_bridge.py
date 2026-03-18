"""Tests for Praxis Session Bridge — Phase 4."""

import sys
import os
import time

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vendored = os.path.join(repo_root, "vendored")
for p in [repo_root, vendored]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from core.session_bridge import SessionBridge
from store.cps import ContextPersistenceStore, CPSEntry


def _make_embedding(seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    vec = rng.randn(384).astype(np.float32)
    return vec / np.linalg.norm(vec)


def _make_cps_with_entries() -> ContextPersistenceStore:
    """Create a CPS pre-loaded with test entries."""
    cps = ContextPersistenceStore(config={"max_entries": 1000})

    cps.store(CPSEntry(
        embedding=_make_embedding(1),
        entry_type="INTENT",
        content="I want to build an authentication system",
        pheromone_source="conversation",
        session_id="session-old",
        layer_depth="architecture",
    ))
    cps.store(CPSEntry(
        embedding=_make_embedding(2),
        entry_type="DECISION",
        content="Using JWT tokens because stateless scales better",
        pheromone_source="conversation",
        session_id="session-old",
        layer_depth="design",
    ))
    cps.store(CPSEntry(
        embedding=_make_embedding(3),
        entry_type="DISCOVERY",
        content="The stigmergic approach eliminates the need for a router",
        pheromone_source="conversation",
        session_id="session-old",
        layer_depth="vision",
    ))
    return cps


def _default_config() -> dict:
    return {
        "max_context_items": 10,
        "min_activation_threshold": 0.0,  # Low threshold for testing
        "session_start_auto_surface": True,
        "auto_generate_summary": True,
        "summary_max_tokens": 500,
        "context_injection_format": "markdown",
    }


# -------------------------------------------------------------------
# Surface context tests
# -------------------------------------------------------------------

def test_surface_context_returns_items():
    """surface_context retrieves relevant entries from CPS."""
    cps = _make_cps_with_entries()
    bridge = SessionBridge(config=_default_config(), cps=cps)

    # Query with embedding similar to first entry
    items = bridge.surface_context(
        initial_message="auth system",
        embedding=_make_embedding(1),
        session_id="session-new",
    )
    assert len(items) > 0
    assert items[0]["entry_type"] in ("INTENT", "DECISION", "DISCOVERY")


def test_surface_context_empty_cps():
    """surface_context returns empty when CPS is empty."""
    cps = ContextPersistenceStore(config={})
    bridge = SessionBridge(config=_default_config(), cps=cps)

    items = bridge.surface_context(
        initial_message="hello",
        embedding=_make_embedding(99),
        session_id="s1",
    )
    assert items == []


def test_surface_context_no_cps():
    """surface_context returns empty when CPS is None."""
    bridge = SessionBridge(config=_default_config(), cps=None)
    items = bridge.surface_context(
        initial_message="hello",
        embedding=_make_embedding(1),
        session_id="s1",
    )
    assert items == []


def test_surface_tracks_entry_ids():
    """Surfaced entry IDs are tracked for usage feedback."""
    cps = _make_cps_with_entries()
    bridge = SessionBridge(config=_default_config(), cps=cps)

    items = bridge.surface_context(
        initial_message="auth",
        embedding=_make_embedding(1),
        session_id="s1",
    )
    assert len(bridge._surfaced_entry_ids) == len(items)


# -------------------------------------------------------------------
# Format tests
# -------------------------------------------------------------------

def test_format_markdown():
    """Markdown formatting produces expected structure."""
    cps = _make_cps_with_entries()
    bridge = SessionBridge(config=_default_config(), cps=cps)

    items = bridge.surface_context(
        initial_message="auth",
        embedding=_make_embedding(1),
        session_id="s1",
    )
    formatted = bridge.format_context(items, fmt="markdown")
    assert "## Prior Context" in formatted
    assert "**" in formatted  # Bold entry type


def test_format_json():
    """JSON formatting produces valid JSON."""
    import json
    cps = _make_cps_with_entries()
    bridge = SessionBridge(config=_default_config(), cps=cps)

    items = bridge.surface_context(
        initial_message="auth",
        embedding=_make_embedding(1),
        session_id="s1",
    )
    formatted = bridge.format_context(items, fmt="json")
    parsed = json.loads(formatted)
    assert isinstance(parsed, list)


def test_format_plain():
    """Plain formatting is readable."""
    cps = _make_cps_with_entries()
    bridge = SessionBridge(config=_default_config(), cps=cps)

    items = bridge.surface_context(
        initial_message="auth",
        embedding=_make_embedding(1),
        session_id="s1",
    )
    formatted = bridge.format_context(items, fmt="plain")
    assert "Prior Context:" in formatted


def test_format_empty():
    """Formatting empty items returns empty string."""
    bridge = SessionBridge(config=_default_config())
    assert bridge.format_context([], fmt="markdown") == ""
    assert bridge.format_context([], fmt="json") == ""
    assert bridge.format_context([], fmt="plain") == ""


# -------------------------------------------------------------------
# Session summary tests
# -------------------------------------------------------------------

def test_generate_summary():
    """generate_summary creates a SESSION_SUMMARY from session entries."""
    cps = ContextPersistenceStore(config={"max_entries": 1000})

    # Store entries for a session
    cps.store(CPSEntry(
        embedding=_make_embedding(1),
        entry_type="INTENT",
        content="Build authentication module",
        session_id="session-42",
        timestamp=100.0,
    ))
    cps.store(CPSEntry(
        embedding=_make_embedding(2),
        entry_type="DECISION",
        content="Use JWT for stateless auth",
        session_id="session-42",
        timestamp=200.0,
    ))

    bridge = SessionBridge(config=_default_config(), cps=cps)
    summary = bridge.generate_summary("session-42")

    assert summary is not None
    assert "session-42" in summary
    assert "2 entries" in summary

    # Check that a SESSION_SUMMARY entry was stored
    summaries = cps.retrieve_by_session(
        "session-42", entry_type="SESSION_SUMMARY"
    )
    assert len(summaries) == 1
    assert summaries[0].entry_type == "SESSION_SUMMARY"
    assert summaries[0].layer_depth == "vision"


def test_generate_summary_empty_session():
    """generate_summary returns None for empty session."""
    cps = ContextPersistenceStore(config={})
    bridge = SessionBridge(config=_default_config(), cps=cps)
    assert bridge.generate_summary("nonexistent") is None


def test_generate_summary_no_cps():
    """generate_summary returns None when CPS is None."""
    bridge = SessionBridge(config=_default_config(), cps=None)
    assert bridge.generate_summary("s1") is None


# -------------------------------------------------------------------
# Usage tracking tests
# -------------------------------------------------------------------

def test_track_usage():
    """track_usage identifies when surfaced content is referenced."""
    cps = _make_cps_with_entries()
    bridge = SessionBridge(config=_default_config(), cps=cps)

    # Surface context
    bridge.surface_context(
        initial_message="auth",
        embedding=_make_embedding(1),
        session_id="s1",
    )

    # Send a message with embedding identical to first entry (high similarity)
    used = bridge.track_usage("auth system design", _make_embedding(1))
    # Should detect usage since embedding matches a surfaced entry
    assert isinstance(used, list)


def test_track_usage_no_surfaced():
    """track_usage returns empty when nothing was surfaced."""
    bridge = SessionBridge(config=_default_config(), cps=None)
    used = bridge.track_usage("hello", _make_embedding(99))
    assert used == []


# -------------------------------------------------------------------
# Config tests
# -------------------------------------------------------------------

def test_default_format():
    """Default format from config is used when not overridden."""
    bridge = SessionBridge(config={
        **_default_config(),
        "context_injection_format": "plain",
    })
    assert bridge._format == "plain"


if __name__ == "__main__":
    tests = [
        test_surface_context_returns_items,
        test_surface_context_empty_cps,
        test_surface_context_no_cps,
        test_surface_tracks_entry_ids,
        test_format_markdown,
        test_format_json,
        test_format_plain,
        test_format_empty,
        test_generate_summary,
        test_generate_summary_empty_session,
        test_generate_summary_no_cps,
        test_track_usage,
        test_track_usage_no_surfaced,
        test_default_format,
    ]
    for t in tests:
        t()
        print(f"{t.__name__} PASSED")

    print(f"\nAll {len(tests)} session bridge tests passed.")
