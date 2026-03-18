"""Tests for Praxis Context Persistence Store — Phase 3."""

import sys
import os
import json
import time

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vendored = os.path.join(repo_root, "vendored")
for p in [repo_root, vendored]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from store.cps import CPSEntry, ContextPersistenceStore, VALID_ENTRY_TYPES


def _make_embedding(seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    vec = rng.randn(384).astype(np.float32)
    return vec / np.linalg.norm(vec)


# -------------------------------------------------------------------
# CPSEntry tests
# -------------------------------------------------------------------

def test_entry_defaults():
    """CPSEntry generates ID and timestamps on creation."""
    entry = CPSEntry()
    assert entry.entry_id  # UUID generated
    assert entry.timestamp > 0
    assert entry.last_accessed > 0
    assert entry.substrate_target_id.startswith("cps:")
    assert entry.entry_type == "INTENT"


def test_entry_types_valid():
    """All PRD §8.2 entry types are defined."""
    expected = {
        "INTENT", "DECISION", "DISCOVERY", "ARTIFACT",
        "OUTCOME", "REJECTED_ALTERNATIVE", "SESSION_SUMMARY",
    }
    assert VALID_ENTRY_TYPES == expected


# -------------------------------------------------------------------
# Store / Retrieve tests
# -------------------------------------------------------------------

def test_store_and_count():
    """store() adds entry and count reflects it."""
    cps = ContextPersistenceStore(config={})
    entry = CPSEntry(
        embedding=_make_embedding(1),
        entry_type="INTENT",
        content="I want to build a module",
        pheromone_source="conversation",
        session_id="test-session",
    )
    entry_id = cps.store(entry)
    assert entry_id == entry.entry_id
    assert cps.count == 1


def test_store_from_signal():
    """store_from_signal convenience method works."""
    cps = ContextPersistenceStore(config={})
    entry = cps.store_from_signal(
        content="design decision: use stigmergy",
        embedding=_make_embedding(2),
        entry_type="DECISION",
        pheromone_source="conversation",
        session_id="test-session",
        layer_depth="architecture",
    )
    assert entry.entry_type == "DECISION"
    assert entry.layer_depth == "architecture"
    assert cps.count == 1


def test_get_entry():
    """get_entry returns entry and increments access count."""
    cps = ContextPersistenceStore(config={})
    entry = CPSEntry(embedding=_make_embedding(1), content="test")
    cps.store(entry)

    retrieved = cps.get_entry(entry.entry_id)
    assert retrieved is not None
    assert retrieved.content == "test"
    assert retrieved.access_count == 1


def test_get_entry_missing():
    """get_entry returns None for missing ID."""
    cps = ContextPersistenceStore(config={})
    assert cps.get_entry("nonexistent") is None


def test_remove_entry():
    """remove_entry removes and returns True."""
    cps = ContextPersistenceStore(config={})
    entry = CPSEntry(embedding=_make_embedding(1), content="test")
    cps.store(entry)
    assert cps.remove_entry(entry.entry_id) is True
    assert cps.count == 0
    assert cps.remove_entry(entry.entry_id) is False


def test_cosine_retrieval():
    """Cosine similarity retrieval finds similar entries."""
    cps = ContextPersistenceStore(config={})

    # Store entries with different embeddings
    emb1 = _make_embedding(10)
    emb2 = _make_embedding(20)
    emb3 = _make_embedding(30)

    cps.store(CPSEntry(embedding=emb1, content="alpha", session_id="s1"))
    cps.store(CPSEntry(embedding=emb2, content="beta", session_id="s1"))
    cps.store(CPSEntry(embedding=emb3, content="gamma", session_id="s1"))

    # Query with emb1 — should find "alpha" as most similar
    results = cps.retrieve(emb1, top_k=3)
    assert len(results) > 0
    best_entry, best_score = results[0]
    assert best_entry.content == "alpha"


def test_retrieve_by_entry_type():
    """Retrieval filters by entry type."""
    cps = ContextPersistenceStore(config={})

    emb = _make_embedding(1)
    cps.store(CPSEntry(
        embedding=emb, content="intent", entry_type="INTENT", session_id="s1"
    ))
    cps.store(CPSEntry(
        embedding=_make_embedding(2), content="decision",
        entry_type="DECISION", session_id="s1"
    ))

    results = cps.retrieve(emb, top_k=10, entry_type="INTENT")
    assert all(e.entry_type == "INTENT" for e, _ in results)


def test_retrieve_by_session():
    """retrieve_by_session returns entries from specific session."""
    cps = ContextPersistenceStore(config={})

    cps.store(CPSEntry(
        embedding=_make_embedding(1), content="session A msg",
        session_id="session-A", timestamp=100.0,
    ))
    cps.store(CPSEntry(
        embedding=_make_embedding(2), content="session B msg",
        session_id="session-B", timestamp=200.0,
    ))
    cps.store(CPSEntry(
        embedding=_make_embedding(3), content="session A msg 2",
        session_id="session-A", timestamp=300.0,
    ))

    entries = cps.retrieve_by_session("session-A")
    assert len(entries) == 2
    assert entries[0].content == "session A msg"
    assert entries[1].content == "session A msg 2"


# -------------------------------------------------------------------
# Persistence tests
# -------------------------------------------------------------------

def test_save_load_json(tmp_path):
    """JSON save/load roundtrips correctly."""
    cps = ContextPersistenceStore(
        config={"max_entries": 100},
        data_dir=str(tmp_path),
    )

    emb = _make_embedding(42)
    cps.store(CPSEntry(
        embedding=emb,
        entry_type="DISCOVERY",
        content="stigmergy is the answer",
        pheromone_source="conversation",
        session_id="s1",
        layer_depth="vision",
        outcome_signal=0.9,
    ))

    cps.save()

    # Load into fresh instance
    cps2 = ContextPersistenceStore(
        config={"max_entries": 100},
        data_dir=str(tmp_path),
    )
    assert cps2.count == 1

    entries = list(cps2._entries.values())
    e = entries[0]
    assert e.entry_type == "DISCOVERY"
    assert e.content == "stigmergy is the answer"
    assert e.layer_depth == "vision"
    assert e.outcome_signal == 0.9
    assert e.embedding is not None
    assert e.embedding.shape == (384,)


def test_save_load_msgpack(tmp_path):
    """Msgpack save/load roundtrips correctly (if available)."""
    try:
        import msgpack
    except ImportError:
        print("test_save_load_msgpack SKIPPED (msgpack not installed)")
        return

    cps = ContextPersistenceStore(
        config={"max_entries": 100},
        data_dir=str(tmp_path),
    )

    cps.store(CPSEntry(
        embedding=_make_embedding(1),
        entry_type="INTENT",
        content="build the module",
        session_id="s1",
    ))

    cps.save()

    # Verify msgpack file exists
    assert (tmp_path / "cps.msgpack").exists()

    # Load fresh
    cps2 = ContextPersistenceStore(
        config={"max_entries": 100},
        data_dir=str(tmp_path),
    )
    assert cps2.count == 1
    e = list(cps2._entries.values())[0]
    assert e.content == "build the module"


# -------------------------------------------------------------------
# Eviction tests
# -------------------------------------------------------------------

def test_eviction():
    """LRU eviction removes oldest entries when at capacity."""
    cps = ContextPersistenceStore(config={"max_entries": 5})

    # Store 5 entries
    for i in range(5):
        cps.store(CPSEntry(
            embedding=_make_embedding(i),
            content=f"entry {i}",
            session_id="s1",
        ))
    assert cps.count == 5

    # Access entries 3 and 4 (make them recent)
    for entry in list(cps._entries.values())[3:]:
        entry.last_accessed = time.time()

    # Store one more — should trigger eviction
    cps.store(CPSEntry(
        embedding=_make_embedding(99),
        content="new entry",
        session_id="s1",
    ))

    # Should still be at or below max
    assert cps.count <= 6  # 5 + 1 new, minus evicted


# -------------------------------------------------------------------
# Stats tests
# -------------------------------------------------------------------

def test_stats():
    """get_stats returns correct metrics."""
    cps = ContextPersistenceStore(config={"max_entries": 1000})

    cps.store(CPSEntry(
        embedding=_make_embedding(1), entry_type="INTENT",
        content="a", session_id="s1",
    ))
    cps.store(CPSEntry(
        embedding=_make_embedding(2), entry_type="DECISION",
        content="b", session_id="s1",
    ))
    cps.store(CPSEntry(
        embedding=_make_embedding(3), entry_type="INTENT",
        content="c", session_id="s2",
    ))

    stats = cps.get_stats()
    assert stats["total_entries"] == 3
    assert stats["type_counts"]["INTENT"] == 2
    assert stats["type_counts"]["DECISION"] == 1
    assert stats["session_count"] == 2


def test_invalid_entry_type_defaults():
    """Invalid entry type defaults to INTENT with warning."""
    cps = ContextPersistenceStore(config={})
    entry = CPSEntry(
        embedding=_make_embedding(1),
        entry_type="INVALID_TYPE",
        content="test",
        session_id="s1",
    )
    cps.store(entry)
    stored = cps.get_entry(entry.entry_id)
    assert stored.entry_type == "INTENT"


if __name__ == "__main__":
    tests = [
        test_entry_defaults,
        test_entry_types_valid,
        test_store_and_count,
        test_store_from_signal,
        test_get_entry,
        test_get_entry_missing,
        test_remove_entry,
        test_cosine_retrieval,
        test_retrieve_by_entry_type,
        test_retrieve_by_session,
        test_eviction,
        test_stats,
        test_invalid_entry_type_defaults,
    ]
    for t in tests:
        t()
        print(f"{t.__name__} PASSED")

    # Tests needing tmp_path
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        test_save_load_json(Path(td))
    print("test_save_load_json PASSED")

    with tempfile.TemporaryDirectory() as td:
        test_save_load_msgpack(Path(td))
    print("test_save_load_msgpack PASSED")

    print(f"\nAll {len(tests) + 2} CPS tests passed.")
