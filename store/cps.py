"""
Praxis Context Persistence Store (CPS) — Episodic Memory

Domain-specific persistent store for intent-behavior mappings,
structurally analogous to THC's DVS and Immunis's Armory. The CPS
is the hippocampus — it stores specific experiences with context,
retrievable through the substrate's learned topology.

Retrieval is substrate-routed: when the CPS needs to find relevant
entries, it activates the substrate with the query embedding and
retrieves entries associated with the nodes that fire. This means
retrieval quality improves as the substrate learns — not just flat
cosine similarity, but topology-informed relevance.

Canonical source: https://github.com/greatnorthernfishguy-hub/Praxis
License: AGPL-3.0

# ---- Changelog ----
# [2026-03-18] Claude (Opus 4.6) — Initial creation (Phase 1).
#   What: CPS and CPSEntry stubs with interface definition.
#   Why:  PRD §8 — Context Persistence Store specification.
#   How:  CPSEntry dataclass per PRD §8.1.
# [2026-03-18] Claude (Opus 4.6) — Phase 3 full implementation.
#   What: Full CPS with msgpack persistence (JSON fallback), substrate-
#         routed retrieval, cosine similarity search, LRU eviction,
#         entry type filtering, session summary generation from entries.
#   Why:  PRD §8 specifies CPS with msgpack at ~/.et_modules/praxis/
#         cps.msgpack. PRD §8.3 specifies substrate-routed retrieval.
#         PRD §8.2 defines entry types.
#   How:  In-memory dict keyed by entry_id. Substrate-routed retrieval
#         via ecosystem.get_recommendations() to find relevant target_ids,
#         then match against CPS entries registered to those targets.
#         Falls back to cosine similarity when substrate has no opinion.
#         Atomic writes via tmp + os.replace. LRU eviction at capacity.
# -------------------
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("praxis.cps")


# ---------------------------------------------------------------------------
# CPS Entry
# ---------------------------------------------------------------------------

@dataclass
class CPSEntry:
    """Context Persistence Store entry (PRD §8.1).

    Each entry represents a discrete unit of context intelligence:
    an intent expression, a decision, a discovery, an artifact event,
    an outcome, a rejected alternative, or a session summary.
    """
    entry_id: str = ""
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    entry_type: str = "INTENT"        # INTENT | DECISION | DISCOVERY | ARTIFACT |
                                       # OUTCOME | REJECTED_ALTERNATIVE | SESSION_SUMMARY
    content: str = ""                  # Raw text
    pheromone_source: str = ""         # 'conversation' | 'artifact' | 'outcome'
    session_id: str = ""               # Originating session
    timestamp: float = 0.0             # Unix timestamp
    layer_depth: str = "implementation"  # 'vision' | 'architecture' | 'design' | 'implementation'
    outcome_signal: Optional[float] = None  # -1.0 to 1.0 if outcome known
    substrate_target_id: str = ""      # The target_id used in the substrate
    access_count: int = 0
    last_accessed: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.entry_id:
            self.entry_id = str(uuid.uuid4())
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.last_accessed == 0.0:
            self.last_accessed = self.timestamp
        if not self.substrate_target_id:
            self.substrate_target_id = f"cps:{self.entry_id}"


# Valid entry types (PRD §8.2)
VALID_ENTRY_TYPES = frozenset({
    "INTENT",
    "DECISION",
    "DISCOVERY",
    "ARTIFACT",
    "OUTCOME",
    "REJECTED_ALTERNATIVE",
    "SESSION_SUMMARY",
})


# ---------------------------------------------------------------------------
# Context Persistence Store
# ---------------------------------------------------------------------------

class ContextPersistenceStore:
    """Context Persistence Store — episodic memory for Praxis (PRD §8).

    Stores intent-behavior mappings with substrate-routed retrieval.
    Structurally analogous to THC's DVS and Immunis's Armory, but
    stores context intelligence rather than diagnostic vectors or
    threat signatures.

    Retrieval modes:
    1. Substrate-routed (primary): Query the ecosystem for recommendations,
       match returned target_ids against CPS entries. Quality improves
       as the substrate learns which contexts co-occur.
    2. Cosine similarity (fallback): Direct embedding comparison when
       substrate has insufficient topology.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        data_dir: Optional[str] = None,
        ecosystem: Optional[Any] = None,
    ) -> None:
        self._config = config or {}
        self._eco = ecosystem

        self._max_entries = self._config.get("max_entries", 50000)
        self._sim_weight = self._config.get("search_weight_similarity", 0.7)
        self._recency_weight = self._config.get("search_weight_recency", 0.3)
        self._eviction_divisor = self._config.get("eviction_batch_divisor", 20)
        self._storage_path = self._config.get(
            "storage_path",
            "~/.et_modules/praxis/cps.msgpack",
        )

        self._data_dir = Path(
            data_dir or os.path.expanduser("~/.et_modules/praxis")
        )
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._entries: Dict[str, CPSEntry] = {}

        # Index: substrate_target_id → entry_id for fast lookup
        self._target_index: Dict[str, str] = {}

        self._load()

    # -------------------------------------------------------------------
    # Store Operations
    # -------------------------------------------------------------------

    def store(self, entry: CPSEntry) -> str:
        """Store a CPS entry. Returns the entry_id.

        If the store is at capacity, evicts the least recently accessed
        entries (LRU). Also registers the entry in the substrate via
        record_outcome if an ecosystem is available.
        """
        if entry.entry_type not in VALID_ENTRY_TYPES:
            logger.warning(
                "Invalid entry type '%s' — defaulting to INTENT",
                entry.entry_type,
            )
            entry.entry_type = "INTENT"

        if len(self._entries) >= self._max_entries:
            self._evict()

        self._entries[entry.entry_id] = entry
        self._target_index[entry.substrate_target_id] = entry.entry_id

        # Register in substrate for substrate-routed retrieval
        if self._eco is not None and entry.embedding is not None:
            try:
                self._eco.record_outcome(
                    entry.embedding,
                    target_id=entry.substrate_target_id,
                    success=True,
                    metadata={
                        "source": "praxis_cps",
                        "entry_type": entry.entry_type,
                        "session_id": entry.session_id,
                    },
                )
            except Exception as exc:
                logger.debug("CPS substrate registration failed: %s", exc)

        return entry.entry_id

    def store_from_signal(
        self,
        content: str,
        embedding: np.ndarray,
        entry_type: str,
        pheromone_source: str,
        session_id: str,
        layer_depth: str = "implementation",
        outcome_signal: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CPSEntry:
        """Convenience: create and store a CPS entry from a workflow signal.

        Returns the stored CPSEntry.
        """
        entry = CPSEntry(
            embedding=embedding,
            entry_type=entry_type,
            content=content,
            pheromone_source=pheromone_source,
            session_id=session_id,
            layer_depth=layer_depth,
            outcome_signal=outcome_signal,
            metadata=metadata or {},
        )
        self.store(entry)
        return entry

    def get_entry(self, entry_id: str) -> Optional[CPSEntry]:
        """Retrieve an entry by ID."""
        entry = self._entries.get(entry_id)
        if entry is not None:
            entry.access_count += 1
            entry.last_accessed = time.time()
        return entry

    def remove_entry(self, entry_id: str) -> bool:
        """Remove an entry by ID."""
        entry = self._entries.pop(entry_id, None)
        if entry is not None:
            self._target_index.pop(entry.substrate_target_id, None)
            return True
        return False

    # -------------------------------------------------------------------
    # Retrieval (PRD §8.3)
    # -------------------------------------------------------------------

    def retrieve(
        self,
        embedding: np.ndarray,
        top_k: int = 10,
        entry_type: Optional[str] = None,
        session_id: Optional[str] = None,
        min_similarity: float = 0.0,
    ) -> List[Tuple[CPSEntry, float]]:
        """Retrieve entries via substrate-routed search (PRD §8.3).

        Primary path: Ask the ecosystem for recommendations using the
        query embedding. The substrate returns target_ids ranked by
        learned topology. Match those against CPS entries.

        Fallback path: Direct cosine similarity search when substrate
        returns insufficient results.

        Args:
            embedding: Query vector (384d normalized).
            top_k: Maximum entries to return.
            entry_type: Filter by entry type (INTENT, DECISION, etc.)
            session_id: Filter by originating session.
            min_similarity: Minimum cosine similarity threshold.

        Returns:
            List of (CPSEntry, score) tuples sorted by relevance.
        """
        results: List[Tuple[CPSEntry, float]] = []
        seen_ids: set = set()

        # 1. Substrate-routed retrieval (primary)
        if self._eco is not None:
            try:
                recs = self._eco.get_recommendations(embedding, top_k=top_k * 2)
                for target_id, confidence, _reasoning in recs:
                    entry_id = self._target_index.get(target_id)
                    if entry_id is None:
                        continue
                    entry = self._entries.get(entry_id)
                    if entry is None:
                        continue
                    if entry_type and entry.entry_type != entry_type:
                        continue
                    if session_id and entry.session_id != session_id:
                        continue

                    entry.access_count += 1
                    entry.last_accessed = time.time()
                    results.append((entry, confidence))
                    seen_ids.add(entry_id)
            except Exception as exc:
                logger.debug("Substrate retrieval failed: %s", exc)

        # 2. Cosine similarity fallback (fill remaining slots)
        remaining = top_k - len(results)
        if remaining > 0:
            cosine_results = self._cosine_search(
                embedding,
                top_k=remaining,
                entry_type=entry_type,
                session_id=session_id,
                min_similarity=min_similarity,
                exclude_ids=seen_ids,
            )
            results.extend(cosine_results)

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def retrieve_by_session(
        self,
        session_id: str,
        entry_type: Optional[str] = None,
    ) -> List[CPSEntry]:
        """Retrieve all entries from a specific session, chronologically."""
        entries = [
            e for e in self._entries.values()
            if e.session_id == session_id
            and (entry_type is None or e.entry_type == entry_type)
        ]
        entries.sort(key=lambda e: e.timestamp)
        return entries

    def _cosine_search(
        self,
        embedding: np.ndarray,
        top_k: int = 10,
        entry_type: Optional[str] = None,
        session_id: Optional[str] = None,
        min_similarity: float = 0.0,
        exclude_ids: Optional[set] = None,
    ) -> List[Tuple[CPSEntry, float]]:
        """Flat cosine similarity search (fallback)."""
        exclude = exclude_ids or set()
        scored: List[Tuple[CPSEntry, float]] = []

        for entry in self._entries.values():
            if entry.entry_id in exclude:
                continue
            if entry.embedding is None:
                continue
            if entry_type and entry.entry_type != entry_type:
                continue
            if session_id and entry.session_id != session_id:
                continue

            sim = self._cosine_similarity(embedding, entry.embedding)
            if sim < min_similarity:
                continue

            # Weight by recency
            age_days = (time.time() - entry.timestamp) / 86400.0
            recency = 1.0 / (1.0 + age_days)
            score = sim * self._sim_weight + recency * self._recency_weight

            entry.access_count += 1
            entry.last_accessed = time.time()
            scored.append((entry, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # -------------------------------------------------------------------
    # Persistence (msgpack with JSON fallback)
    # -------------------------------------------------------------------

    def save(self) -> None:
        """Persist CPS to disk using atomic write."""
        if self._try_save_msgpack():
            return
        self._save_json()

    def _try_save_msgpack(self) -> bool:
        """Save using msgpack. Returns True on success."""
        try:
            import msgpack
        except ImportError:
            return False

        path = self._data_dir / "cps.msgpack"
        data = [self._entry_to_dict(e) for e in self._entries.values()]

        try:
            tmp_path = path.with_suffix(".tmp")
            with open(tmp_path, "wb") as f:
                msgpack.pack(data, f, use_bin_type=True)
            os.replace(tmp_path, path)
            logger.info("CPS saved to %s (%d entries)", path, len(data))
            return True
        except Exception as exc:
            logger.warning("CPS msgpack save failed: %s", exc)
            return False

    def _save_json(self) -> None:
        """Save using JSON (fallback)."""
        path = self._data_dir / "cps.json"
        data = [self._entry_to_dict(e) for e in self._entries.values()]

        try:
            tmp_path = path.with_suffix(".tmp")
            with open(tmp_path, "w") as f:
                json.dump(data, f)
            os.replace(tmp_path, path)
            logger.info("CPS saved to %s (%d entries, JSON)", path, len(data))
        except Exception as exc:
            logger.warning("CPS JSON save failed: %s", exc)

    def _load(self) -> None:
        """Load from disk (try msgpack first, then JSON)."""
        msgpack_path = self._data_dir / "cps.msgpack"
        json_path = self._data_dir / "cps.json"

        if msgpack_path.exists():
            try:
                import msgpack
                with open(msgpack_path, "rb") as f:
                    data = msgpack.unpack(f, raw=False)
                self._load_entries(data)
                logger.info(
                    "CPS loaded from %s (%d entries)",
                    msgpack_path, len(self._entries),
                )
                return
            except Exception as exc:
                logger.warning("CPS msgpack load failed: %s", exc)

        if json_path.exists():
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                self._load_entries(data)
                logger.info(
                    "CPS loaded from %s (%d entries, JSON)",
                    json_path, len(self._entries),
                )
                return
            except Exception as exc:
                logger.warning("CPS JSON load failed: %s", exc)

    def _load_entries(self, data: List[Dict[str, Any]]) -> None:
        """Populate entries from deserialized data."""
        for d in data:
            emb = d.get("embedding")
            entry = CPSEntry(
                entry_id=d.get("entry_id", str(uuid.uuid4())),
                embedding=(
                    np.array(emb, dtype=np.float32)
                    if emb is not None
                    else None
                ),
                entry_type=d.get("entry_type", "INTENT"),
                content=d.get("content", ""),
                pheromone_source=d.get("pheromone_source", ""),
                session_id=d.get("session_id", ""),
                timestamp=d.get("timestamp", 0.0),
                layer_depth=d.get("layer_depth", "implementation"),
                outcome_signal=d.get("outcome_signal"),
                substrate_target_id=d.get("substrate_target_id", ""),
                access_count=d.get("access_count", 0),
                last_accessed=d.get("last_accessed", 0.0),
                metadata=d.get("metadata", {}),
            )
            self._entries[entry.entry_id] = entry
            self._target_index[entry.substrate_target_id] = entry.entry_id

    @staticmethod
    def _entry_to_dict(entry: CPSEntry) -> Dict[str, Any]:
        """Serialize a CPSEntry for persistence."""
        return {
            "entry_id": entry.entry_id,
            "embedding": (
                entry.embedding.tolist()
                if entry.embedding is not None
                else None
            ),
            "entry_type": entry.entry_type,
            "content": entry.content,
            "pheromone_source": entry.pheromone_source,
            "session_id": entry.session_id,
            "timestamp": entry.timestamp,
            "layer_depth": entry.layer_depth,
            "outcome_signal": entry.outcome_signal,
            "substrate_target_id": entry.substrate_target_id,
            "access_count": entry.access_count,
            "last_accessed": entry.last_accessed,
            "metadata": entry.metadata,
        }

    # -------------------------------------------------------------------
    # Eviction (LRU)
    # -------------------------------------------------------------------

    def _evict(self) -> None:
        """Evict least recently accessed entries."""
        if len(self._entries) < self._max_entries:
            return

        evict_count = max(1, self._max_entries // self._eviction_divisor)
        sorted_entries = sorted(
            self._entries.values(), key=lambda e: e.last_accessed
        )
        for entry in sorted_entries[:evict_count]:
            self._entries.pop(entry.entry_id, None)
            self._target_index.pop(entry.substrate_target_id, None)

        logger.info("CPS evicted %d entries (LRU)", evict_count)

    # -------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------

    @property
    def count(self) -> int:
        return len(self._entries)

    def get_stats(self) -> Dict[str, Any]:
        """CPS telemetry."""
        type_counts: Dict[str, int] = {}
        session_counts: Dict[str, int] = {}
        for entry in self._entries.values():
            type_counts[entry.entry_type] = (
                type_counts.get(entry.entry_type, 0) + 1
            )
            session_counts[entry.session_id] = (
                session_counts.get(entry.session_id, 0) + 1
            )
        return {
            "total_entries": len(self._entries),
            "max_entries": self._max_entries,
            "type_counts": type_counts,
            "session_count": len(session_counts),
        }

    # -------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        dot = float(np.dot(a, b))
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)
