"""
Praxis Session Bridge — Context Continuity Across Sessions

Solves context death by surfacing relevant prior knowledge when a new
AI session starts. The bridge queries the CPS through the substrate's
learned topology and formats the results for injection into the new
session's context.

Also generates session summaries on session end, storing them as
SESSION_SUMMARY entries in the CPS for future retrieval.

Canonical source: https://github.com/greatnorthernfishguy-hub/Praxis
License: AGPL-3.0

# ---- Changelog ----
# [2026-03-18] Claude (Opus 4.6) — Initial creation (Phase 1).
#   What: SessionBridge stub with interface definition.
#   Why:  PRD §8.4 — session continuity mechanism.
#   How:  Stubs for surface_context() and generate_summary().
# [2026-03-18] Claude (Opus 4.6) — Phase 4 full implementation.
#   What: Full session bridge with substrate-routed context surfacing,
#         configurable output format (markdown/json/plain), session
#         summary generation from CPS entries, surfacing feedback
#         tracking for outcome learning.
#   Why:  PRD §8.4 — the mechanism that solves context death.
#         PRD §14 Phase 4 deliverable.
#   How:  surface_context() embeds the initial message, retrieves from
#         CPS via substrate-routed search, formats as configured.
#         generate_summary() collects session entries from CPS and
#         compresses into a SESSION_SUMMARY entry.
#         track_usage() records which surfaced items were referenced,
#         feeding outcome signals back through the substrate.
# -------------------
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("praxis.session_bridge")


class SessionBridge:
    """Session Bridge — context continuity (PRD §8.4).

    When a new AI session starts:
    1. Capture the initial signal (first human message)
    2. Embed it and query the CPS through the substrate
    3. Format the retrieved context items
    4. Provide to the AI session

    When a session ends:
    1. Collect all CPS entries from this session
    2. Generate a compressed summary
    3. Store as SESSION_SUMMARY entry for future retrieval

    Surfacing feedback:
    - Track which surfaced items the developer actually references
    - Items used → positive outcome signal → substrate strengthens
    - Items ignored → no signal (natural decay handles weakening)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        cps: Any = None,
        ecosystem: Any = None,
        embed_fn: Any = None,
    ) -> None:
        self._config = config
        self._cps = cps
        self._eco = ecosystem
        self._embed_fn = embed_fn

        # Config values (PRD §13)
        self._max_items = config.get("max_context_items", 10)
        self._min_threshold = config.get("min_activation_threshold", 0.6)
        self._auto_surface = config.get("session_start_auto_surface", True)
        self._auto_summary = config.get("auto_generate_summary", True)
        self._max_summary_tokens = config.get("summary_max_tokens", 500)
        self._format = config.get("context_injection_format", "markdown")

        # Track what was surfaced for outcome feedback
        self._last_surfaced: List[Dict[str, Any]] = []
        self._surfaced_entry_ids: List[str] = []

    def surface_context(
        self,
        initial_message: str,
        embedding: np.ndarray,
        session_id: str,
    ) -> List[Dict[str, Any]]:
        """Surface relevant context for a new session (PRD §8.4).

        Queries the CPS through the substrate's learned topology.
        Returns context items sorted by relevance.

        Args:
            initial_message: First message in the new session
            embedding: Pre-computed embedding of the initial message
            session_id: New session identifier

        Returns:
            List of context item dicts with entry metadata and scores.
        """
        if self._cps is None:
            return []

        # Retrieve from CPS via substrate-routed search
        results: List[Tuple[Any, float]] = self._cps.retrieve(
            embedding,
            top_k=self._max_items,
            min_similarity=self._min_threshold,
        )

        # Build context items
        items: List[Dict[str, Any]] = []
        self._surfaced_entry_ids = []

        for entry, score in results:
            item = {
                "entry_id": entry.entry_id,
                "entry_type": entry.entry_type,
                "content": entry.content,
                "session_id": entry.session_id,
                "layer_depth": entry.layer_depth,
                "timestamp": entry.timestamp,
                "score": round(score, 4),
                "outcome_signal": entry.outcome_signal,
            }
            items.append(item)
            self._surfaced_entry_ids.append(entry.entry_id)

        self._last_surfaced = items
        return items

    def format_context(
        self,
        items: List[Dict[str, Any]],
        fmt: Optional[str] = None,
    ) -> str:
        """Format surfaced context items for injection (PRD §13).

        Args:
            items: Context items from surface_context()
            fmt: Override format ('markdown', 'json', 'plain').
                 Defaults to config value.

        Returns:
            Formatted string ready for prompt injection.
        """
        fmt = fmt or self._format

        if not items:
            return ""

        if fmt == "json":
            return self._format_json(items)
        elif fmt == "plain":
            return self._format_plain(items)
        else:
            return self._format_markdown(items)

    def generate_summary(
        self,
        session_id: str,
        embedding: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """Generate and store a session summary (PRD §8.4).

        Collects all CPS entries from this session, compresses them
        into a narrative summary, and stores as a SESSION_SUMMARY
        entry in the CPS for future retrieval.

        Args:
            session_id: Session to summarize
            embedding: Optional embedding for the summary. If None,
                       uses the mean of session entry embeddings.

        Returns:
            Summary text, or None if session had no entries.
        """
        if self._cps is None:
            return None

        entries = self._cps.retrieve_by_session(session_id)
        if not entries:
            return None

        # Build summary from entries
        parts: List[str] = []
        entry_types_seen: Dict[str, int] = {}
        embeddings: List[np.ndarray] = []

        for entry in entries:
            entry_types_seen[entry.entry_type] = (
                entry_types_seen.get(entry.entry_type, 0) + 1
            )
            if entry.embedding is not None:
                embeddings.append(entry.embedding)

            # Include content from significant entry types
            if entry.entry_type in ("INTENT", "DECISION", "DISCOVERY"):
                content = entry.content
                if len(content) > 200:
                    content = content[:197] + "..."
                prefix = entry.entry_type.lower().capitalize()
                parts.append(f"- {prefix}: {content}")

        # Compose summary
        type_summary = ", ".join(
            f"{count} {etype.lower()}{'s' if count > 1 else ''}"
            for etype, count in sorted(entry_types_seen.items())
        )
        header = f"Session {session_id}: {len(entries)} entries ({type_summary})"

        # Limit parts to fit token budget (rough: 4 chars ≈ 1 token)
        char_budget = self._max_summary_tokens * 4
        summary_parts = [header]
        chars_used = len(header)
        for part in parts:
            if chars_used + len(part) + 1 > char_budget:
                break
            summary_parts.append(part)
            chars_used += len(part) + 1

        summary_text = "\n".join(summary_parts)

        # Compute summary embedding (mean of session embeddings)
        if embedding is None and embeddings:
            stacked = np.stack(embeddings)
            mean_emb = np.mean(stacked, axis=0).astype(np.float32)
            norm = np.linalg.norm(mean_emb)
            embedding = mean_emb / norm if norm > 0 else mean_emb

        # Store as SESSION_SUMMARY in CPS
        if embedding is not None:
            try:
                self._cps.store_from_signal(
                    content=summary_text,
                    embedding=embedding,
                    entry_type="SESSION_SUMMARY",
                    pheromone_source="conversation",
                    session_id=session_id,
                    layer_depth="vision",  # Summaries are high-level
                    metadata={
                        "entry_count": len(entries),
                        "type_counts": entry_types_seen,
                    },
                )
            except Exception as exc:
                logger.debug("Session summary store failed: %s", exc)

        return summary_text

    def track_usage(
        self,
        text: str,
        embedding: np.ndarray,
    ) -> List[str]:
        """Track which surfaced items are referenced in conversation.

        Compares the new message against previously surfaced items.
        Items with high similarity to the message are considered
        "used" — their outcome signal is recorded as positive in
        the substrate.

        Args:
            text: The conversation message to check
            embedding: Its embedding

        Returns:
            List of entry_ids that were considered "used".
        """
        if not self._last_surfaced or self._cps is None:
            return []

        used_ids: List[str] = []
        usage_threshold = 0.65  # Cosine similarity threshold

        for item in self._last_surfaced:
            entry = self._cps.get_entry(item["entry_id"])
            if entry is None or entry.embedding is None:
                continue

            sim = float(np.dot(embedding, entry.embedding))
            if sim >= usage_threshold:
                used_ids.append(entry.entry_id)

                # Record positive outcome to substrate
                if self._eco is not None:
                    try:
                        self._eco.record_outcome(
                            entry.embedding,
                            target_id=entry.substrate_target_id,
                            success=True,
                            strength=0.8,
                            metadata={
                                "source": "praxis_session_bridge",
                                "feedback_type": "surfaced_context_used",
                            },
                        )
                    except Exception as exc:
                        logger.debug("Usage feedback record failed: %s", exc)

        return used_ids

    # -----------------------------------------------------------------
    # Formatters
    # -----------------------------------------------------------------

    def _format_markdown(self, items: List[Dict[str, Any]]) -> str:
        """Format as markdown context block."""
        lines = ["## Prior Context\n"]
        for item in items:
            entry_type = item["entry_type"].lower().replace("_", " ").title()
            score = item["score"]
            content = item["content"]
            if len(content) > 300:
                content = content[:297] + "..."
            session = item.get("session_id", "unknown")
            layer = item.get("layer_depth", "")

            lines.append(f"**{entry_type}** (relevance: {score:.2f}, {layer})")
            lines.append(f"> {content}")
            lines.append(f"*from session {session}*\n")

        return "\n".join(lines)

    def _format_json(self, items: List[Dict[str, Any]]) -> str:
        """Format as JSON array."""
        import json
        return json.dumps(items, indent=2, default=str)

    def _format_plain(self, items: List[Dict[str, Any]]) -> str:
        """Format as plain text."""
        lines = ["Prior Context:", ""]
        for i, item in enumerate(items, 1):
            entry_type = item["entry_type"]
            content = item["content"]
            if len(content) > 300:
                content = content[:297] + "..."
            lines.append(f"{i}. [{entry_type}] {content}")
        return "\n".join(lines)
