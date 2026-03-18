"""
Praxis Session Bridge — Context Continuity Across Sessions

Solves context death by surfacing relevant prior knowledge when a new
AI session starts. This is a stub for Phase 1 — full implementation
in Phase 4 (v0.4).

Canonical source: https://github.com/greatnorthernfishguy-hub/Praxis
License: AGPL-3.0

# ---- Changelog ----
# [2026-03-18] Claude (Opus 4.6) — Initial creation (Phase 1).
#   What: SessionBridge stub with interface definition.
#   Why:  PRD §8.4 — session continuity mechanism.
#         Full implementation deferred to Phase 4.
#   How:  Class with surface_context() and generate_summary() stubs.
# -------------------
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class SessionBridge:
    """Session Bridge — context continuity (PRD §8.4).

    When a new AI session starts:
    1. Capture the initial signal (first human message)
    2. Activate the substrate with the signal's embedding
    3. Propagation through learned topology activates related nodes
    4. SurfacingMonitor identifies above-threshold context
    5. CPS entries associated with surfaced nodes are retrieved
    6. Context summary generated and provided to the AI session

    Phase 1: Interface defined. Full implementation in Phase 4.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config

    def surface_context(
        self,
        initial_message: str,
        session_id: str,
    ) -> List[Dict[str, Any]]:
        """Surface relevant context for a new session.

        Phase 1 stub — returns empty list.

        Args:
            initial_message: First message in the new session
            session_id: New session identifier

        Returns:
            List of context items to inject into the session.
        """
        return []

    def generate_summary(self, session_id: str) -> Optional[str]:
        """Generate a summary for an ending session.

        Phase 1 stub — returns None.

        Args:
            session_id: Session to summarize

        Returns:
            Summary text, or None if not available.
        """
        return None
