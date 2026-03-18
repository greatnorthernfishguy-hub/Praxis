"""
Praxis — Entry Point

Standard E-T Systems module entry point. Initializes the Praxis hook
singleton and reports readiness.

Canonical source: https://github.com/greatnorthernfishguy-hub/Praxis
License: AGPL-3.0

# ---- Changelog ----
# [2026-03-18] Claude (Opus 4.6) — Initial creation (Phase 1).
#   What: Module entry point that initializes the PraxisHook singleton.
#   Why:  PRD §14 Phase 1 — et_module.json specifies main.py as entry.
#   How:  Import and call get_instance() to trigger initialization.
# -------------------
"""

from __future__ import annotations

import json
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("praxis.main")


def main() -> None:
    """Initialize Praxis and report health."""
    # Ensure vendored directory is on path
    import os
    vendored_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendored")
    if vendored_path not in sys.path:
        sys.path.insert(0, vendored_path)

    # Also ensure repo root is on path for core/ imports
    repo_root = os.path.dirname(os.path.abspath(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from core.praxis_hook import get_instance

    logger.info("[Praxis] Starting...")
    instance = get_instance()
    health = instance.health()
    logger.info("[Praxis] Health: %s", json.dumps(health, indent=2))
    logger.info("[Praxis] Ready.")


if __name__ == "__main__":
    main()
