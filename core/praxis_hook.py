"""
Praxis OpenClaw Hook — E-T Systems Standard Integration

Exposes Praxis's context intelligence as an OpenClaw skill, using the
standardized OpenClawAdapter base class.

OpenClaw calls get_instance().on_message(text) on every turn.
The adapter handles all ecosystem wiring (Tier 1/2/3 learning) and
memory logging. This file implements what's unique to Praxis:

  - _embed():              Sentence-transformer embedding / hash fallback
  - _module_on_message():  Capture conversation signal, record to substrate
  - _module_stats():       Praxis-specific telemetry

SKILL.md entry:
    name: praxis
    autoload: true
    hook: core/praxis_hook.py::get_instance

Canonical source: https://github.com/greatnorthernfishguy-hub/Praxis
License: AGPL-3.0

# ---- Changelog ----
# [2026-03-18] Claude (Opus 4.6) — Initial creation (Phase 1).
#   What: PraxisHook class subclassing OpenClawAdapter. Initializes
#         config, embedding, autonomic state reader, signal capture.
#   Why:  PRD §14 Phase 1 — foundation module that registers, connects
#         to ecosystem, reports health, and produces/consumes signals.
#   How:  OpenClawAdapter subclass with singleton get_instance().
#         Captures conversation signals on each on_message() call.
#         Records raw embeddings to substrate (Law 7 — no classification
#         before the substrate sees the data).
# -------------------
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from openclaw_adapter import OpenClawAdapter

logger = logging.getLogger("praxis")


class PraxisHook(OpenClawAdapter):
    """OpenClaw integration hook for Praxis (PRD §14 Phase 1)."""

    MODULE_ID = "praxis"
    SKILL_NAME = "Praxis Context Intelligence"
    WORKSPACE_ENV = "PRAXIS_WORKSPACE_DIR"
    DEFAULT_WORKSPACE = "~/.openclaw/praxis"

    def __init__(self) -> None:
        super().__init__()

        # --- Load Configuration (PRD §13) ---
        from core.config import PraxisConfig

        config_path = os.path.join(
            os.path.expanduser("~/.et_modules/praxis"), "config.yaml"
        )
        # Also check local config.yaml
        if not os.path.exists(config_path):
            local_config = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                os.pardir, "config.yaml",
            )
            if os.path.exists(local_config):
                config_path = local_config

        self._cfg = PraxisConfig.from_yaml(config_path)

        # --- Data Directories ---
        self._data_dir = Path(
            os.path.expanduser("~/.et_modules/praxis")
        )
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # --- Session Tracking ---
        self._session_id = f"session_{int(time.time())}"
        self._turn_index = 0
        self._signal_count = 0

        # --- Autonomic State (PRD §9 — read only) ---
        self._autonomic_state = "PARASYMPATHETIC"
        try:
            import ng_autonomic
            state = ng_autonomic.read_state()
            self._autonomic_state = state.get("state", "PARASYMPATHETIC")
        except Exception:
            pass

        # --- Checkpointing ---
        self._last_checkpoint = time.time()
        self._checkpoint_interval = self._cfg.checkpoint_interval_seconds

        logger.info(
            "[Praxis] Initialized — session=%s, autonomic=%s, tier=%d",
            self._session_id,
            self._autonomic_state,
            self._eco.tier,
        )

    # -----------------------------------------------------------------
    # OpenClawAdapter implementation
    # -----------------------------------------------------------------

    def _embed(self, text: str) -> np.ndarray:
        """Embed text using fastembed/sentence-transformers, fall back to hash.

        PRD §6.3: Ecosystem-standard all-MiniLM-L6-v2, 384 dimensions.
        Primary backend: fastembed (ONNX Runtime). Fallback: sentence-transformers.
        Last resort: hash-based embedding.
        """
        if self._cfg.embedding.device != "disabled":
            # Try fastembed first (ONNX Runtime — no torch dependency)
            try:
                if not hasattr(self, "_fe_model"):
                    from fastembed import TextEmbedding
                    self._fe_model = TextEmbedding(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                embeddings = list(self._fe_model.embed([text]))
                vec = np.array(embeddings[0], dtype=np.float32)
                norm = np.linalg.norm(vec)
                return vec / norm if norm > 0 else vec
            except Exception:
                pass

            # Fallback: sentence-transformers
            try:
                if not hasattr(self, "_st_model"):
                    from sentence_transformers import SentenceTransformer
                    self._st_model = SentenceTransformer(
                        self._cfg.embedding.model
                    )
                vec = self._st_model.encode(text, normalize_embeddings=True)
                return np.array(vec, dtype=np.float32)
            except Exception:
                pass

        return self._hash_embed(text)

    def _module_on_message(
        self, text: str, embedding: np.ndarray
    ) -> Dict[str, Any]:
        """Praxis-specific processing on each OpenClaw message.

        Phase 1: Capture conversation signal and record to substrate.
        Raw embedding enters the substrate — no classification (Law 7).

        1. Create a WorkflowSignal for the conversation turn
        2. Record the signal to the substrate via NG-Lite
        3. Read autonomic state and adjust behavior
        4. Auto-checkpoint
        """
        from core.signals import WorkflowSignal, ConversationMeta

        self._turn_index += 1
        self._signal_count += 1

        # Read autonomic state (PRD §9)
        try:
            import ng_autonomic
            state = ng_autonomic.read_state()
            self._autonomic_state = state.get("state", "PARASYMPATHETIC")
        except Exception:
            pass

        # Skip very short messages unless in SYMPATHETIC (capture everything)
        if (
            len(text) < self._cfg.sensors.conversation.min_message_length
            and self._autonomic_state != "SYMPATHETIC"
        ):
            return {"status": "skipped", "reason": "below_min_length"}

        # Build conversation metadata
        conv_meta = ConversationMeta(
            direction="human",
            speaker_id="default",
            modality="text",
            turn_index=self._turn_index,
            message_length=len(text),
        )

        # Create workflow signal
        signal = WorkflowSignal(
            pheromone="conversation",
            timestamp=time.time(),
            embedding=embedding,
            metadata=conv_meta.__dict__,
            session_id=self._session_id,
            layer_depth="implementation",
        )

        # Record to substrate — raw embedding, no classification (Law 7)
        try:
            self._eco.record_outcome(
                embedding,
                target_id=f"conv:{signal.signal_id}",
                success=True,
                metadata={
                    "source": "praxis",
                    "pheromone": "conversation",
                    "session_id": self._session_id,
                    "turn_index": self._turn_index,
                },
            )
        except Exception as exc:
            logger.debug("Substrate record failed: %s", exc)

        # Auto-checkpoint
        now = time.time()
        if now - self._last_checkpoint > self._checkpoint_interval:
            self._checkpoint()
            self._last_checkpoint = now

        return {
            "status": "captured",
            "signal_id": signal.signal_id,
            "pheromone": "conversation",
            "turn_index": self._turn_index,
            "autonomic_state": self._autonomic_state,
            "total_signals": self._signal_count,
        }

    def _module_stats(self) -> Dict[str, Any]:
        """Praxis-specific telemetry."""
        return {
            "session_id": self._session_id,
            "turn_index": self._turn_index,
            "signal_count": self._signal_count,
            "autonomic_state": self._autonomic_state,
            "sensors": {
                "conversation": {"enabled": self._cfg.sensors.conversation.enabled},
                "artifact": {"enabled": self._cfg.sensors.artifact.enabled},
                "outcome": {"enabled": self._cfg.sensors.outcome.enabled},
            },
        }

    # -----------------------------------------------------------------
    # Health endpoint (PRD §14 Phase 1)
    # -----------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        """Basic health check for Praxis."""
        return {
            "module": "praxis",
            "status": "healthy",
            "tier": self._eco.tier,
            "tier_name": self._eco.tier_name,
            "session_id": self._session_id,
            "signal_count": self._signal_count,
            "autonomic_state": self._autonomic_state,
            "uptime_seconds": round(time.time() - self._start_time, 1),
        }

    # -----------------------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------------------

    def _checkpoint(self) -> None:
        """Save ecosystem state."""
        try:
            self._eco.save()
        except Exception as exc:
            logger.debug("Ecosystem checkpoint failed: %s", exc)

    # -----------------------------------------------------------------
    # Shutdown
    # -----------------------------------------------------------------

    def shutdown(self) -> None:
        """Graceful shutdown — checkpoint."""
        self._checkpoint()
        logger.info("[Praxis] Shutdown complete")


# --------------------------------------------------------------------------
# Singleton wiring — identical pattern for all E-T Systems modules
# --------------------------------------------------------------------------

_INSTANCE: Optional[PraxisHook] = None


def get_instance() -> PraxisHook:
    """Return the Praxis OpenClaw hook singleton."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = PraxisHook()
    return _INSTANCE
