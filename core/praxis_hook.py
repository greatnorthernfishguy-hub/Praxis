"""
Praxis OpenClaw Hook — E-T Systems Standard Integration

Exposes Praxis's context intelligence as an OpenClaw skill, using the
standardized OpenClawAdapter base class.

OpenClaw calls get_instance().on_message(text) on every turn.
The adapter handles all ecosystem wiring (Tier 1/2/3 learning) and
memory logging. This file implements what's unique to Praxis:

  - _embed():              Sentence-transformer embedding / hash fallback
  - _module_on_message():  Capture conversation signal, create temporal
                           synapses, record to substrate
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
#   Why:  PRD §14 Phase 1 — foundation module.
#   How:  OpenClawAdapter subclass with singleton get_instance().
# [2026-03-18] Claude (Opus 4.6) — Phase 2: Conversation Stream.
#   What: Full conversation sensor integration with temporal synapse
#         binding. ConversationSensor initialized in __init__. Each
#         on_message creates temporal synapses between the new signal
#         and all recent signals within the temporal window (300s).
#   Why:  PRD §14 Phase 2 — conversation signals propagate through
#         substrate with temporal binding. PRD §6.4 specifies the
#         signal-to-substrate flow.
#   How:  On each message: (1) embed via _embed(), (2) feed to
#         ConversationSensor, (3) record to substrate via
#         record_outcome(), (4) create temporal bindings by recording
#         outcomes linking new signal to each recent signal.
#         Temporal binding strength decays with age (newer = stronger).
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
    """OpenClaw integration hook for Praxis."""

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

        # --- Initialize Conversation Sensor (PRD §4.1) ---
        from sensors.conversation import ConversationSensor

        conv_config = {
            "temporal_window_seconds": self._cfg.sensors.conversation.temporal_window_seconds,
            "capture_both_directions": self._cfg.sensors.conversation.capture_both_directions,
            "min_message_length": self._cfg.sensors.conversation.min_message_length,
        }
        self._conv_sensor = ConversationSensor(config=conv_config)
        self._conv_sensor.set_log_path(self._data_dir / "conversation.jsonl")

        # --- Session Tracking ---
        self._session_id = f"session_{int(time.time())}"
        self._turn_index = 0
        self._signal_count = 0

        # --- Autonomic State (PRD §9 — read only) ---
        self._autonomic_state = "PARASYMPATHETIC"
        self._read_autonomic()

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

        Phase 2 signal-to-substrate flow (PRD §6.4):
        1. Feed text + embedding to ConversationSensor
        2. Record the signal to the substrate (raw embedding — Law 7)
        3. Create temporal synapses to recent signals within the window
        4. Read autonomic state and adjust behavior
        5. Auto-checkpoint

        Temporal binding creates associative structure between signals
        that co-occur in time. Binding strength decays with age —
        a message from 10 seconds ago binds more strongly than one
        from 280 seconds ago. This gives the substrate temporal
        context without explicit classification.
        """
        self._turn_index += 1
        self._signal_count += 1

        # Read autonomic state (PRD §9)
        self._read_autonomic()

        # Skip short messages unless SYMPATHETIC (capture everything)
        if (
            len(text) < self._cfg.sensors.conversation.min_message_length
            and self._autonomic_state != "SYMPATHETIC"
        ):
            return {"status": "skipped", "reason": "below_min_length"}

        # 1. Feed to ConversationSensor — creates signal + adds to temporal window
        signal = self._conv_sensor.feed(
            text=text,
            embedding=embedding,
            direction="human",
            session_id=self._session_id,
            modality="text",
            layer_depth="implementation",
        )
        if signal is None:
            return {"status": "skipped", "reason": "sensor_filtered"}

        signal_target_id = f"conv:{signal.signal_id}"

        # 2. Record to substrate — raw embedding, no classification (Law 7)
        try:
            self._eco.record_outcome(
                embedding,
                target_id=signal_target_id,
                success=True,
                metadata={
                    "source": "praxis",
                    "pheromone": "conversation",
                    "session_id": self._session_id,
                    "turn_index": self._turn_index,
                    "direction": "human",
                },
            )
        except Exception as exc:
            logger.debug("Substrate record failed: %s", exc)

        # 3. Create temporal bindings to recent signals (PRD §6.4 step 2)
        bindings_created = self._bind_temporal(
            signal, embedding, signal_target_id
        )

        # 4. Auto-checkpoint
        now = time.time()
        if now - self._last_checkpoint > self._checkpoint_interval:
            self._checkpoint()
            self._last_checkpoint = now

        return {
            "status": "captured",
            "signal_id": signal.signal_id,
            "pheromone": "conversation",
            "turn_index": self._turn_index,
            "temporal_bindings": bindings_created,
            "temporal_window_size": len(self._conv_sensor._recent),
            "autonomic_state": self._autonomic_state,
            "total_signals": self._signal_count,
        }

    def _bind_temporal(
        self,
        signal: Any,
        embedding: np.ndarray,
        signal_target_id: str,
    ) -> int:
        """Create temporal synapses between the new signal and recent ones.

        For each recent signal within the temporal window, record a
        bidirectional outcome linking the new signal's embedding to
        the recent signal's target_id, and vice versa. This creates
        the temporal associative structure in the substrate.

        Binding strength decays linearly with age:
            strength = 1.0 - (age / window_size)

        Newer signals bind more strongly. The substrate's Hebbian
        learning will further refine these weights based on what
        actually co-occurs and produces outcomes.

        Returns:
            Number of temporal bindings created.
        """
        neighbors = self._conv_sensor.get_recent_target_ids(
            exclude_signal_id=signal.signal_id,
            current_timestamp=signal.timestamp,
        )

        if not neighbors:
            return 0

        window = self._cfg.sensors.conversation.temporal_window_seconds
        bindings = 0

        for neighbor_target_id, age_seconds in neighbors:
            # Strength decays linearly with age (PRD §6.4 temporal synapses)
            strength = max(0.1, 1.0 - (age_seconds / window))

            try:
                # Forward binding: new signal → recent signal
                self._eco.record_outcome(
                    embedding,
                    target_id=neighbor_target_id,
                    success=True,
                    strength=strength,
                    metadata={
                        "source": "praxis",
                        "binding_type": "temporal",
                        "age_seconds": round(age_seconds, 2),
                    },
                )
                bindings += 1
            except Exception as exc:
                logger.debug("Temporal binding failed: %s", exc)

        self._conv_sensor.increment_bindings(bindings)
        return bindings

    def _module_stats(self) -> Dict[str, Any]:
        """Praxis-specific telemetry."""
        return {
            "session_id": self._session_id,
            "turn_index": self._turn_index,
            "signal_count": self._signal_count,
            "autonomic_state": self._autonomic_state,
            "conversation_sensor": self._conv_sensor.get_stats(),
            "sensors": {
                "conversation": {"enabled": self._cfg.sensors.conversation.enabled},
                "artifact": {"enabled": self._cfg.sensors.artifact.enabled},
                "outcome": {"enabled": self._cfg.sensors.outcome.enabled},
            },
        }

    # -----------------------------------------------------------------
    # Health endpoint
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
            "conversation_window": len(self._conv_sensor._recent),
            "autonomic_state": self._autonomic_state,
            "uptime_seconds": round(time.time() - self._start_time, 1),
        }

    # -----------------------------------------------------------------
    # Autonomic state (read-only — PRD §9)
    # -----------------------------------------------------------------

    def _read_autonomic(self) -> None:
        """Read current autonomic state. Praxis never writes."""
        try:
            import ng_autonomic
            state = ng_autonomic.read_state()
            self._autonomic_state = state.get("state", "PARASYMPATHETIC")
        except Exception:
            pass

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
        self._conv_sensor.shutdown()
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
