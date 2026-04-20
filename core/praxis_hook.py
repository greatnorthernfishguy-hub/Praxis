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
# [2026-04-19] Claude Code — #5: replace dead eco drain with _drain_river() + _on_river_events()
#   What: _pulse_cycle() now calls _drain_river(); event routing moved to _on_river_events() override
#   Why: #5 — eco._peer_bridge was dead (SKIP_ECOSYSTEM=True → _eco=None); BTF drain is in base class
#   How: _on_river_events() receives new BTF events; each routed through _route_pulse_event()
# [2026-03-28] Claude Code (Opus 4.6) — #109 Pulse loop: drain outcomes between conversations
#   What: Added _pulse_loop() daemon thread following the Tonic pattern.
#   Why:  #109 — organs must run continuously. Praxis needs to absorb outcome
#         and artifact signals from peer modules via the River between
#         conversations, not only during fan-out messages.
#   How:  _shutdown_event + _in_conversation flag. Daemon thread drains
#         River tracts via _eco._peer_bridge.drain(), feeds absorbed events
#         to outcome sensor or artifact sensor based on metadata. Resting 60s /
#         conversation 10s intervals. on_conversation_started/ended swap
#         intervals. Does not touch _module_on_message.
# [2026-03-19] Claude Code (Opus 4.6) — Migrate to BAAI/bge-base-en-v1.5 (#45)
# What: fastembed model all-MiniLM-L6-v2 → BAAI/bge-base-en-v1.5 (768-dim).
#   Removed sentence-transformers fallback entirely — fastembed → hash only.
# Why: Ecosystem-wide embedding migration. sentence-transformers broke and
#   caused the cascade that led to 384-dim contamination. Punchlist #45.
# How: TextEmbedding() model string, removed ST fallback block, updated docstring.
# -------------------
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
# [2026-03-18] Claude (Opus 4.6) — Phase 3: CPS integration.
#   What: Initialize CPS in __init__. Store conversation entries in CPS.
#   Why:  PRD §8 — CPS stores intent-behavior mappings.
#   How:  CPS initialized with ecosystem ref. Conversation → INTENT entries.
# [2026-03-18] Claude (Opus 4.6) — Phase 4: Session Bridge integration.
#   What: Initialize SessionBridge. Auto-surface context on first message.
#         Track usage of surfaced context. Session end/start lifecycle.
#   Why:  PRD §8.4 — solves context death. PRD §14 Phase 4.
#   How:  First message triggers surface_context() → format → return in
#         result dict. Subsequent messages track_usage() against surfaced
#         items. end_session() generates SESSION_SUMMARY. start_session()
#         resets turn tracking.
# [2026-04-15] Claude Code (Sonnet 4.6) — Punchlist #137: Fix bridge.drain() no-op
#   What: Replace bridge.drain() with bridge.sync_state() in _pulse_cycle().
#   Why:  NGTractBridge has no public drain() method. hasattr guard silently
#         skipped the entire tract drain — pulse loop was a no-op for River.
#   How:  sync_state(local_state={}, module_id=MODULE_ID) calls _drain_all()
#         internally, clearing incoming tracts and updating bridge state.
#         Event routing removed — Law 7 violation deferred to punchlist #154.

# [2026-04-19] Claude Code (Sonnet 4.6) — Add record_conversation() standalone public API
#   What: record_conversation(text, direction, layer_depth) — public method symmetric with
#         record_artifact() and record_outcome(). Feeds text directly to conv sensor + CPS.
#   Why: _module_on_message is now a no-op (River path). Tests and standalone integrations
#        (Morphogenesis) need a direct conversation ingestion path without the River.
#   How: Embeds text, calls _conv_sensor.feed(), stores to CPS, runs temporal binding.
#        Returns {"status": "captured", "signal_id": ..., "temporal_bindings": N}.
# [2026-04-18] Claude Code (Sonnet 4.6) — Punchlist #154: Wire _route_pulse_event into _pulse_cycle
#   What: _pulse_cycle now routes each new River event through _route_pulse_event after sync_state.
#   Why:  #137 fixed the drain no-op but deferred event routing. _route_pulse_event was correctly
#         written (raw embeddings, pheromone routing, Law 7 compliant) but never called from
#         the pulse loop. Events were drained and silently discarded.
#   How:  Snapshot len(bridge._peer_events) before sync_state, iterate peer_events[before:],
#         call _route_pulse_event(event) per new event. Error-isolated per event.
# -------------------
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from openclaw_adapter import OpenClawAdapter

logger = logging.getLogger("praxis")


class PraxisHook(OpenClawAdapter):
    """OpenClaw integration hook for Praxis."""

    MODULE_ID = "praxis"
    SKIP_ECOSYSTEM = True
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

        # --- Initialize Artifact Sensor (PRD §4.2) ---
        from sensors.artifact import ArtifactSensor

        art_config = {
            "temporal_window_seconds": self._cfg.sensors.artifact.temporal_window_seconds,
            "stale_threshold_days": self._cfg.sensors.artifact.stale_threshold_days,
        }
        self._art_sensor = ArtifactSensor(config=art_config)
        self._art_sensor.set_log_path(self._data_dir / "artifact.jsonl")

        # --- Initialize Outcome Sensor (PRD §4.3) ---
        from sensors.outcome import OutcomeSensor

        out_config = {
            "positive_reward_strength": self._cfg.sensors.outcome.positive_reward_strength,
            "negative_reward_strength": self._cfg.sensors.outcome.negative_reward_strength,
        }
        self._out_sensor = OutcomeSensor(config=out_config)
        self._out_sensor.set_log_path(self._data_dir / "outcome.jsonl")

        # --- Initialize Context Persistence Store (PRD §8) ---
        from store.cps import ContextPersistenceStore

        cps_config = {
            "max_entries": self._cfg.cps.max_entries,
            "storage_path": self._cfg.cps.storage_path,
        }
        self._cps = ContextPersistenceStore(
            config=cps_config,
            data_dir=str(self._data_dir),
            ecosystem=self._eco,
        )

        # --- Initialize Session Bridge (PRD §8.4) ---
        from core.session_bridge import SessionBridge

        bridge_config = {
            "max_context_items": self._cfg.surfacing.max_context_items,
            "min_activation_threshold": self._cfg.surfacing.min_activation_threshold,
            "session_start_auto_surface": self._cfg.surfacing.session_start_auto_surface,
            "auto_generate_summary": self._cfg.session_bridge.auto_generate_summary,
            "summary_max_tokens": self._cfg.session_bridge.summary_max_tokens,
            "context_injection_format": self._cfg.session_bridge.context_injection_format,
        }
        self._bridge = SessionBridge(
            config=bridge_config,
            cps=self._cps,
            ecosystem=self._eco,
            embed_fn=self._embed,
        )

        # --- Session Tracking ---
        self._session_id = f"session_{int(time.time())}"
        self._previous_session_id: Optional[str] = None
        self._turn_index = 0
        self._signal_count = 0
        self._session_context: Optional[str] = None

        # --- Autonomic State (PRD §9 — read only) ---
        self._autonomic_state = "PARASYMPATHETIC"
        self._read_autonomic()

        # --- Checkpointing ---
        self._last_checkpoint = time.time()
        self._checkpoint_interval = self._cfg.checkpoint_interval_seconds

        # --- #109 Pulse loop infrastructure ---
        self._shutdown_event = threading.Event()
        self._in_conversation = False
        self._resting_interval = 60.0
        self._conversation_interval = 10.0

        self._pulse_thread = threading.Thread(
            target=self._pulse_loop, name="praxis-pulse", daemon=True
        )
        self._pulse_thread.start()

        logger.info(
            "[Praxis] Initialized — session=%s, autonomic=%s, tier=%d, pulse=%.0fs/%.0fs",
            self._session_id,
            self._autonomic_state,
            self._eco.tier if self._eco else 0,
            self._resting_interval,
            self._conversation_interval,
        )

    # -----------------------------------------------------------------
    # #109 Pulse loop — drain outcomes between conversations
    # -----------------------------------------------------------------

    def _pulse_loop(self):
        """Continuous outcome absorption — Praxis alive between conversations.

        Follows the Tonic pattern: daemon thread with shutdown event wait.
        Each cycle drains River tracts for outcome/artifact signals from
        peer modules and feeds them to the appropriate sensors.
        """
        while not self._shutdown_event.is_set():
            try:
                self._pulse_cycle()
            except Exception as exc:
                logger.debug("Pulse cycle error: %s", exc)
            interval = (
                self._conversation_interval
                if self._in_conversation
                else self._resting_interval
            )
            self._shutdown_event.wait(timeout=interval)

    def _pulse_cycle(self):
        """One pulse cycle — drain River tracts, route to sensors (raw embeddings, Law 7)."""
        self._drain_river()

    def _on_river_events(self, events: list) -> None:
        """Route new River events to domain sensors via _route_pulse_event."""
        for event in events:
            try:
                self._route_pulse_event(event)
            except Exception as exc:
                logger.debug("Pulse route error: %s", exc)
    def _route_pulse_event(self, event):
        """Route a drained River event to the appropriate sensor.

        Outcome events go to the outcome sensor. Artifact events go to
        the artifact sensor. Everything else is recorded as raw experience
        on the substrate (Law 7).
        """
        if not isinstance(event, dict):
            return

        metadata = event.get('metadata', {})
        pheromone = metadata.get('pheromone', '')
        embedding = event.get('embedding')

        # Need an embedding to do anything useful
        if embedding is None:
            return

        try:
            embedding = np.asarray(embedding, dtype=np.float32)
        except Exception:
            return

        if pheromone == 'outcome':
            # Feed to outcome sensor
            try:
                self._out_sensor.record_outcome(
                    embedding=embedding,
                    outcome_type=metadata.get('outcome_type', 'unknown'),
                    success=metadata.get('success', True),
                    severity=metadata.get('severity', 0.5),
                    session_id=self._session_id,
                    layer_depth=metadata.get('layer_depth', 'implementation'),
                    metadata={'source': 'river_pulse'},
                )
            except Exception as exc:
                logger.debug("Pulse outcome sensor error: %s", exc)

        elif pheromone == 'artifact':
            # Feed to artifact sensor
            try:
                artifact_id = metadata.get('artifact_id', 'unknown')
                event_type = metadata.get('event_type', 'reference')
                self._art_sensor.record_reference(
                    artifact_id, embedding, self._session_id,
                    metadata.get('layer_depth', 'implementation'),
                )
            except Exception as exc:
                logger.debug("Pulse artifact sensor error: %s", exc)

        elif event.get('type') == 'topology_delta' and event.get('conversation'):
            # Conversation content from the central substrate
            try:
                self._ingest_conversation(event['conversation'], embedding)
            except Exception as exc:
                logger.debug("Pulse conversation ingest error: %s", exc)

        else:
            # Raw experience to substrate (Law 7)
            try:
                target_id = metadata.get('target_id', f"pulse:{time.time():.0f}")
                if self._eco:
                    if self._eco: self._eco.record_outcome(
                        embedding,
                        target_id=target_id,
                        success=True,
                        metadata={'source': 'river_pulse', 'pheromone': pheromone},
                    )
            except Exception as exc:
                logger.debug("Pulse substrate record error: %s", exc)

    def _ingest_conversation(self, conversation: dict, embedding: np.ndarray) -> None:
        """Process conversation content from a topology delta.

        Same flow as the old _module_on_message but triggered by River
        tract drain instead of fan-out push.
        """
        text = conversation.get('text', '')
        conv_emb = conversation.get('embedding')
        if conv_emb is not None:
            embedding = np.asarray(conv_emb, dtype=np.float32)

        if not text or len(text) < self._cfg.sensors.conversation.min_message_length:
            return

        self._turn_index += 1
        self._signal_count += 1

        # Feed to ConversationSensor
        signal = self._conv_sensor.feed(
            text=text,
            embedding=embedding,
            direction="human",
            session_id=self._session_id,
            modality="text",
            layer_depth="implementation",
        )
        if signal is None:
            return

        signal_target_id = f"conv:{signal.signal_id}"

        # Dual-pass record to local substrate
        if self._eco:
            try:
                if self._eco: self._eco.dual_record_outcome(
                    content=text,
                    embedding=embedding,
                    target_id=signal_target_id,
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

        # Store in CPS
        try:
            self._cps.store_from_signal(
                content=text,
                embedding=embedding,
                entry_type="INTENT",
                pheromone_source="conversation",
                session_id=self._session_id,
                layer_depth="implementation",
                metadata={"turn_index": self._turn_index, "signal_id": signal.signal_id},
            )
        except Exception as exc:
            logger.debug("CPS store failed: %s", exc)

        # Temporal bindings
        self._bind_temporal(signal, embedding, signal_target_id)

        # Auto-checkpoint
        now = time.time()
        if now - self._last_checkpoint > self._checkpoint_interval:
            self._checkpoint()
            self._last_checkpoint = now

    # -----------------------------------------------------------------
    # Conversation recording (PRD §4.1 — standalone/dev mode)
    # -----------------------------------------------------------------

    def record_conversation(
        self,
        text: str,
        direction: str = "human",
        layer_depth: str = "implementation",
    ) -> Dict[str, Any]:
        """Record a conversation turn directly (standalone/dev mode).

        Public API for feeding conversation text without the River —
        symmetric with record_artifact() and record_outcome().
        Used in testing and for direct integration (e.g. Morphogenesis).

        In production on the live organism, conversation arrives via
        topology delta in the River (_ingest_conversation). This method
        provides the same processing for contexts without a River.
        """
        if not text or len(text) < self._cfg.sensors.conversation.min_message_length:
            return {"status": "skipped", "reason": "too_short"}

        embedding = self._embed(text)

        self._turn_index += 1
        self._signal_count += 1

        signal = self._conv_sensor.feed(
            text=text,
            embedding=embedding,
            direction=direction,
            session_id=self._session_id,
            modality="text",
            layer_depth=layer_depth,
        )
        if signal is None:
            return {"status": "skipped", "reason": "sensor_rejected"}

        signal_target_id = f"conv:{signal.signal_id}"

        if self._eco:
            try:
                self._eco.dual_record_outcome(
                    content=text,
                    embedding=embedding,
                    target_id=signal_target_id,
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

        try:
            self._cps.store_from_signal(
                content=text,
                embedding=embedding,
                entry_type="INTENT",
                pheromone_source="conversation",
                session_id=self._session_id,
                layer_depth=layer_depth,
                metadata={"turn_index": self._turn_index, "signal_id": signal.signal_id},
            )
        except Exception as exc:
            logger.debug("CPS store failed: %s", exc)

        bindings = self._bind_temporal(signal, embedding, signal_target_id)

        return {
            "status": "captured",
            "signal_id": signal.signal_id,
            "temporal_bindings": bindings,
        }

    def on_conversation_started(self):
        """Mode swap: faster pulse during active conversation."""
        self._in_conversation = True
        logger.debug("[Praxis] conversation started — pulse interval %.0fs",
                      self._conversation_interval)

    def on_conversation_ended(self):
        """Mode swap: slower pulse between conversations."""
        self._in_conversation = False
        logger.debug("[Praxis] conversation ended — pulse interval %.0fs",
                      self._resting_interval)

    # -----------------------------------------------------------------
    # OpenClawAdapter implementation
    # -----------------------------------------------------------------

    def _embed(self, text: str) -> np.ndarray:
        """Embed text via ng_embed (centralized ecosystem embedding).

        Ecosystem standard: Snowflake/snowflake-arctic-embed-m-v1.5 (768-dim).
        ONNX Runtime, no torch dependency. L2-normalized for Praxis.
        """
        if self._cfg.embedding.device != "disabled":
            try:
                from ng_embed import embed
                return embed(text, normalize=True)
            except Exception:
                pass

        return self._hash_embed(text)

    def _module_on_message(
        self, text: str, embedding: np.ndarray
    ) -> Dict[str, Any]:
        """No-op — conversation content arrives via topology delta in the River.

        The central substrate deposits text+embedding in the topology delta
        (version 2). Praxis pulse cycle extracts and processes it via
        _route_pulse_event -> _ingest_conversation.
        """
        return {}

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
                if self._eco: self._eco.record_outcome(
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

    # -----------------------------------------------------------------
    # Artifact recording (PRD §4.2 — Phase 5)
    # -----------------------------------------------------------------

    def record_artifact(
        self,
        artifact_id: str,
        content: str,
        artifact_type: str = "other",
        event_type: str = "reference",
        layer_depth: str = "implementation",
    ) -> Dict[str, Any]:
        """Record an artifact lifecycle event.

        Called externally when artifacts are created, modified, referenced,
        or deleted. Embeds the content, records to sensor, substrate, and CPS.
        """
        embedding = self._embed(content)

        if event_type == "delete":
            signal = self._art_sensor.record_delete(
                artifact_id, embedding, self._session_id
            )
        elif event_type in ("create", "modify"):
            import hashlib
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
            signal = self._art_sensor.register_artifact(
                artifact_id=artifact_id,
                embedding=embedding,
                artifact_type=artifact_type,
                content_hash=content_hash,
                session_id=self._session_id,
                layer_depth=layer_depth,
            )
        else:
            signal = self._art_sensor.record_reference(
                artifact_id, embedding, self._session_id, layer_depth
            )

        if signal is None:
            return {"status": "skipped", "artifact_id": artifact_id}

        self._signal_count += 1

        # Record to substrate
        try:
            if self._eco: self._eco.record_outcome(
                embedding,
                target_id=f"artifact:{artifact_id}",
                success=True,
                metadata={
                    "source": "praxis",
                    "pheromone": "artifact",
                    "event_type": event_type,
                    "artifact_type": artifact_type,
                },
            )
        except Exception as exc:
            logger.debug("Artifact substrate record failed: %s", exc)

        # Store in CPS
        try:
            self._cps.store_from_signal(
                content=content[:500],
                embedding=embedding,
                entry_type="ARTIFACT",
                pheromone_source="artifact",
                session_id=self._session_id,
                layer_depth=layer_depth,
                metadata={
                    "artifact_id": artifact_id,
                    "event_type": event_type,
                    "artifact_type": artifact_type,
                },
            )
        except Exception as exc:
            logger.debug("Artifact CPS store failed: %s", exc)

        return {
            "status": "recorded",
            "signal_id": signal.signal_id,
            "artifact_id": artifact_id,
            "event_type": event_type,
        }

    # -----------------------------------------------------------------
    # Outcome recording (PRD §4.3 — Phase 6)
    # -----------------------------------------------------------------

    def record_outcome(
        self,
        context: str,
        outcome_type: str,
        success: bool,
        severity: float = 0.5,
        related_intent_ids: Optional[List[str]] = None,
        layer_depth: str = "implementation",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record an outcome event with reward signal.

        Called externally when build/test/review/deploy outcomes arrive.
        Embeds the context, records to sensor (which computes reward
        strength), then records to substrate and CPS.
        """
        embedding = self._embed(context)

        signal, reward_strength = self._out_sensor.record_outcome(
            embedding=embedding,
            outcome_type=outcome_type,
            success=success,
            severity=severity,
            related_intent_ids=related_intent_ids,
            session_id=self._session_id,
            layer_depth=layer_depth,
            metadata=metadata,
        )

        self._signal_count += 1

        # Record to substrate with reward strength
        try:
            if self._eco: self._eco.record_outcome(
                embedding,
                target_id=f"outcome:{signal.signal_id}",
                success=success,
                strength=abs(reward_strength),
                metadata={
                    "source": "praxis",
                    "pheromone": "outcome",
                    "outcome_type": outcome_type,
                    "reward_strength": round(reward_strength, 4),
                },
            )
        except Exception as exc:
            logger.debug("Outcome substrate record failed: %s", exc)

        # Store in CPS
        try:
            self._cps.store_from_signal(
                content=context[:500],
                embedding=embedding,
                entry_type="OUTCOME",
                pheromone_source="outcome",
                session_id=self._session_id,
                layer_depth=layer_depth,
                outcome_signal=reward_strength,
                metadata={
                    "outcome_type": outcome_type,
                    "success": success,
                    "severity": severity,
                    "signal_id": signal.signal_id,
                },
            )
        except Exception as exc:
            logger.debug("Outcome CPS store failed: %s", exc)

        return {
            "status": "recorded",
            "signal_id": signal.signal_id,
            "outcome_type": outcome_type,
            "success": success,
            "reward_strength": round(reward_strength, 4),
        }

    def _module_stats(self) -> Dict[str, Any]:
        """Praxis-specific telemetry."""
        return {
            "session_id": self._session_id,
            "turn_index": self._turn_index,
            "signal_count": self._signal_count,
            "autonomic_state": self._autonomic_state,
            "conversation_sensor": self._conv_sensor.get_stats(),
            "artifact_sensor": self._art_sensor.get_stats(),
            "outcome_sensor": self._out_sensor.get_stats(),
            "cps": self._cps.get_stats(),
        }

    # -----------------------------------------------------------------
    # Health endpoint
    # -----------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        """Basic health check for Praxis."""
        return {
            "module": "praxis",
            "status": "healthy",
            "tier": self._eco.tier if self._eco else 0,
            "tier_name": self._eco.tier_name if self._eco else "tract-only",
            "session_id": self._session_id,
            "signal_count": self._signal_count,
            "cps_entries": self._cps.count,
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
        """Save ecosystem and CPS state."""
        try:
            pass  # state via tracts
        except Exception as exc:
            logger.debug("Ecosystem checkpoint failed: %s", exc)
        try:
            self._cps.save()
        except Exception as exc:
            logger.debug("CPS checkpoint failed: %s", exc)

    # -----------------------------------------------------------------
    # Session lifecycle (PRD §8.4)
    # -----------------------------------------------------------------

    def end_session(self) -> Optional[str]:
        """End the current session — generate summary, checkpoint.

        Called when a session ends. Generates a SESSION_SUMMARY entry
        in the CPS from the session's accumulated context.

        Returns:
            Summary text, or None if session was empty.
        """
        summary = None
        if self._cfg.session_bridge.auto_generate_summary:
            summary = self._bridge.generate_summary(self._session_id)

        self._previous_session_id = self._session_id
        self._checkpoint()
        logger.info(
            "[Praxis] Session %s ended — %d signals, %d CPS entries",
            self._session_id, self._signal_count, self._cps.count,
        )
        return summary

    def start_session(self) -> str:
        """Start a new session, preserving state from the previous one.

        Returns:
            The new session ID.
        """
        self._session_id = f"session_{int(time.time())}"
        self._turn_index = 0
        self._session_context = None
        logger.info("[Praxis] New session started: %s", self._session_id)
        return self._session_id

    # -----------------------------------------------------------------
    # Shutdown
    # -----------------------------------------------------------------

    def shutdown(self) -> None:
        """Graceful shutdown — stop pulse, end session, checkpoint."""
        self._shutdown_event.set()
        self.end_session()
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
