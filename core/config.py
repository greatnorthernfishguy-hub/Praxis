"""
Praxis Configuration — YAML Config with Dataclass Loader

Centralizes all user-configurable settings. Loads from config.yaml
with PRD-specified defaults for every value. Follows the identical
pattern to Immunis core/config.py and THC core/config.py.

Canonical source: https://github.com/greatnorthernfishguy-hub/Praxis
License: AGPL-3.0

# ---- Changelog ----
# [2026-03-19] Claude Code (Opus 4.6) — Migrate to BAAI/bge-base-en-v1.5 (#45)
# What: CPSConfig and EmbeddingConfig defaults → BAAI/bge-base-en-v1.5, dim → 768.
# Why: Ecosystem-wide embedding migration. Punchlist #45.
# How: Model string and dim default changes in two dataclasses.
# -------------------
# [2026-03-18] Claude (Opus 4.6) — Initial creation (Phase 1).
#   What: PraxisConfig dataclass with from_yaml() class method.
#         All default values match PRD §13 exactly.
#   Why:  PRD §13 specifies YAML config with dataclass loader,
#         following THC core/config.py pattern.
#   How:  Nested dataclasses for each config section. YAML loaded
#         via PyYAML, merged over defaults. Path expansion for ~.
# -------------------
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("praxis.config")


# ---------------------------------------------------------------------------
# Sensor Configuration Dataclasses (PRD §13)
# ---------------------------------------------------------------------------

@dataclass
class ConversationSensorConfig:
    enabled: bool = True
    temporal_window_seconds: float = 300.0    # Link signals within 5 minutes
    capture_both_directions: bool = True       # Capture human AND AI messages
    min_message_length: int = 10               # Ignore very short messages
    min_temporal_strength: float = 0.1        # SVG Phase 3: floor for temporal binding


@dataclass
class ArtifactSensorConfig:
    enabled: bool = True
    temporal_window_seconds: float = 3600.0    # Link signals within 1 hour
    monitor_interval_seconds: float = 60.0     # Check for changes every minute
    stale_threshold_days: int = 30             # Unreferenced artifacts decay


@dataclass
class OutcomeSensorConfig:
    enabled: bool = True
    positive_reward_strength: float = 0.7      # inject_reward strength for success
    negative_reward_strength: float = -0.5     # inject_reward strength for failure
    reward_scope: Optional[str] = None         # null = global


@dataclass
class SensorsConfig:
    conversation: ConversationSensorConfig = field(
        default_factory=ConversationSensorConfig
    )
    artifact: ArtifactSensorConfig = field(
        default_factory=ArtifactSensorConfig
    )
    outcome: OutcomeSensorConfig = field(
        default_factory=OutcomeSensorConfig
    )


# ---------------------------------------------------------------------------
# CPS Configuration (PRD §13)
# ---------------------------------------------------------------------------

@dataclass
class CPSConfig:
    storage_path: str = "~/.et_modules/praxis/cps.msgpack"
    max_entries: int = 50000
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768
    # SVG Phase 3 — substrate's concern
    search_weight_similarity: float = 0.7  # Cosine similarity blend weight
    search_weight_recency: float = 0.3     # Recency blend weight
    eviction_batch_divisor: int = 20       # Evict 1/N of max_entries per batch


# ---------------------------------------------------------------------------
# Surfacing Configuration (PRD §13)
# ---------------------------------------------------------------------------

@dataclass
class SurfacingConfig:
    max_context_items: int = 10
    min_activation_threshold: float = 0.6
    session_start_auto_surface: bool = True


# ---------------------------------------------------------------------------
# Session Bridge Configuration (PRD §13)
# ---------------------------------------------------------------------------

@dataclass
class SessionBridgeConfig:
    auto_generate_summary: bool = True
    summary_max_tokens: int = 500
    context_injection_format: str = "markdown"  # markdown | json | plain
    # SVG Phase 3 — substrate's concern
    token_char_ratio: int = 4              # Approximation: 1 token ≈ N chars
    usage_detection_threshold: float = 0.65  # Cosine similarity to detect context usage
    usage_reward_strength: float = 0.8     # Reward for surfaced context that was used


# ---------------------------------------------------------------------------
# Threshold Configuration (Ecosystem-wide — PRD §16)
# ---------------------------------------------------------------------------

@dataclass
class ThresholdsConfig:
    auto_execute: float = 0.70
    recommend: float = 0.40
    host_premium: float = 0.15


# ---------------------------------------------------------------------------
# NG-Lite Integration (PRD §13)
# ---------------------------------------------------------------------------

@dataclass
class NGLiteConfig:
    enabled: bool = True
    module_id: str = "praxis"
    state_path: str = "~/.et_modules/praxis/ng_lite_state.json"
    checkpoint_interval_seconds: int = 300


# ---------------------------------------------------------------------------
# Embedding Configuration
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingConfig:
    model: str = "BAAI/bge-base-en-v1.5"
    dim: int = 768
    device: str = "auto"
    fallback_to_hash: bool = True


# ---------------------------------------------------------------------------
# Top-Level Configuration
# ---------------------------------------------------------------------------

@dataclass
class PraxisConfig:
    """Full configuration for Praxis.

    All values have PRD-specified defaults (§13). Override via config.yaml.
    """

    sensors: SensorsConfig = field(default_factory=SensorsConfig)
    cps: CPSConfig = field(default_factory=CPSConfig)
    surfacing: SurfacingConfig = field(default_factory=SurfacingConfig)
    session_bridge: SessionBridgeConfig = field(default_factory=SessionBridgeConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    ng_lite: NGLiteConfig = field(default_factory=NGLiteConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    checkpoint_interval_seconds: int = 300

    @classmethod
    def from_yaml(cls, path: Optional[str] = None) -> "PraxisConfig":
        """Load configuration from YAML file, merging over defaults.

        Args:
            path: Path to config.yaml. Defaults to
                  ~/.et_modules/praxis/config.yaml

        Returns:
            PraxisConfig with all values populated.
        """
        if path is None:
            path = str(Path.home() / ".et_modules" / "praxis" / "config.yaml")

        config = cls()
        expanded = os.path.expanduser(path)

        if not os.path.exists(expanded):
            logger.info("No config file at %s — using defaults", expanded)
            return config

        try:
            import yaml
        except ImportError:
            logger.warning("PyYAML not installed — using defaults")
            return config

        try:
            with open(expanded, "r") as f:
                raw = yaml.safe_load(f)
        except Exception as exc:
            logger.warning("Failed to load %s: %s — using defaults", expanded, exc)
            return config

        if not raw or "praxis" not in raw:
            return config

        data = raw["praxis"]
        _apply_dict(config, data)
        return config


def _apply_dict(target: Any, source: Dict[str, Any]) -> None:
    """Recursively apply dict values to a dataclass instance."""
    if not isinstance(source, dict):
        return
    for key, val in source.items():
        if not hasattr(target, key):
            continue
        current = getattr(target, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(val, dict):
            _apply_dict(current, val)
        else:
            setattr(target, key, val)
