"""Tests for Praxis configuration."""

import sys
import os

# Ensure repo root and vendored dir are on path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vendored = os.path.join(repo_root, "vendored")
for p in [repo_root, vendored]:
    if p not in sys.path:
        sys.path.insert(0, p)

from core.config import (
    PraxisConfig,
    ConversationSensorConfig,
    ArtifactSensorConfig,
    OutcomeSensorConfig,
    CPSConfig,
    SurfacingConfig,
    SessionBridgeConfig,
    ThresholdsConfig,
    NGLiteConfig,
    EmbeddingConfig,
)


def test_defaults():
    """All defaults match PRD §13."""
    cfg = PraxisConfig()

    # Conversation sensor
    assert cfg.sensors.conversation.temporal_window_seconds == 300.0
    assert cfg.sensors.conversation.capture_both_directions is True
    assert cfg.sensors.conversation.min_message_length == 10

    # Artifact sensor
    assert cfg.sensors.artifact.temporal_window_seconds == 3600.0
    assert cfg.sensors.artifact.monitor_interval_seconds == 60.0
    assert cfg.sensors.artifact.stale_threshold_days == 30

    # Outcome sensor
    assert cfg.sensors.outcome.positive_reward_strength == 0.7
    assert cfg.sensors.outcome.negative_reward_strength == -0.5

    # CPS
    assert cfg.cps.max_entries == 50000
    assert cfg.cps.embedding_dim == 384

    # Surfacing
    assert cfg.surfacing.max_context_items == 10
    assert cfg.surfacing.min_activation_threshold == 0.6

    # Session bridge
    assert cfg.session_bridge.auto_generate_summary is True
    assert cfg.session_bridge.context_injection_format == "markdown"

    # Ecosystem thresholds
    assert cfg.thresholds.auto_execute == 0.70
    assert cfg.thresholds.recommend == 0.40
    assert cfg.thresholds.host_premium == 0.15

    # NG-Lite
    assert cfg.ng_lite.module_id == "praxis"
    assert cfg.ng_lite.checkpoint_interval_seconds == 300

    # Embedding
    assert cfg.embedding.dim == 384
    assert cfg.embedding.fallback_to_hash is True


def test_from_yaml_missing_file():
    """Missing config file returns defaults."""
    cfg = PraxisConfig.from_yaml("/nonexistent/path/config.yaml")
    assert cfg.sensors.conversation.temporal_window_seconds == 300.0


def test_from_yaml_with_overrides(tmp_path):
    """YAML values override defaults."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "praxis:\n"
        "  sensors:\n"
        "    conversation:\n"
        "      min_message_length: 20\n"
        "  cps:\n"
        "    max_entries: 10000\n"
    )

    cfg = PraxisConfig.from_yaml(str(config_file))
    assert cfg.sensors.conversation.min_message_length == 20
    assert cfg.cps.max_entries == 10000
    # Non-overridden values stay default
    assert cfg.sensors.conversation.temporal_window_seconds == 300.0


if __name__ == "__main__":
    test_defaults()
    print("test_defaults PASSED")

    test_from_yaml_missing_file()
    print("test_from_yaml_missing_file PASSED")

    # tmp_path equivalent for manual runs
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        test_from_yaml_with_overrides(Path(td))
    print("test_from_yaml_with_overrides PASSED")

    print("\nAll config tests passed.")
