"""Tests for PipelineBridge — composed organism pipelines."""

from __future__ import annotations

import base64
import gzip
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_morpho_bytes() -> bytes:
    """Return plausible non-empty bytes for a fake .morpho file."""
    return b"MORPHO_FAKE_CONTENT_FOR_TESTING"


def _make_grow_result(name: str, morpho_path: str, calibrated: bool = False):
    """Return a GrowResult-like mock for CreationBridge.grow()."""
    r = MagicMock()
    r.name = name
    r.morpho_path = morpho_path
    r.behaviors = ["filter"]
    r.fitness = 0.85
    r.alive = True
    r.fingerprint = "abc123"
    r.zone_graduations = 0.9
    r.calibrated = calibrated
    return r


# ---------------------------------------------------------------------------
# TestPipelineResult
# ---------------------------------------------------------------------------

class TestPipelineResult:
    def test_pipeline_result_fields(self):
        from core.pipeline_bridge import PipelineResult

        result = PipelineResult(
            pipeline_path="/tmp/test.pipeline",
            name="filter → classify",
            stage_names=["filter_noise", "classify_signal"],
            stage_morpho_paths=["/tmp/a.morpho", "/tmp/b.morpho"],
            stage_behaviors=[["filter"], ["classify"]],
            stage_fitnesses=[0.85, 0.92],
            calibrated=True,
        )

        assert result.pipeline_path == "/tmp/test.pipeline"
        assert result.name == "filter → classify"
        assert result.stage_names == ["filter_noise", "classify_signal"]
        assert result.stage_morpho_paths == ["/tmp/a.morpho", "/tmp/b.morpho"]
        assert result.stage_behaviors == [["filter"], ["classify"]]
        assert result.stage_fitnesses == [0.85, 0.92]
        assert result.calibrated is True

    def test_pipeline_result_calibrated_defaults_false(self):
        from core.pipeline_bridge import PipelineResult

        result = PipelineResult(
            pipeline_path="/tmp/test.pipeline",
            name="test",
            stage_names=["a"],
            stage_morpho_paths=["/tmp/a.morpho"],
            stage_behaviors=[["filter"]],
            stage_fitnesses=[0.8],
        )
        assert result.calibrated is False


# ---------------------------------------------------------------------------
# TestOrganismPipeline
# ---------------------------------------------------------------------------

class TestOrganismPipeline:
    def _make_runtime(self, process_return=None):
        rt = MagicMock()
        rt.process.return_value = (
            process_return if process_return is not None
            else np.array([0.1, 0.2, 0.3])
        )
        return rt

    def test_run_single_stage_no_decoder(self):
        from core.pipeline_bridge import OrganismPipeline

        rt = self._make_runtime(np.array([0.5]))
        pipeline = OrganismPipeline(stages=[(rt, None)], names=["only"])

        result = pipeline.run(np.array([1.0]))

        rt.process.assert_called_once()
        assert result is not None

    def test_run_single_stage_with_decoder(self):
        from core.pipeline_bridge import OrganismPipeline

        rt = self._make_runtime()
        decoder = MagicMock()
        decoded_output = MagicMock(score=0.9)
        rt.predict.return_value = decoded_output
        pipeline = OrganismPipeline(stages=[(rt, decoder)], names=["only"])

        result = pipeline.run(np.array([1.0]))

        rt.predict.assert_called_once()
        assert result is decoded_output

    def test_run_two_stages_intermediate_uses_process(self):
        from core.pipeline_bridge import OrganismPipeline

        intermediate_out = np.array([0.2, 0.3])
        rt1 = self._make_runtime(intermediate_out)
        rt2 = self._make_runtime(np.array([0.9]))
        pipeline = OrganismPipeline(stages=[(rt1, None), (rt2, None)], names=["a", "b"])

        pipeline.run(np.array([1.0]))

        rt1.process.assert_called_once()
        # rt2 receives the output of rt1
        rt2.process.assert_called_once_with(intermediate_out)

    def test_run_two_stages_last_uses_decoder(self):
        from core.pipeline_bridge import OrganismPipeline

        intermediate_out = np.array([0.2, 0.3])
        rt1 = self._make_runtime(intermediate_out)
        rt2 = MagicMock()
        decoder2 = MagicMock()
        final_output = MagicMock(score=0.88)
        rt2.predict.return_value = final_output
        pipeline = OrganismPipeline(stages=[(rt1, None), (rt2, decoder2)], names=["a", "b"])

        result = pipeline.run(np.array([1.0]))

        rt1.process.assert_called_once()
        rt2.predict.assert_called_once_with(intermediate_out, decoder2)
        assert result is final_output

    def test_start_calls_all_runtimes(self):
        from core.pipeline_bridge import OrganismPipeline

        rt1, rt2 = MagicMock(), MagicMock()
        pipeline = OrganismPipeline(stages=[(rt1, None), (rt2, None)], names=["a", "b"])
        pipeline.start()

        rt1.start.assert_called_once()
        rt2.start.assert_called_once()

    def test_stop_calls_all_runtimes(self):
        from core.pipeline_bridge import OrganismPipeline

        rt1, rt2 = MagicMock(), MagicMock()
        pipeline = OrganismPipeline(stages=[(rt1, None), (rt2, None)], names=["a", "b"])
        pipeline.stop()

        rt1.stop.assert_called_once()
        rt2.stop.assert_called_once()

    def test_run_empty_stages_raises(self):
        from core.pipeline_bridge import OrganismPipeline
        with pytest.raises(ValueError, match="requires at least one stage"):
            OrganismPipeline(stages=[], names=[])


# ---------------------------------------------------------------------------
# TestPipelineBridgeStub
# ---------------------------------------------------------------------------

class TestPipelineBridgeStub:
    def test_grow_pipeline_raises_not_implemented(self):
        """Stub raises NotImplementedError — implementation comes in Task 2."""
        import core.pipeline_bridge as pb
        with patch.object(pb, "_MORPHOGENESIS_AVAILABLE", True):
            from core.pipeline_bridge import PipelineBridge
            bridge = PipelineBridge()
            with pytest.raises(NotImplementedError):
                bridge.grow_pipeline(["filter noise"])

    def test_load_pipeline_raises_not_implemented(self):
        import core.pipeline_bridge as pb
        with patch.object(pb, "_MORPHOGENESIS_AVAILABLE", True):
            from core.pipeline_bridge import PipelineBridge
            bridge = PipelineBridge()
            with pytest.raises(NotImplementedError):
                bridge.load_pipeline("/tmp/fake.pipeline")

    def test_init_raises_when_morphogenesis_unavailable(self):
        import core.pipeline_bridge as pb
        with patch.object(pb, "_MORPHOGENESIS_AVAILABLE", False):
            from core.pipeline_bridge import PipelineBridge
            with pytest.raises(ImportError, match="morphogenesis is not installed"):
                PipelineBridge()
