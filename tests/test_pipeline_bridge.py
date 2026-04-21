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
    def test_load_pipeline_raises_on_missing_file(self):
        import core.pipeline_bridge as pb
        with patch.object(pb, "_MORPHOGENESIS_AVAILABLE", True):
            from core.pipeline_bridge import PipelineBridge
            bridge = PipelineBridge()
            with pytest.raises((FileNotFoundError, OSError)):
                bridge.load_pipeline("/tmp/nonexistent_pipeline_xyz.pipeline")

    def test_init_raises_when_morphogenesis_unavailable(self):
        import core.pipeline_bridge as pb
        with patch.object(pb, "_MORPHOGENESIS_AVAILABLE", False):
            from core.pipeline_bridge import PipelineBridge
            with pytest.raises(ImportError, match="morphogenesis is not installed"):
                PipelineBridge()


# ---------------------------------------------------------------------------
# TestPipelineBridgeGrow
# ---------------------------------------------------------------------------

class TestPipelineBridgeGrow:
    """Tests for PipelineBridge.grow_pipeline()."""

    def _make_grow_result(self, name: str, morpho_path: str, calibrated: bool = False):
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

    def _setup_pipeline_bridge(self, tmp_path):
        """Return a patched PipelineBridge and write fake .morpho files."""
        import core.pipeline_bridge as pb

        morpho_a = str(tmp_path / "stage_a_42.morpho")
        morpho_b = str(tmp_path / "stage_b_99.morpho")
        Path(morpho_a).write_bytes(b"MORPHO_A_BYTES")
        Path(morpho_b).write_bytes(b"MORPHO_B_BYTES")

        result_a = self._make_grow_result("stage_a", morpho_a, calibrated=False)
        result_b = self._make_grow_result("stage_b", morpho_b, calibrated=True)

        mock_bridge = MagicMock()
        mock_bridge.grow.side_effect = [result_a, result_b]

        return mock_bridge, result_a, result_b

    def test_grow_pipeline_two_stages_returns_pipeline_result(self, tmp_path):
        import core.pipeline_bridge as pb

        mock_bridge, result_a, result_b = self._setup_pipeline_bridge(tmp_path)

        with patch.object(pb, "_MORPHOGENESIS_AVAILABLE", True), \
             patch("core.pipeline_bridge.CreationBridge", return_value=mock_bridge):
            from core.pipeline_bridge import PipelineBridge
            bridge = PipelineBridge()
            result = bridge.grow_pipeline(
                ["filter noise", "classify signal"],
                seeds=[42, 99],
                output_dir=str(tmp_path),
            )

        assert isinstance(result, pb.PipelineResult)
        assert result.stage_names == ["stage_a", "stage_b"]
        assert result.stage_morpho_paths == [str(tmp_path / "stage_a_42.morpho"), str(tmp_path / "stage_b_99.morpho")]
        assert result.pipeline_path.endswith(".pipeline")
        assert result.calibrated is True

    def test_grow_pipeline_only_calibrates_last_stage(self, tmp_path):
        import core.pipeline_bridge as pb

        mock_bridge, result_a, result_b = self._setup_pipeline_bridge(tmp_path)
        normal_data = [[1.0, 2.0]]

        with patch.object(pb, "_MORPHOGENESIS_AVAILABLE", True), \
             patch("core.pipeline_bridge.CreationBridge", return_value=mock_bridge):
            from core.pipeline_bridge import PipelineBridge
            bridge = PipelineBridge()
            bridge.grow_pipeline(
                ["filter noise", "classify signal"],
                seeds=[42, 99],
                output_dir=str(tmp_path),
                normal_examples=normal_data,
            )

        calls = mock_bridge.grow.call_args_list
        assert len(calls) == 2
        # Stage 0 — no calibration args
        _, kwargs0 = calls[0]
        assert kwargs0["normal_examples"] is None
        assert kwargs0["anomaly_examples"] is None
        # Stage 1 (last) — calibration args passed
        _, kwargs1 = calls[1]
        assert kwargs1["normal_examples"] is normal_data

    def test_grow_pipeline_pipeline_file_is_gzip_json(self, tmp_path):
        import core.pipeline_bridge as pb

        mock_bridge, _, _ = self._setup_pipeline_bridge(tmp_path)

        with patch.object(pb, "_MORPHOGENESIS_AVAILABLE", True), \
             patch("core.pipeline_bridge.CreationBridge", return_value=mock_bridge):
            from core.pipeline_bridge import PipelineBridge
            bridge = PipelineBridge()
            result = bridge.grow_pipeline(
                ["filter noise", "classify signal"],
                seeds=[42, 99],
                output_dir=str(tmp_path),
            )

        with gzip.open(result.pipeline_path, "rb") as fh:
            payload = json.loads(fh.read())

        assert payload["version"] == "1.0"
        assert len(payload["stages"]) == 2
        assert payload["stages"][0]["name"] == "stage_a"
        # morpho bytes are embedded as base64
        decoded = base64.b64decode(payload["stages"][0]["morpho_b64"])
        assert decoded == b"MORPHO_A_BYTES"

    def test_grow_pipeline_empty_descriptions_raises(self, tmp_path):
        import core.pipeline_bridge as pb

        with patch.object(pb, "_MORPHOGENESIS_AVAILABLE", True):
            from core.pipeline_bridge import PipelineBridge
            bridge = PipelineBridge()
            with pytest.raises(ValueError, match="descriptions must be non-empty"):
                bridge.grow_pipeline([])

    def test_grow_pipeline_seeds_length_mismatch_raises(self, tmp_path):
        import core.pipeline_bridge as pb

        with patch.object(pb, "_MORPHOGENESIS_AVAILABLE", True):
            from core.pipeline_bridge import PipelineBridge
            bridge = PipelineBridge()
            with pytest.raises(ValueError, match="seeds length"):
                bridge.grow_pipeline(
                    ["filter", "classify"],
                    seeds=[42],  # wrong length
                )

    def test_grow_pipeline_random_seeds_when_omitted(self, tmp_path):
        import core.pipeline_bridge as pb

        mock_bridge, _, _ = self._setup_pipeline_bridge(tmp_path)

        with patch.object(pb, "_MORPHOGENESIS_AVAILABLE", True), \
             patch("core.pipeline_bridge.CreationBridge", return_value=mock_bridge):
            from core.pipeline_bridge import PipelineBridge
            bridge = PipelineBridge()
            result = bridge.grow_pipeline(
                ["filter noise", "classify signal"],
                output_dir=str(tmp_path),
            )

        # Should not raise — seeds were auto-generated
        assert result.pipeline_path.endswith(".pipeline")

    def test_grow_pipeline_custom_name(self, tmp_path):
        import core.pipeline_bridge as pb

        mock_bridge, _, _ = self._setup_pipeline_bridge(tmp_path)

        with patch.object(pb, "_MORPHOGENESIS_AVAILABLE", True), \
             patch("core.pipeline_bridge.CreationBridge", return_value=mock_bridge):
            from core.pipeline_bridge import PipelineBridge
            bridge = PipelineBridge()
            result = bridge.grow_pipeline(
                ["filter noise", "classify signal"],
                seeds=[42, 99],
                output_dir=str(tmp_path),
                name="my_custom_pipeline",
            )

        assert result.name == "my_custom_pipeline"


# ---------------------------------------------------------------------------
# TestPipelineBridgeLoad
# ---------------------------------------------------------------------------

class TestPipelineBridgeLoad:
    """Tests for PipelineBridge.load_pipeline()."""

    def _write_fake_pipeline(self, tmp_path, n_stages: int = 2) -> str:
        """Write a minimal .pipeline file with fake .morpho bytes."""
        stages = []
        for i in range(n_stages):
            stages.append({
                "name": f"stage_{i}",
                "morpho_b64": base64.b64encode(f"MORPHO_{i}".encode()).decode("ascii"),
                "behaviors": ["filter"],
                "fitness": 0.8,
                "calibrated": (i == n_stages - 1),
            })
        payload = {
            "version": "1.0",
            "name": "test_pipeline",
            "created_at": 1234567890.0,
            "stages": stages,
        }
        pipeline_path = str(tmp_path / "test.pipeline")
        with gzip.open(pipeline_path, "wb") as fh:
            fh.write(json.dumps(payload).encode("utf-8"))
        return pipeline_path

    def test_load_pipeline_returns_organism_pipeline(self, tmp_path):
        import core.pipeline_bridge as pb
        from core.pipeline_bridge import OrganismPipeline

        pipeline_path = self._write_fake_pipeline(tmp_path, n_stages=2)
        mock_runtime = MagicMock()
        mock_decoder = MagicMock()

        # instantiate_morpho returns (organism, decoder, runtime) per stage
        mock_instantiate = MagicMock(side_effect=[
            (MagicMock(), None, mock_runtime),
            (MagicMock(), mock_decoder, mock_runtime),
        ])

        with patch.object(pb, "_MORPHOGENESIS_AVAILABLE", True), \
             patch.object(pb, "load_morpho", return_value=MagicMock()), \
             patch.object(pb, "instantiate_morpho", mock_instantiate):
            from core.pipeline_bridge import PipelineBridge
            bridge = PipelineBridge()
            pipeline = bridge.load_pipeline(pipeline_path)

        assert isinstance(pipeline, OrganismPipeline)

    def test_load_pipeline_stage_count(self, tmp_path):
        import core.pipeline_bridge as pb

        pipeline_path = self._write_fake_pipeline(tmp_path, n_stages=3)
        mock_instantiate = MagicMock(side_effect=[
            (MagicMock(), None, MagicMock()),
            (MagicMock(), None, MagicMock()),
            (MagicMock(), MagicMock(), MagicMock()),
        ])

        with patch.object(pb, "_MORPHOGENESIS_AVAILABLE", True), \
             patch.object(pb, "load_morpho", return_value=MagicMock()), \
             patch.object(pb, "instantiate_morpho", mock_instantiate):
            from core.pipeline_bridge import PipelineBridge
            bridge = PipelineBridge()
            pipeline = bridge.load_pipeline(pipeline_path)

        assert len(pipeline._stages) == 3
        assert mock_instantiate.call_count == 3

    def test_load_pipeline_names_preserved(self, tmp_path):
        import core.pipeline_bridge as pb

        pipeline_path = self._write_fake_pipeline(tmp_path, n_stages=2)
        mock_instantiate = MagicMock(side_effect=[
            (MagicMock(), None, MagicMock()),
            (MagicMock(), None, MagicMock()),
        ])

        with patch.object(pb, "_MORPHOGENESIS_AVAILABLE", True), \
             patch.object(pb, "load_morpho", return_value=MagicMock()), \
             patch.object(pb, "instantiate_morpho", mock_instantiate):
            from core.pipeline_bridge import PipelineBridge
            bridge = PipelineBridge()
            pipeline = bridge.load_pipeline(pipeline_path)

        assert pipeline._names == ["stage_0", "stage_1"]

    def test_load_pipeline_cleans_up_temp_files(self, tmp_path):
        """Temp .morpho files written during load must be deleted after."""
        import core.pipeline_bridge as pb

        pipeline_path = self._write_fake_pipeline(tmp_path, n_stages=1)
        created_temps = []

        original_load_morpho = MagicMock(return_value=MagicMock())

        def tracking_instantiate(boundary):
            # Count temp .morpho files in /tmp that match our pattern
            temps = list(Path(tempfile.gettempdir()).glob("*.morpho"))
            created_temps.extend(temps)
            return (MagicMock(), None, MagicMock())

        with patch.object(pb, "_MORPHOGENESIS_AVAILABLE", True), \
             patch.object(pb, "load_morpho", original_load_morpho), \
             patch.object(pb, "instantiate_morpho", side_effect=tracking_instantiate):
            from core.pipeline_bridge import PipelineBridge
            bridge = PipelineBridge()
            bridge.load_pipeline(pipeline_path)

        # After load, temp files should be gone
        for tmp_file in created_temps:
            assert not tmp_file.exists(), f"Temp file not cleaned up: {tmp_file}"


# ---------------------------------------------------------------------------
# TestPipelineIntegration
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    """End-to-end grow → load → run tests with mocked morphogenesis."""

    def test_grow_then_load_produces_runnable_pipeline(self, tmp_path):
        """Grow a 2-stage pipeline, load it, run data through — all mocked."""
        import core.pipeline_bridge as pb

        # --- Setup fake .morpho files on disk ---
        morpho_a = str(tmp_path / "stage_a_1.morpho")
        morpho_b = str(tmp_path / "stage_b_2.morpho")
        Path(morpho_a).write_bytes(b"MORPHO_A")
        Path(morpho_b).write_bytes(b"MORPHO_B")

        grow_a = MagicMock()
        grow_a.name = "stage_a"
        grow_a.morpho_path = morpho_a
        grow_a.behaviors = ["filter"]
        grow_a.fitness = 0.85
        grow_a.alive = True
        grow_a.fingerprint = "fp1"
        grow_a.zone_graduations = 0.9
        grow_a.calibrated = False

        grow_b = MagicMock()
        grow_b.name = "stage_b"
        grow_b.morpho_path = morpho_b
        grow_b.behaviors = ["classify"]
        grow_b.fitness = 0.92
        grow_b.alive = True
        grow_b.fingerprint = "fp2"
        grow_b.zone_graduations = 0.95
        grow_b.calibrated = True

        mock_creation_bridge = MagicMock()
        mock_creation_bridge.grow.side_effect = [grow_a, grow_b]

        # --- Runtimes for load step ---
        intermediate_out = np.array([0.3, 0.4])
        rt_a = MagicMock()
        rt_a.process.return_value = intermediate_out
        rt_b = MagicMock()
        decoder_b = MagicMock()
        final_output = MagicMock(score=0.95)
        rt_b.predict.return_value = final_output

        mock_instantiate = MagicMock(side_effect=[
            (MagicMock(), None, rt_a),
            (MagicMock(), decoder_b, rt_b),
        ])

        with patch.object(pb, "_MORPHOGENESIS_AVAILABLE", True), \
             patch("core.pipeline_bridge.CreationBridge", return_value=mock_creation_bridge), \
             patch.object(pb, "load_morpho", return_value=MagicMock()), \
             patch.object(pb, "instantiate_morpho", mock_instantiate):
            from core.pipeline_bridge import PipelineBridge

            bridge = PipelineBridge()
            result = bridge.grow_pipeline(
                ["filter noise", "classify signal"],
                seeds=[1, 2],
                output_dir=str(tmp_path),
            )

            pipeline = bridge.load_pipeline(result.pipeline_path)
            pipeline.start()
            output = pipeline.run(np.array([1.0, 2.0]))
            pipeline.stop()

        # Stage A processed the input
        rt_a.process.assert_called_once()
        # Stage B got stage A's output and used the decoder
        rt_b.predict.assert_called_once_with(intermediate_out, decoder_b)
        assert output is final_output


class TestHookGrowPipeline:
    """Tests for PraxisHook.grow_pipeline()."""

    def _make_pipeline_result(self, tmp_path) -> "PipelineResult":
        from core.pipeline_bridge import PipelineResult
        return PipelineResult(
            pipeline_path=str(tmp_path / "test.pipeline"),
            name="filter → classify",
            stage_names=["filter", "classify"],
            stage_morpho_paths=[str(tmp_path / "a.morpho"), str(tmp_path / "b.morpho")],
            stage_behaviors=[["filter"], ["classify"]],
            stage_fitnesses=[0.85, 0.92],
            calibrated=True,
        )

    def test_grow_pipeline_returns_status_dict(self, tmp_path):
        """grow_pipeline() returns a dict with 'status' and pipeline metadata."""
        import core.pipeline_bridge as pb
        from core.praxis_hook import PraxisHook

        pipeline_result = self._make_pipeline_result(tmp_path)
        mock_pipeline_bridge = MagicMock()
        mock_pipeline_bridge.grow_pipeline.return_value = pipeline_result

        hook = MagicMock(spec=PraxisHook)
        hook.grow_pipeline = PraxisHook.grow_pipeline.__get__(hook, PraxisHook)

        with patch.object(pb, "_MORPHOGENESIS_AVAILABLE", True), \
             patch("core.praxis_hook.PipelineBridge", return_value=mock_pipeline_bridge):
            hook.record_conversation = MagicMock(return_value={"status": "captured"})
            hook.record_artifact = MagicMock(return_value={"status": "recorded"})
            hook.record_outcome = MagicMock(return_value={"status": "recorded"})

            result = hook.grow_pipeline(
                descriptions=["filter noise", "classify signal"],
                seeds=[1, 2],
            )

        assert result["status"] == "grown"
        assert result["pipeline_path"] == pipeline_result.pipeline_path
        assert result["name"] == "filter → classify"
        assert result["stage_names"] == ["filter", "classify"]
        assert result["stage_fitnesses"] == [0.85, 0.92]
        assert result["calibrated"] is True

    def test_grow_pipeline_records_artifact_and_outcome(self, tmp_path):
        """grow_pipeline() records the .pipeline as artifact and logs outcome."""
        import core.pipeline_bridge as pb
        from core.praxis_hook import PraxisHook

        pipeline_result = self._make_pipeline_result(tmp_path)
        mock_pipeline_bridge = MagicMock()
        mock_pipeline_bridge.grow_pipeline.return_value = pipeline_result

        hook = MagicMock(spec=PraxisHook)
        hook.grow_pipeline = PraxisHook.grow_pipeline.__get__(hook, PraxisHook)

        with patch.object(pb, "_MORPHOGENESIS_AVAILABLE", True), \
             patch("core.praxis_hook.PipelineBridge", return_value=mock_pipeline_bridge):
            hook.record_conversation = MagicMock(return_value={"status": "captured"})
            hook.record_artifact = MagicMock(return_value={"status": "recorded"})
            hook.record_outcome = MagicMock(return_value={"status": "recorded"})

            hook.grow_pipeline(descriptions=["filter noise", "classify signal"])

        hook.record_artifact.assert_called_once()
        artifact_kwargs = hook.record_artifact.call_args
        assert artifact_kwargs[1]["artifact_type"] == "pipeline"

        hook.record_outcome.assert_called_once()
        outcome_kwargs = hook.record_outcome.call_args
        assert outcome_kwargs[1]["success"] is True

    def test_grow_pipeline_returns_failed_on_exception(self):
        """grow_pipeline() returns {'status': 'failed'} when PipelineBridge raises."""
        import core.pipeline_bridge as pb
        from core.praxis_hook import PraxisHook

        mock_pipeline_bridge = MagicMock()
        mock_pipeline_bridge.grow_pipeline.side_effect = ValueError("organism died")

        hook = MagicMock(spec=PraxisHook)
        hook.grow_pipeline = PraxisHook.grow_pipeline.__get__(hook, PraxisHook)

        with patch.object(pb, "_MORPHOGENESIS_AVAILABLE", True), \
             patch("core.praxis_hook.PipelineBridge", return_value=mock_pipeline_bridge):
            hook.record_conversation = MagicMock(return_value={"status": "captured"})
            hook.record_outcome = MagicMock(return_value={"status": "recorded"})

            result = hook.grow_pipeline(descriptions=["filter noise"])

        assert result["status"] == "failed"
        assert "organism died" in result["error"]
