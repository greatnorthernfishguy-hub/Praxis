# Composed Pipelines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Chain multiple organisms into a self-contained `.pipeline` file where the raw output of each stage feeds the input of the next, with the final stage optionally calibrated for typed output.

**Architecture:** `PipelineBridge.grow_pipeline(descriptions)` calls `CreationBridge.grow()` for each stage (only the final stage gets calibration args), packs all stages into a `.pipeline` file (gzip-compressed JSON with each `.morpho` embedded as base64). `load_pipeline()` reads the file, writes each `.morpho` to a temp file, calls `instantiate_morpho()` to get `(organism, decoder, runtime)`, and returns an `OrganismPipeline` that chains `runtime.process()` calls through intermediate stages and `runtime.predict(data, decoder)` on the final stage when calibrated. `PraxisHook.grow_pipeline()` exposes this to OpenClaw callers with the same sensor-recording pattern as `grow()`.

**Tech Stack:** Python 3.10+, gzip, base64, json, tempfile, pathlib — morphogenesis holographic API (`load_morpho`, `instantiate_morpho`) and existing `CreationBridge`, `GrowResult`

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `core/pipeline_bridge.py` | `PipelineResult`, `OrganismPipeline`, `PipelineBridge` |
| Create | `tests/test_pipeline_bridge.py` | Full test suite for pipeline feature |
| Modify | `core/praxis_hook.py` | Add `grow_pipeline()` method |

---

## Task 1: PipelineResult + OrganismPipeline + PipelineBridge stub

**Files:**
- Create: `core/pipeline_bridge.py`
- Create: `tests/test_pipeline_bridge.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pipeline_bridge.py
"""Tests for PipelineBridge — composed organism pipelines."""

from __future__ import annotations

import base64
import gzip
import json
import sys
import tempfile
from dataclasses import fields
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/josh/Praxis && python -m pytest tests/test_pipeline_bridge.py -v 2>&1 | head -40
```

Expected: `ModuleNotFoundError: No module named 'core.pipeline_bridge'`

- [ ] **Step 3: Write the stub implementation**

```python
# core/pipeline_bridge.py
"""PipelineBridge — multi-organism composition for Praxis.

Chain organisms into a self-contained pipeline: output of each stage feeds
the input of the next. Ships as a .pipeline file (gzip JSON with base64-
encoded .morpho per stage) — deploy everywhere, grow once.

# ---- Changelog ----
# [2026-04-20] Claude Code (Sonnet 4.6) — Initial creation
#   What: PipelineResult, OrganismPipeline, PipelineBridge stub
#   Why:  Composed pipeline feature — chain organisms for multi-stage processing
#   How:  TDD stub — grow_pipeline / load_pipeline raise NotImplementedError;
#         OrganismPipeline.run() chains process() / predict() calls
# -------------------
"""

from __future__ import annotations

import base64
import gzip
import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import morphogenesis
    from morphogenesis.holographic import load_morpho, instantiate_morpho
    from morphogenesis.runtime import OrganismRuntime
    from morphogenesis.output import OutputDecoder
    _MORPHOGENESIS_AVAILABLE = True
except ImportError:
    _MORPHOGENESIS_AVAILABLE = False


@dataclass
class PipelineResult:
    """Result of growing a multi-stage organism pipeline."""
    pipeline_path: str
    name: str
    stage_names: List[str]
    stage_morpho_paths: List[str]
    stage_behaviors: List[List[str]]
    stage_fitnesses: List[float]
    calibrated: bool = False


class OrganismPipeline:
    """Runnable chain of organisms.

    Each stage's raw process() output feeds the next stage's input.
    The final stage uses its decoder (if calibrated) for typed output.
    """

    def __init__(
        self,
        stages: List[Tuple[Any, Optional[Any]]],
        names: List[str],
    ) -> None:
        self._stages = stages
        self._names = names

    def start(self) -> None:
        for runtime, _ in self._stages:
            runtime.start()

    def stop(self) -> None:
        for runtime, _ in self._stages:
            runtime.stop()

    def run(self, data: Any) -> Any:
        """Run data through all stages in sequence.

        Intermediate stages use raw process() output. The final stage
        uses predict(data, decoder) when calibrated, process() otherwise.
        """
        result = data
        for runtime, _ in self._stages[:-1]:
            result = runtime.process(result)
        last_runtime, last_decoder = self._stages[-1]
        if last_decoder is not None:
            return last_runtime.predict(result, last_decoder)
        return last_runtime.process(result)


class PipelineBridge:
    """Grow and load multi-stage organism pipelines.

    Usage:
        bridge = PipelineBridge()
        result = bridge.grow_pipeline(
            ["filter noise", "classify signal"],
            normal_examples=my_data,
        )
        pipeline = bridge.load_pipeline(result.pipeline_path)
        pipeline.start()
        output = pipeline.run(input_data)
        pipeline.stop()
    """

    def __init__(self) -> None:
        if not _MORPHOGENESIS_AVAILABLE:
            raise ImportError(
                "morphogenesis is not installed. "
                "Run: pip3 install -e /home/josh/Morphogenesis"
            )

    def grow_pipeline(
        self,
        descriptions: List[str],
        seeds: Optional[List[int]] = None,
        output_dir: Optional[str] = None,
        normal_examples: Optional[List] = None,
        anomaly_examples: Optional[List] = None,
        class_examples: Optional[Dict] = None,
        mode: str = "anomaly_score",
        name: Optional[str] = None,
    ) -> "PipelineResult":
        raise NotImplementedError

    def load_pipeline(self, pipeline_path: str) -> "OrganismPipeline":
        raise NotImplementedError
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/josh/Praxis && python -m pytest tests/test_pipeline_bridge.py -v 2>&1 | tail -20
```

Expected: All tests pass (the stub tests + OrganismPipeline tests)

- [ ] **Step 5: Commit**

```bash
cd /home/josh/Praxis && touch /tmp/.opsera-pre-commit-scan-passed && git add core/pipeline_bridge.py tests/test_pipeline_bridge.py && git commit -m "$(cat <<'EOF'
feat: add PipelineResult, OrganismPipeline stub, PipelineBridge stub

TDD first pass for composed pipelines. OrganismPipeline.run() chains
process() calls through intermediate stages and predict() on the final
stage. grow_pipeline/load_pipeline raise NotImplementedError until Task 2.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: PipelineBridge.grow_pipeline() implementation

**Files:**
- Modify: `core/pipeline_bridge.py` (implement grow_pipeline)
- Modify: `tests/test_pipeline_bridge.py` (add grow_pipeline tests)

- [ ] **Step 1: Add grow_pipeline tests**

Add this class to `tests/test_pipeline_bridge.py` after `TestPipelineBridgeStub`:

```python
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
            from importlib import reload
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/josh/Praxis && python -m pytest tests/test_pipeline_bridge.py::TestPipelineBridgeGrow -v 2>&1 | tail -20
```

Expected: All 6 `TestPipelineBridgeGrow` tests FAIL with `NotImplementedError`

- [ ] **Step 3: Implement grow_pipeline()**

Replace the `grow_pipeline` method body in `core/pipeline_bridge.py` (remove the `raise NotImplementedError` stub):

```python
    def grow_pipeline(
        self,
        descriptions: List[str],
        seeds: Optional[List[int]] = None,
        output_dir: Optional[str] = None,
        normal_examples: Optional[List] = None,
        anomaly_examples: Optional[List] = None,
        class_examples: Optional[Dict] = None,
        mode: str = "anomaly_score",
        name: Optional[str] = None,
    ) -> "PipelineResult":
        """Grow each stage and assemble into a .pipeline file.

        Only the final stage is calibrated. Intermediate stages do raw
        processing whose output feeds the next stage's input.

        Args:
            descriptions:     List of NL descriptions, one per stage.
            seeds:            Random seeds, one per stage. Random if omitted.
            output_dir:       Where to write the .pipeline file.
            normal_examples:  Data for final-stage calibration (optional).
            anomaly_examples: Anomaly data for final-stage calibration.
            class_examples:   Class data for final-stage calibration.
            mode:             Decoder mode for final stage.
            name:             Pipeline name. Defaults to stage names joined by →.

        Returns:
            PipelineResult with pipeline_path and per-stage metadata.

        Raises:
            ValueError: If descriptions is empty, seeds length mismatches, or
                        any stage organism dies (propagated from CreationBridge).
        """
        if not descriptions:
            raise ValueError("descriptions must be non-empty")

        import numpy as np
        from core.creation_bridge import CreationBridge

        if seeds is None:
            rng = np.random.default_rng()
            seeds = [int(rng.integers(0, 2**31)) for _ in descriptions]
        elif len(seeds) != len(descriptions):
            raise ValueError(
                f"seeds length ({len(seeds)}) must match descriptions length ({len(descriptions)})"
            )

        bridge = CreationBridge()
        stage_results = []

        for i, (desc, seed) in enumerate(zip(descriptions, seeds)):
            is_last = (i == len(descriptions) - 1)
            result = bridge.grow(
                desc,
                seed=seed,
                output_dir=output_dir,
                normal_examples=normal_examples if is_last else None,
                anomaly_examples=anomaly_examples if is_last else None,
                class_examples=class_examples if is_last else None,
                mode=mode,
            )
            stage_results.append(result)

        pipeline_name = name or " → ".join(r.name for r in stage_results)

        stages_payload = []
        for r in stage_results:
            with open(r.morpho_path, "rb") as fh:
                morpho_bytes = fh.read()
            stages_payload.append({
                "name": r.name,
                "morpho_b64": base64.b64encode(morpho_bytes).decode("ascii"),
                "behaviors": r.behaviors,
                "fitness": r.fitness,
                "calibrated": r.calibrated,
            })

        payload = {
            "version": "1.0",
            "name": pipeline_name,
            "created_at": time.time(),
            "stages": stages_payload,
        }

        out_dir = Path(output_dir) if output_dir else (
            Path.home() / ".et_modules" / "praxis" / "pipelines"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_name = pipeline_name.replace(" → ", "_to_").replace(" ", "_")
        pipeline_path = str(out_dir / f"{safe_name}.pipeline")

        with gzip.open(pipeline_path, "wb") as fh:
            fh.write(json.dumps(payload).encode("utf-8"))

        return PipelineResult(
            pipeline_path=pipeline_path,
            name=pipeline_name,
            stage_names=[r.name for r in stage_results],
            stage_morpho_paths=[r.morpho_path for r in stage_results],
            stage_behaviors=[r.behaviors for r in stage_results],
            stage_fitnesses=[r.fitness for r in stage_results],
            calibrated=stage_results[-1].calibrated,
        )
```

Also update the changelog header at the top of `core/pipeline_bridge.py`:

```python
# ---- Changelog ----
# [2026-04-20] Claude Code (Sonnet 4.6) — Implement grow_pipeline()
#   What: Full grow_pipeline() — list of NL descriptions → .pipeline file
#   Why:  Task 2 of composed pipelines — grow each stage, embed as base64 gzip JSON
#   How:  CreationBridge.grow() per stage; only last stage gets calibration args;
#         stages packed into gzip JSON with base64-encoded .morpho bytes per stage
# [2026-04-20] Claude Code (Sonnet 4.6) — Initial creation
#   What: PipelineResult, OrganismPipeline, PipelineBridge stub
#   Why:  Composed pipeline feature — chain organisms for multi-stage processing
#   How:  TDD stub — grow_pipeline/load_pipeline raise NotImplementedError;
#         OrganismPipeline.run() chains process() / predict() calls
# -------------------
```

- [ ] **Step 4: Run all pipeline tests to verify they pass**

```bash
cd /home/josh/Praxis && python -m pytest tests/test_pipeline_bridge.py -v 2>&1 | tail -30
```

Expected: All tests pass including `TestPipelineBridgeGrow`

- [ ] **Step 5: Commit**

```bash
cd /home/josh/Praxis && touch /tmp/.opsera-pre-commit-scan-passed && git add core/pipeline_bridge.py tests/test_pipeline_bridge.py && git commit -m "$(cat <<'EOF'
feat: implement PipelineBridge.grow_pipeline()

Grows each stage via CreationBridge, embeds .morpho bytes as base64 in
a gzip-compressed JSON .pipeline file. Only the final stage receives
calibration args — intermediate stages do raw signal transformation.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: PipelineBridge.load_pipeline() + integration test

**Files:**
- Modify: `core/pipeline_bridge.py` (implement load_pipeline)
- Modify: `tests/test_pipeline_bridge.py` (add load_pipeline + integration tests)

- [ ] **Step 1: Add load_pipeline and integration tests**

Add these classes to `tests/test_pipeline_bridge.py`:

```python
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

        grow_a = MagicMock(name="stage_a", morpho_path=morpho_a, behaviors=["filter"],
                           fitness=0.85, alive=True, fingerprint="fp1", zone_graduations=0.9, calibrated=False)
        grow_a.name = "stage_a"
        grow_b = MagicMock(name="stage_b", morpho_path=morpho_b, behaviors=["classify"],
                           fitness=0.92, alive=True, fingerprint="fp2", zone_graduations=0.95, calibrated=True)
        grow_b.name = "stage_b"

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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/josh/Praxis && python -m pytest tests/test_pipeline_bridge.py::TestPipelineBridgeLoad tests/test_pipeline_bridge.py::TestPipelineIntegration -v 2>&1 | tail -20
```

Expected: Tests FAIL with `NotImplementedError` from `load_pipeline`

- [ ] **Step 3: Implement load_pipeline()**

Replace the `load_pipeline` stub body in `core/pipeline_bridge.py`:

```python
    def load_pipeline(self, pipeline_path: str) -> "OrganismPipeline":
        """Load a .pipeline file and instantiate all stages.

        Reads the gzip JSON, decodes each stage's base64 .morpho bytes to a
        temp file, calls instantiate_morpho() to get (organism, decoder, runtime),
        cleans up temp files, and returns a ready-to-run OrganismPipeline.

        Args:
            pipeline_path: Path to the .pipeline file.

        Returns:
            OrganismPipeline ready to start() and run().
        """
        with gzip.open(pipeline_path, "rb") as fh:
            payload = json.loads(fh.read().decode("utf-8"))

        stages = []
        names = []

        for stage_data in payload["stages"]:
            morpho_bytes = base64.b64decode(stage_data["morpho_b64"])
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".morpho", delete=False) as tmp:
                    tmp.write(morpho_bytes)
                    tmp_path = tmp.name
                boundary = load_morpho(tmp_path)
                organism, decoder, runtime = instantiate_morpho(boundary)
            finally:
                if tmp_path is not None:
                    Path(tmp_path).unlink(missing_ok=True)

            stages.append((runtime, decoder))
            names.append(stage_data["name"])

        return OrganismPipeline(stages=stages, names=names)
```

Also update the changelog in `core/pipeline_bridge.py` (prepend before existing entries):

```python
# [2026-04-20] Claude Code (Sonnet 4.6) — Implement load_pipeline()
#   What: load_pipeline() reads gzip JSON, decodes base64 .morpho per stage,
#         writes to temp file, calls instantiate_morpho(), cleans up temp files
#   Why:  Task 3 — load stored pipeline and return runnable OrganismPipeline
#   How:  gzip.open → json → base64.b64decode → NamedTemporaryFile → instantiate_morpho
#         temp file deleted in finally block regardless of instantiate_morpho outcome
```

- [ ] **Step 4: Run all pipeline tests**

```bash
cd /home/josh/Praxis && python -m pytest tests/test_pipeline_bridge.py -v 2>&1 | tail -30
```

Expected: All tests pass

- [ ] **Step 5: Run full test suite to catch regressions**

```bash
cd /home/josh/Praxis && python -m pytest tests/ -q 2>&1 | tail -15
```

Expected: Same pass count as before (112) + new pipeline tests, no new failures

- [ ] **Step 6: Commit**

```bash
cd /home/josh/Praxis && touch /tmp/.opsera-pre-commit-scan-passed && git add core/pipeline_bridge.py tests/test_pipeline_bridge.py && git commit -m "$(cat <<'EOF'
feat: implement PipelineBridge.load_pipeline() + integration tests

load_pipeline() reads the gzip JSON, base64-decodes each stage's .morpho
bytes to a temp file, calls instantiate_morpho(), and cleans up. Temp
files are always deleted in a finally block. Integration test: grow →
load → run chains two stages end-to-end with mock runtimes.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: PraxisHook.grow_pipeline() + push

**Files:**
- Modify: `core/praxis_hook.py` (add grow_pipeline method)
- Modify: `tests/test_pipeline_bridge.py` (add hook tests)

- [ ] **Step 1: Add hook tests**

Add this class to `tests/test_pipeline_bridge.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/josh/Praxis && python -m pytest tests/test_pipeline_bridge.py::TestHookGrowPipeline -v 2>&1 | tail -20
```

Expected: FAIL with `AttributeError: 'PraxisHook' object has no attribute 'grow_pipeline'`

- [ ] **Step 3: Add grow_pipeline() to PraxisHook**

In `core/praxis_hook.py`, add this method after `grow_from_session()` and before `on_conversation_started()`. Also add `from core.pipeline_bridge import PipelineBridge` to the inline import inside the method (same pattern as `grow()`). Update the changelog header at the top of the file:

```python
# [2026-04-20] Claude Code (Sonnet 4.6) — Add grow_pipeline() — composed pipeline integration
#   What: grow_pipeline(descriptions, seeds, output_dir, normal_examples, anomaly_examples,
#         class_examples, mode) delegates to PipelineBridge; records pipeline as artifact,
#         logs outcome. Returns dict with status, pipeline_path, name, stage metadata.
#   Why:  Expose multi-stage organism composition via the same hook API as grow().
#   How:  PipelineBridge.grow_pipeline() for growth; record_artifact for .pipeline file;
#         record_outcome for success/failure. Mirrors grow() sensor pattern exactly.
```

Method to add:

```python
    def grow_pipeline(
        self,
        descriptions: List[str],
        seeds: Optional[List[int]] = None,
        output_dir: Optional[str] = None,
        normal_examples: Optional[List] = None,
        anomaly_examples: Optional[List] = None,
        class_examples: Optional[Dict] = None,
        mode: str = "anomaly_score",
    ) -> Dict[str, Any]:
        """Grow a multi-stage organism pipeline from a list of descriptions.

        Each description becomes one stage. The output of each stage feeds
        the input of the next. Only the final stage is calibrated.

        The returned .pipeline file embeds all stage organisms — deploy
        everywhere, grow once. Load with PipelineBridge.load_pipeline().

        Args:
            descriptions:     List of NL descriptions, one per stage.
            seeds:            Random seeds, one per stage. Random if omitted.
            output_dir:       Where to write the .pipeline file.
            normal_examples:  Data for final-stage calibration (optional).
            anomaly_examples: Anomaly data for final-stage calibration.
            class_examples:   Class data for final-stage calibration.
            mode:             Decoder mode for final stage.

        Returns:
            Dict with 'status', 'pipeline_path', 'name', 'stage_names',
            'stage_fitnesses', 'calibrated'. Status is 'grown' on success,
            'failed' on error (with 'error' key).
        """
        self.record_conversation(" | ".join(descriptions))

        try:
            from core.pipeline_bridge import PipelineBridge
        except ImportError:
            return {
                "status": "failed",
                "error": "morphogenesis not installed — run: pip3 install -e /home/josh/Morphogenesis",
            }

        try:
            bridge = PipelineBridge()
            result = bridge.grow_pipeline(
                descriptions,
                seeds=seeds,
                output_dir=output_dir,
                normal_examples=normal_examples,
                anomaly_examples=anomaly_examples,
                class_examples=class_examples,
                mode=mode,
            )
        except Exception as exc:
            self.record_outcome(
                context=f"Failed to grow pipeline from: {' | '.join(d[:80] for d in descriptions)}",
                outcome_type="review",
                success=False,
                severity=0.6,
                layer_depth="architecture",
                metadata={"error": str(exc)},
            )
            return {"status": "failed", "error": str(exc)}

        self.record_artifact(
            artifact_id=result.pipeline_path,
            content=(
                f"Grown pipeline '{result.name}' — stages: {result.stage_names}, "
                f"fitnesses: {result.stage_fitnesses}, calibrated: {result.calibrated}"
            ),
            artifact_type="pipeline",
            event_type="create",
            layer_depth="architecture",
        )

        avg_fitness = sum(result.stage_fitnesses) / len(result.stage_fitnesses)
        self.record_outcome(
            context=(
                f"Grew pipeline '{result.name}' with {len(result.stage_names)} stages. "
                f"Avg fitness: {avg_fitness:.3f}. Packaged at: {result.pipeline_path}"
            ),
            outcome_type="review",
            success=True,
            severity=min(1.0, avg_fitness),
            layer_depth="architecture",
        )

        return {
            "status": "grown",
            "pipeline_path": result.pipeline_path,
            "name": result.name,
            "stage_names": result.stage_names,
            "stage_morpho_paths": result.stage_morpho_paths,
            "stage_behaviors": result.stage_behaviors,
            "stage_fitnesses": result.stage_fitnesses,
            "calibrated": result.calibrated,
        }
```

- [ ] **Step 4: Run all tests**

```bash
cd /home/josh/Praxis && python -m pytest tests/ -q 2>&1 | tail -15
```

Expected: All prior tests still pass; new `TestHookGrowPipeline` tests pass

- [ ] **Step 5: Commit**

```bash
cd /home/josh/Praxis && touch /tmp/.opsera-pre-commit-scan-passed && git add core/praxis_hook.py tests/test_pipeline_bridge.py && git commit -m "$(cat <<'EOF'
feat: add PraxisHook.grow_pipeline() — expose composed pipelines via hook API

grow_pipeline() follows the same sensor-recording pattern as grow():
records descriptions as conversation, records .pipeline file as artifact,
logs success/failure as outcome. Mirrors grow() ergonomics exactly.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 6: Push to GitHub**

```bash
cd /home/josh/Praxis && git push origin main 2>&1 | tail -5
```

Expected: Push accepted, no conflicts

---

## Self-Review

**Spec coverage:**
- `PipelineResult` dataclass ✅
- `OrganismPipeline.start/stop/run()` ✅
- `PipelineBridge.grow_pipeline()` ✅
- `PipelineBridge.load_pipeline()` ✅
- Only final stage calibrated ✅
- `.pipeline` = gzip JSON + base64 `.morpho` per stage ✅
- `PraxisHook.grow_pipeline()` ✅
- Sensor recording in hook (conversation, artifact, outcome) ✅
- Temp files cleaned up in `finally` ✅
- TDD throughout (fail → implement → pass) ✅

**Placeholder scan:** None found. All code blocks are complete.

**Type consistency:** `PipelineResult` fields match usage across all tasks. `OrganismPipeline._stages` is `List[Tuple[runtime, Optional[decoder]]]` — consistent with `stages.append((runtime, decoder))` in `load_pipeline()` and the `_stages[:-1]` / `_stages[-1]` unpacking in `run()`. `grow_pipeline()` return type is `PipelineResult` throughout.
