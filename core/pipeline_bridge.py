"""PipelineBridge — multi-organism composition for Praxis.

Chain organisms into a self-contained pipeline: output of each stage feeds
the input of the next. Ships as a .pipeline file (gzip JSON with base64-
encoded .morpho per stage) — deploy everywhere, grow once.

# ---- Changelog ----
# [2026-04-20] Claude Code (Sonnet 4.6) — Implement load_pipeline()
#   What: load_pipeline() reads gzip JSON, decodes base64 .morpho per stage,
#         writes to temp file, calls instantiate_morpho(), cleans up temp files
#   Why:  Task 3 — load stored pipeline and return runnable OrganismPipeline
#   How:  gzip.open → json → base64.b64decode → NamedTemporaryFile → instantiate_morpho
#         temp file deleted in finally block regardless of instantiate_morpho outcome
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
"""

from __future__ import annotations

import base64
import gzip
import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import morphogenesis
    from morphogenesis.holographic import load_morpho, instantiate_morpho
    from morphogenesis.runtime import OrganismRuntime
    from morphogenesis.output import OutputDecoder
    from core.creation_bridge import CreationBridge
    _MORPHOGENESIS_AVAILABLE = True
except ImportError:
    CreationBridge = None  # type: ignore[assignment,misc]
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
        if not stages:
            raise ValueError("OrganismPipeline requires at least one stage")
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
