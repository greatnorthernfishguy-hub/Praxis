"""PipelineBridge — multi-organism composition for Praxis.

Chain organisms into a self-contained pipeline: output of each stage feeds
the input of the next. Ships as a .pipeline file (gzip JSON with base64-
encoded .morpho per stage) — deploy everywhere, grow once.

# ---- Changelog ----
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
        raise NotImplementedError

    def load_pipeline(self, pipeline_path: str) -> "OrganismPipeline":
        raise NotImplementedError
