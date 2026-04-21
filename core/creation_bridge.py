"""CreationBridge — Morphogenesis→Praxis integration.

Praxis detects creation intent in natural language. CreationBridge calls
Morphogenesis to grow a living organism and package it as a .morpho file.

The .morpho file is a holographic boundary — a self-contained install
package that regrows the organism on any machine. No source code ships.

# ---- Changelog ----
# [2026-04-20] Claude Code (Sonnet 4.6) — Fix: answer() trusts clarify() re-evaluation
#   What: answer() simplified — removed answered-fields filter. Now returns engine.clarify(state) directly.
#         Trust clarify() to re-evaluate from current state rather than filtering by answered fields.
#   Why:  The filter incorrectly hid re-asks when a bad answer didn't update state.
#   How:  engine.clarify(state) re-evaluates from current state naturally — resolved questions
#         drop off because their trigger conditions are gone; unresolved ones reappear correctly.
# [2026-04-20] Claude Code (Sonnet 4.6) — Add multi-turn refinement to CreationBridge
#   What: refine(), answer(), grow_refined() methods. _grow_from_intent() factored out
#         so grow() and grow_refined() share growth logic.
#         ConversationState, ClarificationQuestion, Tuple added to imports.
#   Why:  Ambiguous descriptions get clarifying questions answered before growth.
#   How:  PraxisEngine.hear/clarify/answer/finalize conversation flow exposed via bridge.
# [2026-04-20] Claude Code (Sonnet 4.6) — Fix: don't bake uncalibrated decoder
#   What: If calibrate_from_data() sets _calibrated=False, we now set decoder=None.
#         gate decoder on calibrated — don't bake uncalibrated decoder on calibration failure
#   Why:  Consistency: GrowResult.calibrated=False matches no decoder in .morpho.
#   How:  After reading calibrated flag, check it: if not calibrated, decoder=None.
# [2026-04-20] Claude Code (Sonnet 4.6) — Add calibrated growth
#   What: GrowResult.calibrated field (Task 1). grow() now accepts
#         normal_examples, anomaly_examples, class_examples, mode params.
#         When examples provided, OutputDecoder calibrated and baked into .morpho.
#   Why:  Ship organisms that predict() immediately without post-install setup.
#   How:  OrganismRuntime + OutputDecoder.calibrate_from_data() after growth;
#         package_organism(organism, decoder=decoder) bakes calibration in.
# [2026-04-20] Claude Code (Sonnet 4.6) — Initial creation
#   What: GrowResult dataclass + CreationBridge stub
#   Why:  Task 1 of Morphogenesis→Praxis integration. TDD first pass.
#   How:  GrowResult captures grow() output. CreationBridge.grow() raises
#         NotImplementedError until Task 2 fills it in.
#         also fixed type annotations per quality review
# [2026-04-20] Claude Code (Sonnet 4.6) — Task 2: implement grow()
#   What: Full grow() implementation — NL description → .morpho file
#   Why:  CreationBridge is the Praxis→Morphogenesis integration wire.
#   How:  PraxisEngine.quick() for intent extraction; grow_organism() for
#         growth; package_organism() + save_morpho() for packaging.
# [2026-04-20] Claude Code (Sonnet 4.6) — Task 3: move inline imports to module level
#   What: numpy, PraxisEngine, OrganismIntent, grow_organism, package_organism,
#         save_morpho, inspect_morpho moved from grow() body to module-level try block.
#   Why:  Inline imports in hot paths mask ImportError at call time; module-level
#         placement surfaces missing deps at import time and improves performance.
#   How:  Extend existing try: import morphogenesis block with all 7 imports.
# -------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import morphogenesis
    import numpy as np
    from morphogenesis.praxis import PraxisEngine, ConversationState, ClarificationQuestion
    from morphogenesis.intent import OrganismIntent
    from morphogenesis.compiler import grow_organism
    from morphogenesis.holographic import package_organism, save_morpho, inspect_morpho
    from morphogenesis.output import OutputDecoder, OutputType
    from morphogenesis.runtime import OrganismRuntime
    _MORPHOGENESIS_AVAILABLE = True
except ImportError:
    _MORPHOGENESIS_AVAILABLE = False


@dataclass
class GrowResult:
    """Result of growing an organism from a natural language description."""
    morpho_path: str
    name: str
    behaviors: List[str]
    fitness: float
    alive: bool
    fingerprint: str
    zone_graduations: float
    calibrated: bool = False


class CreationBridge:
    """Bridge between Praxis intent extraction and Morphogenesis organism growth.

    Usage:
        bridge = CreationBridge()
        result = bridge.grow("filter noise from sensor stream", seed=42)
        # result.morpho_path is ready to distribute
    """

    def __init__(self) -> None:
        if not _MORPHOGENESIS_AVAILABLE:
            raise ImportError(
                "morphogenesis is not installed. "
                "Run: pip3 install -e /home/josh/Morphogenesis"
            )

    def _grow_from_intent(
        self,
        intent: "OrganismIntent",
        seed: int,
        output_dir: Optional[str],
        normal_examples: Optional[List],
        anomaly_examples: Optional[List],
        class_examples: Optional[Dict],
        mode: str,
    ) -> "GrowResult":
        """Grow, calibrate, and package an organism from a resolved intent.

        Shared by grow() and grow_refined(). Intent is already extracted
        by the time this is called.
        """
        rng = np.random.default_rng(seed)
        organism = grow_organism(intent, rng=rng)

        if not organism.alive:
            raise ValueError(
                f"Organism '{intent.name}' died during growth (seed={seed}). "
                f"Try a more descriptive prompt or a different seed."
            )

        decoder = None
        calibrated = False
        if normal_examples is not None:
            try:
                output_type = OutputType(mode)
            except ValueError:
                valid = [t.value for t in OutputType]
                raise ValueError(
                    f"Unknown decoder mode '{mode}'. Valid: {', '.join(valid)}"
                )
            runtime = OrganismRuntime.from_organism(organism)
            decoder = OutputDecoder(mode=output_type)
            decoder.calibrate_from_data(
                runtime,
                normal_data=normal_examples,
                anomaly_data=anomaly_examples,
                class_data=class_examples,
            )
            calibrated = decoder._calibrated
            if not calibrated:
                decoder = None

        boundary = package_organism(organism, decoder=decoder)

        out_dir = Path(output_dir) if output_dir else (
            Path.home() / ".et_modules" / "praxis" / "organisms"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        morpho_path = str(out_dir / f"{intent.name}_{seed}.morpho")

        save_morpho(boundary, morpho_path)
        info = inspect_morpho(morpho_path)

        return GrowResult(
            morpho_path=morpho_path,
            name=boundary.name,
            behaviors=info.get("behaviors") or [],
            fitness=info.get("fitness", 0.0),
            alive=True,
            fingerprint=boundary.fingerprint,
            zone_graduations=info.get("zone_graduations", 0.0),
            calibrated=calibrated,
        )

    def grow(
        self,
        description: str,
        seed: Optional[int] = None,
        output_dir: Optional[str] = None,
        normal_examples: Optional[List] = None,
        anomaly_examples: Optional[List] = None,
        class_examples: Optional[Dict] = None,
        mode: str = "anomaly_score",
        _override_intent: Optional[dict] = None,
    ) -> GrowResult:
        """Grow a living organism from a natural language description.

        Extracts intent using PraxisEngine, grows the organism, optionally
        calibrates the output decoder, and packages as a .morpho file.

        When normal_examples are provided, the decoder is calibrated before
        packaging — the .morpho ships ready to predict() without any setup.
        When no examples are provided, the .morpho ships uncalibrated.

        Args:
            description:      Natural language description of desired behavior.
            seed:             Random seed for reproducible growth.
            output_dir:       Directory to write the .morpho file.
                              Defaults to ~/.et_modules/praxis/organisms/
            normal_examples:  Data items representing expected/normal behavior.
                              Required to ship a calibrated organism.
            anomaly_examples: Data items representing anomalous behavior.
                              Used for anomaly_score and threshold modes.
            class_examples:   Dict of {label: [items]} for classification mode.
            mode:             Output decoder mode — 'anomaly_score' (default),
                              'threshold', 'classification', 'signal_level', 'ranking'.
            _override_intent: Dict to use instead of NL extraction (for testing).

        Returns:
            GrowResult with path to the .morpho file and summary metadata.
            result.calibrated is True when examples were provided.

        Raises:
            ValueError: If the organism dies, or mode is unknown.
        """
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**31))

        if _override_intent is not None:
            intent = OrganismIntent.from_dict(_override_intent)
        else:
            engine = PraxisEngine()
            intent = engine.quick(description)

        return self._grow_from_intent(
            intent, seed, output_dir, normal_examples, anomaly_examples, class_examples, mode
        )

    def refine(self, description: str) -> Tuple["ConversationState", List["ClarificationQuestion"]]:
        """Begin a multi-turn refinement conversation.

        Runs hear() + clarify() on the description. Returns the mutable
        ConversationState and any clarifying questions. Pass state to
        answer() for each question, then grow_refined() to finalize.

        Args:
            description: Natural language description of desired behavior.

        Returns:
            (state, questions) — state is mutable; questions may be empty
            if the description is unambiguous.
        """
        engine = PraxisEngine()
        state = engine.hear(description)
        questions = engine.clarify(state)
        return state, questions

    def answer(
        self,
        state: "ConversationState",
        field: str,
        answer_text: str,
    ) -> List["ClarificationQuestion"]:
        """Process an answer to a clarifying question.

        Mutates state with the answer. Returns the current clarifying questions
        re-evaluated on the updated state — correctly-answered questions drop
        off because the state conditions no longer trigger them. Incorrectly-
        answered questions (unrecognized keywords) are re-asked.

        Args:
            state:       ConversationState from refine().
            field:       Which question field to answer ('behaviors', 'size', etc.)
            answer_text: The answer string (e.g. "filter transform").

        Returns:
            List of current ClarificationQuestion objects after the answer.
        """
        engine = PraxisEngine()
        engine.answer(state, field, answer_text)
        return engine.clarify(state)

    def grow_refined(
        self,
        state: "ConversationState",
        seed: Optional[int] = None,
        output_dir: Optional[str] = None,
        normal_examples: Optional[List] = None,
        anomaly_examples: Optional[List] = None,
        class_examples: Optional[Dict] = None,
        mode: str = "anomaly_score",
    ) -> GrowResult:
        """Finalize a refinement conversation and grow the organism.

        Calls finalize() on the state to produce an OrganismIntent, then
        grows. Accepts the same calibration parameters as grow().

        Args:
            state:            ConversationState after answering questions.
            seed:             Random seed. Random if omitted.
            output_dir:       Where to write the .morpho file.
            normal_examples:  Data for calibration (optional).
            anomaly_examples: Anomaly data for calibration (optional).
            class_examples:   Class data for calibration (optional).
            mode:             Decoder mode (default 'anomaly_score').

        Returns:
            GrowResult — same shape as grow().
        """
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**31))
        engine = PraxisEngine()
        intent = engine.finalize(state)
        return self._grow_from_intent(
            intent, seed, output_dir, normal_examples, anomaly_examples, class_examples, mode
        )
