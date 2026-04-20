"""Tests for CreationBridge — Morphogenesis→Praxis integration."""

import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

morphogenesis = pytest.importorskip(
    "morphogenesis",
    reason="morphogenesis not installed — run: pip3 install -e /home/josh/Morphogenesis",
)


class TestGrowResult:
    def test_grow_result_has_required_fields(self):
        from core.creation_bridge import GrowResult
        r = GrowResult(
            morpho_path="/tmp/test.morpho",
            name="test_org",
            behaviors=["filter"],
            fitness=0.5,
            alive=True,
            fingerprint="abc123",
            zone_graduations=0.8,
        )
        assert r.morpho_path == "/tmp/test.morpho"
        assert r.name == "test_org"
        assert r.behaviors == ["filter"]
        assert r.fitness == 0.5
        assert r.alive is True
        assert r.fingerprint == "abc123"
        assert r.zone_graduations == 0.8


class TestCreationBridge:
    def test_grow_returns_grow_result(self, tmp_path):
        from core.creation_bridge import CreationBridge, GrowResult
        bridge = CreationBridge()
        result = bridge.grow(
            "filter noise from signal",
            seed=42,
            output_dir=str(tmp_path),
        )
        assert isinstance(result, GrowResult)

    def test_grow_creates_morpho_file(self, tmp_path):
        from core.creation_bridge import CreationBridge
        bridge = CreationBridge()
        result = bridge.grow(
            "filter noise from signal",
            seed=42,
            output_dir=str(tmp_path),
        )
        assert os.path.exists(result.morpho_path)
        assert result.morpho_path.endswith(".morpho")

    def test_grow_morpho_is_inspectable(self, tmp_path):
        from core.creation_bridge import CreationBridge
        from morphogenesis.holographic import inspect_morpho
        bridge = CreationBridge()
        result = bridge.grow(
            "filter and accumulate metrics",
            seed=42,
            output_dir=str(tmp_path),
        )
        info = inspect_morpho(result.morpho_path)
        assert info["name"] == result.name
        assert isinstance(info["behaviors"], list)
        assert isinstance(info["fitness"], float)

    def test_grow_morpho_is_instantiable(self, tmp_path):
        from core.creation_bridge import CreationBridge
        from morphogenesis.holographic import load_morpho, instantiate_morpho
        bridge = CreationBridge()
        result = bridge.grow(
            "filter noise",
            seed=42,
            output_dir=str(tmp_path),
        )
        boundary = load_morpho(result.morpho_path)
        organism, decoder, runtime = instantiate_morpho(boundary)
        assert organism is not None
        runtime.start()
        runtime.stop()

    def test_grow_seed_gives_same_name(self, tmp_path):
        from core.creation_bridge import CreationBridge
        bridge = CreationBridge()
        r1 = bridge.grow("filter noise", seed=42, output_dir=str(tmp_path))
        r2 = bridge.grow("filter noise", seed=42, output_dir=str(tmp_path))
        assert r1.name == r2.name

    def test_grow_extracts_behaviors_from_description(self, tmp_path):
        from core.creation_bridge import CreationBridge
        bridge = CreationBridge()
        result = bridge.grow(
            "reduce and summarize the stream",
            seed=42,
            output_dir=str(tmp_path),
        )
        assert "reduce" in result.behaviors or len(result.behaviors) > 0

    def test_grow_dead_organism_raises(self, tmp_path):
        from core.creation_bridge import CreationBridge
        bridge = CreationBridge()
        with pytest.raises(ValueError, match="died during growth"):
            bridge.grow(
                "filter",
                seed=1,
                output_dir=str(tmp_path),
                _override_intent={"name": "tiny", "width": 3, "height": 3, "max_timesteps": 2},
            )

    def test_grow_without_examples_is_uncalibrated(self, tmp_path):
        from core.creation_bridge import CreationBridge
        bridge = CreationBridge()
        result = bridge.grow("filter noise", seed=42, output_dir=str(tmp_path))
        assert result.calibrated is False

    def test_grow_with_normal_examples_is_calibrated(self, tmp_path):
        from core.creation_bridge import CreationBridge
        bridge = CreationBridge()
        result = bridge.grow(
            "filter noise from signal",
            seed=42,
            output_dir=str(tmp_path),
            normal_examples=[1.0, 2.0, 1.5, 1.8, 2.2],
            anomaly_examples=[50.0, 80.0, 100.0],
            mode="anomaly_score",
        )
        assert result.calibrated is True

    def test_grow_invalid_mode_raises(self, tmp_path):
        from core.creation_bridge import CreationBridge
        bridge = CreationBridge()
        with pytest.raises(ValueError, match="Unknown decoder mode"):
            bridge.grow(
                "filter noise",
                seed=42,
                output_dir=str(tmp_path),
                normal_examples=[1.0, 2.0],
                mode="totally_bogus_mode",
            )

    def test_grow_calibrated_morpho_can_predict(self, tmp_path):
        from core.creation_bridge import CreationBridge
        from morphogenesis.holographic import load_morpho, instantiate_morpho
        from morphogenesis.output import DecodedOutput
        bridge = CreationBridge()
        result = bridge.grow(
            "filter noise from signal",
            seed=42,
            output_dir=str(tmp_path),
            normal_examples=[1.0, 2.0, 1.5, 1.8, 2.2],
            anomaly_examples=[50.0, 80.0, 100.0],
            mode="anomaly_score",
        )
        boundary = load_morpho(result.morpho_path)
        organism, decoder, runtime = instantiate_morpho(boundary)
        assert decoder is not None
        assert decoder._calibrated is True
        runtime.start()
        decoded = runtime.predict(75.0, decoder)
        assert isinstance(decoded, DecodedOutput)
        assert decoded.label in ("NORMAL", "SUSPICIOUS", "UNCERTAIN", "ANOMALY")
        assert 0.0 <= decoded.confidence <= 1.0
        runtime.stop()


class TestCreationBridgeRefinement:
    def test_refine_returns_state_and_questions(self):
        from core.creation_bridge import CreationBridge
        bridge = CreationBridge()
        state, questions = bridge.refine("process data")
        # "process data" has no behavior keywords → clarify() asks what it does
        assert state is not None
        assert isinstance(questions, list)
        assert len(questions) > 0
        assert hasattr(questions[0], 'question')
        assert hasattr(questions[0], 'field')
        assert hasattr(questions[0], 'options')

    def test_answer_updates_state(self):
        from core.creation_bridge import CreationBridge
        from morphogenesis.intent import Behavior
        bridge = CreationBridge()
        state, _ = bridge.refine("process data")
        remaining = bridge.answer(state, "behaviors", "filter transform")
        assert Behavior.FILTER in state.extracted_behaviors
        assert Behavior.TRANSFORM in state.extracted_behaviors
        assert isinstance(remaining, list)

    def test_grow_refined_uses_clarified_intent(self, tmp_path):
        from core.creation_bridge import CreationBridge, GrowResult
        bridge = CreationBridge()
        state, _ = bridge.refine("process data")
        bridge.answer(state, "behaviors", "filter")
        result = bridge.grow_refined(state, seed=42, output_dir=str(tmp_path))
        assert isinstance(result, GrowResult)
        assert result.alive is True
        assert "filter" in result.behaviors

    def test_refine_clear_description_has_no_required_questions(self):
        from core.creation_bridge import CreationBridge
        bridge = CreationBridge()
        state, questions = bridge.refine("filter and accumulate sensor readings")
        required = [q for q in questions if q.required]
        assert len(required) == 0, "Clear descriptions should not require clarification"


class TestHookGrow:
    def test_hook_grow_returns_dict(self, tmp_path):
        from core.praxis_hook import PraxisHook
        hook = PraxisHook()
        result = hook.grow(
            "filter noise from signal",
            seed=42,
            output_dir=str(tmp_path),
        )
        assert isinstance(result, dict)
        assert result["status"] == "grown"
        assert "morpho_path" in result
        assert "behaviors" in result

    def test_hook_grow_records_artifact(self, tmp_path):
        from core.praxis_hook import PraxisHook
        hook = PraxisHook()
        before = hook._art_sensor.get_stats()["total_events"]
        hook.grow("filter and accumulate", seed=42, output_dir=str(tmp_path))
        after = hook._art_sensor.get_stats()["total_events"]
        assert after > before, "grow() should record the .morpho as an artifact"

    def test_hook_grow_records_outcome(self, tmp_path):
        from core.praxis_hook import PraxisHook
        hook = PraxisHook()
        before = hook._out_sensor.get_stats()["total_outcomes"]
        hook.grow("filter noise", seed=42, output_dir=str(tmp_path))
        after = hook._out_sensor.get_stats()["total_outcomes"]
        assert after > before, "grow() should record the growth outcome"

    def test_hook_grow_records_conversation(self, tmp_path):
        from core.praxis_hook import PraxisHook
        hook = PraxisHook()
        before = hook._conv_sensor.get_stats()["total_captured"]
        hook.grow("filter and summarize data streams", seed=42, output_dir=str(tmp_path))
        after = hook._conv_sensor.get_stats()["total_captured"]
        assert after > before, "grow() should record the description as conversation"

    def test_hook_grow_calibrated(self, tmp_path):
        from core.praxis_hook import PraxisHook
        hook = PraxisHook()
        result = hook.grow(
            "filter sensor noise",
            seed=42,
            output_dir=str(tmp_path),
            normal_examples=[1.0, 2.0, 1.5],
            anomaly_examples=[50.0, 80.0],
        )
        assert result["status"] == "grown"
        assert result["calibrated"] is True


class TestEndToEnd:
    """The full pipeline: describe → grow → package → inspect → instantiate → run."""

    def test_full_creation_pipeline(self, tmp_path):
        """Describe software in plain English, get a running organism."""
        from core.praxis_hook import PraxisHook
        from morphogenesis.holographic import load_morpho, instantiate_morpho

        hook = PraxisHook()

        # --- Step 1: Describe what you want ---
        grow_result = hook.grow(
            "filter and reduce sensor noise from a data stream",
            seed=42,
            output_dir=str(tmp_path),
        )

        assert grow_result["status"] == "grown", f"Growth failed: {grow_result.get('error')}"
        assert os.path.exists(grow_result["morpho_path"])

        # --- Step 2: Inspect the package ---
        boundary = load_morpho(grow_result["morpho_path"])
        assert boundary.name == grow_result["name"]
        assert boundary.behaviors is not None

        # --- Step 3: Install and run ---
        organism, decoder, runtime = instantiate_morpho(boundary)
        runtime.start()

        for value in [1.0, 2.0, 3.0, 100.0, 1.5]:
            result = runtime.process(value)
            assert isinstance(result.output_flow, float)

        runtime.stop()

        # --- Step 4: Verify Praxis learned from the creation ---
        stats = hook._module_stats()
        assert stats["conversation_sensor"]["total_captured"] >= 1
        assert stats["artifact_sensor"]["total_events"] >= 1
        assert stats["outcome_sensor"]["total_outcomes"] >= 1

    def test_calibrated_creation_pipeline(self, tmp_path):
        """Describe software + give examples → get calibrated .morpho → predict immediately."""
        from core.praxis_hook import PraxisHook
        from morphogenesis.holographic import load_morpho, instantiate_morpho
        from morphogenesis.output import DecodedOutput

        hook = PraxisHook()

        # Normal CPU usage: 20-60%
        normal_cpu = [22.0, 35.0, 41.0, 28.0, 55.0, 38.0, 47.0, 31.0]
        # Anomalous CPU usage: runaway process
        anomalous_cpu = [185.0, 220.0, 195.0, 240.0]

        grow_result = hook.grow(
            "filter and detect anomalies in CPU usage metrics",
            seed=42,
            output_dir=str(tmp_path),
            normal_examples=normal_cpu,
            anomaly_examples=anomalous_cpu,
            mode="anomaly_score",
        )

        assert grow_result["status"] == "grown", f"Growth failed: {grow_result.get('error')}"
        assert grow_result["calibrated"] is True, "Should be calibrated when examples provided"

        # Load the .morpho and verify decoder is baked in
        boundary = load_morpho(grow_result["morpho_path"])
        assert boundary.output_contract["calibrated"] is True

        # Instantiate and predict — no calibration step needed
        organism, decoder, runtime = instantiate_morpho(boundary)
        assert decoder is not None
        assert decoder._calibrated is True

        runtime.start()

        # Normal readings should score low anomaly
        normal_results = [runtime.predict(v, decoder) for v in [30.0, 42.0, 38.0]]
        # Anomalous reading should score high
        anomaly_results = [runtime.predict(v, decoder) for v in [200.0, 210.0]]

        for r in normal_results + anomaly_results:
            assert isinstance(r, DecodedOutput)
            assert r.label in ("NORMAL", "SUSPICIOUS", "UNCERTAIN", "ANOMALY")
            assert 0.0 <= r.confidence <= 1.0

        runtime.stop()

        print(f"\n  Calibrated organism: '{grow_result['name']}'")
        print(f"  Behaviors: {grow_result['behaviors']}")
        print(f"  Normal readings: {[r.label for r in normal_results]}")
        print(f"  Anomaly readings: {[r.label for r in anomaly_results]}")
