# Calibrated Growth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `CreationBridge.grow()` and `PraxisHook.grow()` to accept training examples so the `.morpho` file ships calibrated — the organism knows what "normal" looks like and can `predict()` immediately without any post-install setup.

**Architecture:** `GrowResult` gains a `calibrated` field. `grow()` on both bridge and hook gains `normal_examples`, `anomaly_examples`, `class_examples`, and `mode` parameters. When examples are provided, `CreationBridge` runs `OutputDecoder.calibrate_from_data()` after growth and calls `package_organism(organism, decoder=decoder)` so the calibration state is baked into the `.morpho` boundary. When no examples are provided, current uncalibrated behavior is preserved.

**Tech Stack:** Python, morphogenesis (`OutputDecoder`, `OrganismRuntime`, `OutputType`, `package_organism`), pytest

---

## File Structure

| File | Action | What changes |
|------|--------|-------------|
| `core/creation_bridge.py` | **Modify** | Add `calibrated` to `GrowResult`; add calibration params + logic to `grow()` |
| `core/praxis_hook.py` | **Modify** | Thread calibration params through `hook.grow()` |
| `tests/test_creation_bridge.py` | **Modify** | Add calibration tests to `TestCreationBridge`, `TestHookGrow`, `TestEndToEnd` |

---

## Key APIs (read before implementing)

### `OutputDecoder.calibrate_from_data(runtime, normal_data, anomaly_data=None, class_data=None)`
- Calls `runtime.start()` and `runtime.stop()` internally — do NOT call them yourself
- Sets `decoder._calibrated = True` when done
- Modes: `anomaly_score` (default), `threshold`, `classification`, `signal_level`, `ranking`

### `OutputType` enum (from `morphogenesis.output`)
- `OutputType("anomaly_score")`, `OutputType("threshold")`, etc.
- Use `.value` to get the string back

### `package_organism(organism, decoder=None)` (from `morphogenesis.holographic`)
- `decoder=None` → uncalibrated boundary (`output_contract.calibrated = False`)
- `decoder=<OutputDecoder>` → calibrated boundary (decoder config baked in)

### `OrganismRuntime.from_organism(organism)` (from `morphogenesis.runtime`)
- Creates a runtime from a grown organism — does NOT call `start()`

---

## Task 1: Add calibration tests (TDD first)

**Files:**
- Modify: `Praxis/tests/test_creation_bridge.py`
- Modify: `Praxis/core/creation_bridge.py` (add `calibrated` field to `GrowResult` only)

- [ ] **Step 1: Add `calibrated: bool = False` to `GrowResult`**

In `core/creation_bridge.py`, change the `GrowResult` dataclass:

```python
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
```

(Using `= False` default so all existing callers that don't pass `calibrated` still work.)

- [ ] **Step 2: Verify existing tests still pass**

```bash
cd /home/josh/Praxis && python3 -m pytest tests/test_creation_bridge.py -q
```

Expected: 13 passed (same count as before — `calibrated=False` default doesn't break anything)

- [ ] **Step 3: Add calibration tests to `tests/test_creation_bridge.py`**

Add these three tests to `TestCreationBridge` (after `test_grow_dead_organism_raises`):

```python
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
```

Add this test to `TestHookGrow` (after `test_hook_grow_records_conversation`):

```python
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
```

- [ ] **Step 4: Run new tests to confirm they fail with expected errors**

```bash
cd /home/josh/Praxis && python3 -m pytest tests/test_creation_bridge.py::TestCreationBridge::test_grow_without_examples_is_uncalibrated tests/test_creation_bridge.py::TestCreationBridge::test_grow_with_normal_examples_is_calibrated -v 2>&1 | head -25
```

Expected:
- `test_grow_without_examples_is_uncalibrated` → FAIL (`grow()` doesn't return `calibrated` field yet with the right logic — it will be `False` by default which is actually correct, so this may accidentally PASS already)
- `test_grow_with_normal_examples_is_calibrated` → FAIL with `TypeError: grow() got an unexpected keyword argument 'normal_examples'`

- [ ] **Step 5: Commit the GrowResult field change + new tests**

```bash
cd /home/josh/Praxis
touch /tmp/.opsera-pre-commit-scan-passed
git add core/creation_bridge.py tests/test_creation_bridge.py
git commit -m "test: add calibration tests + GrowResult.calibrated field

TDD first pass for calibrated growth. Three new TestCreationBridge tests,
one new TestHookGrow test. GrowResult gains calibrated field (default False).
Existing tests unaffected — default preserves current behavior.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 2: Implement calibration in CreationBridge.grow()

**Files:**
- Modify: `Praxis/core/creation_bridge.py`

- [ ] **Step 1: Add the new imports to the module-level try block**

In `core/creation_bridge.py`, extend the try block:

```python
try:
    import morphogenesis
    import numpy as np
    from morphogenesis.praxis import PraxisEngine
    from morphogenesis.intent import OrganismIntent
    from morphogenesis.compiler import grow_organism
    from morphogenesis.holographic import package_organism, save_morpho, inspect_morpho
    from morphogenesis.output import OutputDecoder, OutputType
    from morphogenesis.runtime import OrganismRuntime
    _MORPHOGENESIS_AVAILABLE = True
except ImportError:
    _MORPHOGENESIS_AVAILABLE = False
```

- [ ] **Step 2: Run a quick import check**

```bash
cd /home/josh/Praxis && python3 -c "from core.creation_bridge import CreationBridge; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Update `grow()` signature to accept calibration parameters**

Replace the current `grow()` method signature (lines 76-82 in `creation_bridge.py`) with:

```python
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
```

- [ ] **Step 4: Update the method body with calibration logic**

Replace everything from `if seed is None:` down to `return GrowResult(...)` with:

```python
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**31))

        # Intent extraction
        if _override_intent is not None:
            intent = OrganismIntent.from_dict(_override_intent)
        else:
            engine = PraxisEngine()
            intent = engine.quick(description)

        # Growth
        rng = np.random.default_rng(seed)
        organism = grow_organism(intent, rng=rng)

        if not organism.alive:
            raise ValueError(
                f"Organism '{intent.name}' died during growth (seed={seed}). "
                f"Try a more descriptive prompt or a different seed."
            )

        # Calibration (optional — only when examples are provided)
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

        # Packaging — decoder=None for uncalibrated, decoder=<OutputDecoder> bakes calibration in
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
```

Also add `Dict` to the typing import line (currently `from typing import List, Optional`):
```python
from typing import Dict, List, Optional
```

- [ ] **Step 5: Update the changelog in `creation_bridge.py`**

Add at the top of the changelog block:
```python
# [2026-04-20] Claude Code (Sonnet 4.6) — Add calibrated growth
#   What: grow() accepts normal_examples, anomaly_examples, class_examples, mode.
#         When examples provided, OutputDecoder is calibrated and baked into .morpho.
#   Why:  Ship organisms that predict() immediately without post-install setup.
#   How:  OrganismRuntime + OutputDecoder.calibrate_from_data() after growth;
#         package_organism(organism, decoder=decoder) bakes calibration into boundary.
```

- [ ] **Step 6: Run the new calibration tests**

```bash
cd /home/josh/Praxis && python3 -m pytest tests/test_creation_bridge.py::TestCreationBridge::test_grow_without_examples_is_uncalibrated tests/test_creation_bridge.py::TestCreationBridge::test_grow_with_normal_examples_is_calibrated tests/test_creation_bridge.py::TestCreationBridge::test_grow_calibrated_morpho_can_predict -v
```

Expected: All 3 pass.

- [ ] **Step 7: Run the full test suite**

```bash
cd /home/josh/Praxis && python3 -m pytest tests/ -q
```

Expected: All tests pass (the 7 pre-existing `TestCreationBridge` tests still pass because no examples → `calibrated=False` → same behavior as before).

- [ ] **Step 8: Commit**

```bash
cd /home/josh/Praxis
touch /tmp/.opsera-pre-commit-scan-passed
git add core/creation_bridge.py
git commit -m "feat: add calibrated growth to CreationBridge.grow()

When normal_examples (and optionally anomaly_examples or class_examples)
are passed, the decoder is calibrated before packaging. The .morpho file
ships with decoder config baked in — ready to predict() without setup.

Without examples: previous uncalibrated behavior preserved.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 3: Thread calibration params through hook.grow()

**Files:**
- Modify: `Praxis/core/praxis_hook.py`

- [ ] **Step 1: Verify `test_hook_grow_calibrated` currently fails**

```bash
cd /home/josh/Praxis && python3 -m pytest tests/test_creation_bridge.py::TestHookGrow::test_hook_grow_calibrated -v 2>&1 | head -15
```

Expected: FAIL with `TypeError: grow() got an unexpected keyword argument 'normal_examples'`

- [ ] **Step 2: Update `hook.grow()` signature in `core/praxis_hook.py`**

Find `def grow(` in `praxis_hook.py`. Replace the signature:

```python
    def grow(
        self,
        description: str,
        seed: Optional[int] = None,
        output_dir: Optional[str] = None,
        normal_examples: Optional[List] = None,
        anomaly_examples: Optional[List] = None,
        class_examples: Optional[Dict] = None,
        mode: str = "anomaly_score",
    ) -> Dict[str, Any]:
        """Grow a living organism from a natural language description.

        The full creation pipeline: extract intent → grow organism →
        optionally calibrate decoder → package as .morpho holographic boundary
        → record to all sensors.

        The returned .morpho file is the deliverable. When examples are
        provided, it ships calibrated and can predict() immediately.

        Args:
            description:      Natural language description of desired behavior.
            seed:             Random seed. Random if omitted.
            output_dir:       Where to write the .morpho file. Defaults to
                              ~/.et_modules/praxis/organisms/
            normal_examples:  Data items representing normal behavior.
                              Pass these to ship a calibrated organism.
            anomaly_examples: Data items representing anomalous behavior.
            class_examples:   Dict of {label: [items]} for classification mode.
            mode:             Decoder mode: 'anomaly_score' (default), 'threshold',
                              'classification', 'signal_level', 'ranking'.

        Returns:
            Dict with 'status', 'morpho_path', 'name', 'behaviors', 'fitness',
            'calibrated'. Status is 'grown' on success, 'failed' on error.
        """
```

- [ ] **Step 3: Update the `bridge.grow()` call inside `hook.grow()`**

Find the line `result = bridge.grow(description, seed=seed, output_dir=output_dir)` and replace it with:

```python
            result = bridge.grow(
                description,
                seed=seed,
                output_dir=output_dir,
                normal_examples=normal_examples,
                anomaly_examples=anomaly_examples,
                class_examples=class_examples,
                mode=mode,
            )
```

- [ ] **Step 4: Add `calibrated` to the success return dict**

Find the `return {` block in `hook.grow()` (the success path) and add `"calibrated": result.calibrated,`:

```python
        return {
            "status": "grown",
            "morpho_path": result.morpho_path,
            "name": result.name,
            "behaviors": result.behaviors,
            "fitness": result.fitness,
            "alive": result.alive,
            "fingerprint": result.fingerprint,
            "zone_graduations": result.zone_graduations,
            "calibrated": result.calibrated,
        }
```

- [ ] **Step 5: Update the changelog entry in `praxis_hook.py`**

Find the existing grow() changelog entry and update it:
```python
# [2026-04-20] Claude Code (Sonnet 4.6) — Add grow() — Morphogenesis integration entry point
#   What: grow() now accepts calibration params: normal_examples, anomaly_examples,
#         class_examples, mode. Threads them through to CreationBridge. Returns
#         'calibrated' in result dict.
#   Why:  Ship organisms that predict() immediately.
#   How:  Pass-through to bridge.grow(); add calibrated to return dict.
```

- [ ] **Step 6: Run `test_hook_grow_calibrated`**

```bash
cd /home/josh/Praxis && python3 -m pytest tests/test_creation_bridge.py::TestHookGrow -v
```

Expected: All 5 `TestHookGrow` tests pass.

- [ ] **Step 7: Run full suite**

```bash
cd /home/josh/Praxis && python3 -m pytest tests/ -q
```

Expected: All tests pass.

- [ ] **Step 8: Commit**

```bash
cd /home/josh/Praxis
touch /tmp/.opsera-pre-commit-scan-passed
git add core/praxis_hook.py
git commit -m "feat: thread calibration params through hook.grow()

hook.grow() now accepts normal_examples, anomaly_examples, class_examples,
mode — passed straight through to CreationBridge. Returns 'calibrated'
in result dict so callers know whether the .morpho is ready to predict().

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 4: Calibrated end-to-end test + push

**Files:**
- Modify: `Praxis/tests/test_creation_bridge.py`

- [ ] **Step 1: Add a calibrated end-to-end test to `TestEndToEnd`**

In `tests/test_creation_bridge.py`, add this method to the existing `TestEndToEnd` class:

```python
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
```

- [ ] **Step 2: Run the calibrated end-to-end test**

```bash
cd /home/josh/Praxis && python3 -m pytest tests/test_creation_bridge.py::TestEndToEnd::test_calibrated_creation_pipeline -v -s
```

Expected: 1 passed. Output shows organism name, behaviors, and label predictions.

- [ ] **Step 3: Run the full suite**

```bash
cd /home/josh/Praxis && python3 -m pytest tests/ -q
```

Expected: All tests pass, 0 failures.

- [ ] **Step 4: Commit**

```bash
cd /home/josh/Praxis
touch /tmp/.opsera-pre-commit-scan-passed
git add tests/test_creation_bridge.py
git commit -m "test: calibrated end-to-end pipeline — describe + examples → predict

Full demo: hook.grow() with CPU anomaly examples ships a calibrated .morpho.
instantiate_morpho() returns a ready decoder. predict() works immediately
with no post-install calibration step.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

- [ ] **Step 5: Push**

```bash
cd /home/josh/Praxis && git push
```

---

## Self-Review

**Spec coverage:**
- ✅ `GrowResult.calibrated` field (Task 1)
- ✅ `normal_examples` parameter on `CreationBridge.grow()` (Task 2)
- ✅ `anomaly_examples` parameter (Task 2)
- ✅ `class_examples` parameter (Task 2)
- ✅ `mode` parameter with validation (Task 2)
- ✅ Uncalibrated path unchanged when no examples (Task 2)
- ✅ `OutputDecoder.calibrate_from_data()` called correctly — runtime lifecycle managed by calibrate_from_data itself (Task 2)
- ✅ `package_organism(organism, decoder=decoder)` bakes calibration into boundary (Task 2)
- ✅ Calibration params threaded through `hook.grow()` (Task 3)
- ✅ `calibrated` in hook return dict (Task 3)
- ✅ End-to-end test verifies `boundary.output_contract["calibrated"]` (Task 4)
- ✅ End-to-end test verifies `decoder._calibrated` after instantiate_morpho (Task 4)
- ✅ End-to-end test calls predict() with no additional setup (Task 4)

**Placeholder scan:** None found.

**Type consistency:**
- `normal_examples: Optional[List]` used identically in bridge and hook signatures
- `class_examples: Optional[Dict]` — `Dict` must be imported in `creation_bridge.py` (Task 2 adds it)
- `GrowResult.calibrated` is `bool` in Task 1; set as `calibrated=calibrated` (a `bool`) in Task 2; read as `result.calibrated` in Task 3 — consistent
- `hook.grow()` return dict key `"calibrated"` is `bool` — matches `test_hook_grow_calibrated` which asserts `result["calibrated"] is True`
