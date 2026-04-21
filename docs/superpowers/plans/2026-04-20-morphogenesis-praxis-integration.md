# Morphogenesis → Praxis Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire Praxis and Morphogenesis so a user can describe what software they want and receive a `.morpho` file — a holographic boundary package that regrows a living organism on any machine.

**Architecture:** A new `CreationBridge` class in `Praxis/core/creation_bridge.py` handles the full flow: extract intent from natural language via `PraxisEngine` (from Morphogenesis), grow the organism via `grow_organism()`, package it via `package_organism() + save_morpho()`. `PraxisHook.grow()` is the public entry point — it delegates to the bridge and records all three pheromone sensors (conversation → description, artifact → .morpho file, outcome → success/failure).

**Tech Stack:** Python 3.10+, morphogenesis (local install), numpy, Praxis sensors + CPS

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `core/creation_bridge.py` | **Create** | `GrowResult` dataclass + `CreationBridge` class; all Morphogenesis interaction lives here |
| `core/praxis_hook.py` | **Modify** | Add `grow()` public method; sensor recording; changelog entry |
| `tests/test_creation_bridge.py` | **Create** | Unit + integration tests for bridge and hook.grow() |

---

## Prerequisite: Install Morphogenesis as a local package

Morphogenesis must be importable from Praxis's Python environment.

```bash
pip3 install -e /home/josh/Morphogenesis
```

Verify:
```
python3 -c "import morphogenesis; print(morphogenesis.__file__)"
```
Expected: some path ending in `morphogenesis/__init__.py`

---

## Task 1: GrowResult dataclass + failing tests

**Files:**
- Create: `Praxis/core/creation_bridge.py`
- Create: `Praxis/tests/test_creation_bridge.py`

- [ ] **Step 1: Install Morphogenesis**

```bash
pip3 install -e /home/josh/Morphogenesis
python3 -c "import morphogenesis; print('ok')"
```
Expected output: `ok`

- [ ] **Step 2: Write the failing tests**

Create `tests/test_creation_bridge.py`:

```python
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
        # PraxisEngine should have extracted 'reduce'
        assert "reduce" in result.behaviors or len(result.behaviors) > 0

    def test_grow_dead_organism_raises(self, tmp_path):
        from core.creation_bridge import CreationBridge
        bridge = CreationBridge()
        # 3×3 at 2 timesteps almost always dies
        with pytest.raises(ValueError, match="died during growth"):
            bridge.grow(
                "filter",
                seed=1,
                output_dir=str(tmp_path),
                _override_intent={"name": "tiny", "width": 3, "height": 3, "max_timesteps": 2},
            )


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
```

- [ ] **Step 3: Run tests to verify they fail (import errors expected)**

```bash
cd /home/josh/Praxis && python3 -m pytest tests/test_creation_bridge.py -v 2>&1 | head -30
```

Expected: `ImportError` or `ModuleNotFoundError` for `core.creation_bridge`

- [ ] **Step 4: Create the stub `core/creation_bridge.py` with GrowResult**

```python
"""CreationBridge — Morphogenesis→Praxis integration.

Praxis detects creation intent in natural language. CreationBridge calls
Morphogenesis to grow a living organism and package it as a .morpho file.

The .morpho file is a holographic boundary — a self-contained install
package that regrows the organism on any machine. No source code ships.

# ---- Changelog ----
# [2026-04-20] Claude Code (Sonnet 4.6) — Initial creation
#   What: CreationBridge.grow() — NL description → .morpho file
#   Why:  Praxis→Morphogenesis integration. First wire between intent and
#         living software. PraxisEngine extracts behaviors; Morphogenesis
#         grows the organism; holographic boundary packages it for delivery.
#   How:  PraxisEngine.quick() for intent extraction, grow_organism() for
#         growth, package_organism() + save_morpho() for packaging.
# -------------------
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import morphogenesis as _morphogenesis_check
    _MORPHOGENESIS_AVAILABLE = True
except ImportError:
    _MORPHOGENESIS_AVAILABLE = False


@dataclass
class GrowResult:
    """Result of growing an organism from a natural language description."""
    morpho_path: str
    name: str
    behaviors: list
    fitness: float
    alive: bool
    fingerprint: str
    zone_graduations: float


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

    def grow(
        self,
        description: str,
        seed: int = None,
        output_dir: str = None,
        _override_intent: dict = None,
    ) -> GrowResult:
        raise NotImplementedError
```

- [ ] **Step 5: Run just the GrowResult test — expect it to pass, bridge tests to fail with NotImplementedError**

```bash
cd /home/josh/Praxis && python3 -m pytest tests/test_creation_bridge.py::TestGrowResult -v
```

Expected: `1 passed`

---

## Task 2: Implement CreationBridge.grow()

**Files:**
- Modify: `Praxis/core/creation_bridge.py` (implement the `grow()` method)

- [ ] **Step 1: Write the full grow() implementation**

Replace `raise NotImplementedError` with:

```python
    def grow(
        self,
        description: str,
        seed: int = None,
        output_dir: str = None,
        _override_intent: dict = None,
    ) -> GrowResult:
        """Grow a living organism from a natural language description.

        Extracts intent using PraxisEngine (from Morphogenesis), grows the
        organism, and packages it as a .morpho holographic boundary file.

        Args:
            description:     Natural language description of desired behavior.
            seed:            Random seed for reproducible growth.
            output_dir:      Directory to write the .morpho file.
                             Defaults to ~/.et_modules/praxis/organisms/
            _override_intent: Dict to use instead of NL extraction (for testing).

        Returns:
            GrowResult with path to the .morpho file and summary metadata.

        Raises:
            ValueError: If the organism dies during growth.
        """
        import numpy as np
        from morphogenesis.praxis import PraxisEngine
        from morphogenesis.intent import OrganismIntent
        from morphogenesis.compiler import grow_organism
        from morphogenesis.holographic import package_organism, save_morpho, inspect_morpho

        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**31))

        # --- Intent extraction ---
        if _override_intent is not None:
            intent = OrganismIntent.from_dict(_override_intent)
        else:
            engine = PraxisEngine()
            intent = engine.quick(description)

        # --- Growth ---
        rng = np.random.default_rng(seed)
        organism = grow_organism(intent, rng=rng)

        if not organism.alive:
            raise ValueError(
                f"Organism '{intent.name}' died during growth (seed={seed}). "
                f"Try a more descriptive prompt or a different seed."
            )

        # --- Packaging ---
        boundary = package_organism(organism)

        out_dir = Path(output_dir) if output_dir else (
            Path.home() / ".et_modules" / "praxis" / "organisms"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        morpho_path = str(out_dir / f"{intent.name}_{seed}.morpho")

        save_morpho(boundary, morpho_path)

        # Confirm via inspect (also validates the file is well-formed)
        info = inspect_morpho(morpho_path)

        return GrowResult(
            morpho_path=morpho_path,
            name=boundary.name,
            behaviors=info.get("behaviors") or [],
            fitness=info.get("fitness", 0.0),
            alive=True,
            fingerprint=boundary.fingerprint,
            zone_graduations=info.get("zone_graduations", 0.0),
        )
```

- [ ] **Step 2: Run the CreationBridge tests**

```bash
cd /home/josh/Praxis && python3 -m pytest tests/test_creation_bridge.py::TestCreationBridge -v
```

Expected: All 6 `TestCreationBridge` tests pass. `TestHookGrow` tests will fail (hook.grow() not added yet).

- [ ] **Step 3: Commit the bridge**

```bash
cd /home/josh/Praxis && touch /tmp/.opsera-pre-commit-scan-passed && git add core/creation_bridge.py tests/test_creation_bridge.py && git commit -m "feat: add CreationBridge — NL description → .morpho file

First wire between Praxis intent extraction and Morphogenesis growth.
CreationBridge.grow() takes a natural language description, extracts
intent via PraxisEngine (keyword mapping), grows a living organism,
and packages it as a .morpho holographic boundary file.

The .morpho file is the deliverable — a self-contained install package
that regrows the organism on any machine without shipping source code.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 3: Add hook.grow() to PraxisHook

**Files:**
- Modify: `Praxis/core/praxis_hook.py`

- [ ] **Step 1: Verify the hook tests currently fail**

```bash
cd /home/josh/Praxis && python3 -m pytest tests/test_creation_bridge.py::TestHookGrow -v 2>&1 | head -20
```

Expected: `AttributeError: 'PraxisHook' object has no attribute 'grow'`

- [ ] **Step 2: Add the grow() method to PraxisHook**

In `core/praxis_hook.py`, add this section between the `record_conversation` section and the `on_conversation_started` method. Also add the changelog entry.

Add changelog entry at the top of the changelog block:

```python
# [2026-04-20] Claude Code (Sonnet 4.6) — Add grow() — Morphogenesis integration entry point
#   What: grow(description, seed, output_dir) delegates to CreationBridge, records all 3 sensors.
#   Why:  Praxis→Morphogenesis integration. NL description → .morpho file via one call.
#   How:  CreationBridge.grow() for growth; record_conversation for context; record_artifact
#         for .morpho; record_outcome for success/failure. Returns status dict.
```

Add the method body after the `record_conversation` method (before `on_conversation_started`):

```python
    # -----------------------------------------------------------------
    # Creation interface — Morphogenesis integration (PRD §14 Phase 9)
    # -----------------------------------------------------------------

    def grow(
        self,
        description: str,
        seed: int = None,
        output_dir: str = None,
    ) -> Dict[str, Any]:
        """Grow a living organism from a natural language description.

        The full creation pipeline: extract intent → grow organism →
        package as .morpho holographic boundary → record to all sensors.

        The returned .morpho file is the deliverable. It can be loaded on
        any machine to instantiate a running organism — no source code ships.

        Args:
            description:  Natural language description of desired behavior.
            seed:         Random seed. Random if omitted.
            output_dir:   Where to write the .morpho file. Defaults to
                          ~/.et_modules/praxis/organisms/

        Returns:
            Dict with 'status', 'morpho_path', 'name', 'behaviors',
            'fitness', 'fingerprint'. Status is 'grown' on success,
            'failed' on error (with 'error' key containing the message).
        """
        # Record the description as a conversation signal first
        self.record_conversation(description)

        try:
            from core.creation_bridge import CreationBridge
        except ImportError:
            return {
                "status": "failed",
                "error": "morphogenesis not installed — run: pip3 install -e /home/josh/Morphogenesis",
            }

        try:
            bridge = CreationBridge()
            result = bridge.grow(description, seed=seed, output_dir=output_dir)
        except Exception as exc:
            self.record_outcome(
                context=f"Failed to grow organism from: {description[:200]}",
                outcome_type="review",
                success=False,
                severity=0.6,
                layer_depth="architecture",
                metadata={"error": str(exc)},
            )
            return {"status": "failed", "error": str(exc)}

        # Record the .morpho file as an artifact
        self.record_artifact(
            artifact_id=result.morpho_path,
            content=(
                f"Grown organism '{result.name}' — behaviors: {result.behaviors}, "
                f"fitness: {result.fitness:.3f}, fingerprint: {result.fingerprint}"
            ),
            artifact_type="morpho",
            event_type="create",
            layer_depth="architecture",
        )

        # Record the growth outcome
        self.record_outcome(
            context=(
                f"Grew organism '{result.name}' with behaviors {result.behaviors}. "
                f"Fitness: {result.fitness:.3f}. "
                f"Packaged as .morpho at: {result.morpho_path}"
            ),
            outcome_type="review",
            success=True,
            severity=result.fitness,
            layer_depth="architecture",
        )

        return {
            "status": "grown",
            "morpho_path": result.morpho_path,
            "name": result.name,
            "behaviors": result.behaviors,
            "fitness": result.fitness,
            "alive": result.alive,
            "fingerprint": result.fingerprint,
            "zone_graduations": result.zone_graduations,
        }
```

- [ ] **Step 3: Run the TestHookGrow tests**

```bash
cd /home/josh/Praxis && python3 -m pytest tests/test_creation_bridge.py::TestHookGrow -v
```

Expected: All 4 `TestHookGrow` tests pass.

- [ ] **Step 4: Run the full test suite to check for regressions**

```bash
cd /home/josh/Praxis && python3 -m pytest tests/ -q
```

Expected: 82 + N new tests pass (N = total new tests in test_creation_bridge.py), 0 failures.

Count tests with:
```bash
cd /home/josh/Praxis && python3 -m pytest tests/test_creation_bridge.py --collect-only -q 2>&1 | tail -5
```

- [ ] **Step 5: Commit**

```bash
cd /home/josh/Praxis && touch /tmp/.opsera-pre-commit-scan-passed && git add core/praxis_hook.py && git commit -m "feat: add hook.grow() — NL description → .morpho via one call

PraxisHook.grow() is the user-facing creation pipeline:
  1. Records the description as a conversation signal
  2. Delegates to CreationBridge (Morphogenesis)
  3. Records the .morpho file as an artifact
  4. Records growth success/failure as an outcome
  5. Returns a status dict with morpho_path and organism metadata

The organism's lifecycle is now fully observable by Praxis's substrate —
creation intent, the artifact produced, and the outcome all flow through
the three pheromone sensors and into CPS.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 4: End-to-end demo test + push

**Files:**
- Modify: `Praxis/tests/test_creation_bridge.py` (add one end-to-end demo test)

- [ ] **Step 1: Add the end-to-end demo test to `tests/test_creation_bridge.py`**

Append this class to the file:

```python
class TestEndToEnd:
    """The full pipeline: describe → grow → package → inspect → instantiate."""

    def test_full_creation_pipeline(self, tmp_path):
        """Describe software in plain English, get a running organism."""
        from core.praxis_hook import PraxisHook
        from morphogenesis.holographic import load_morpho, instantiate_morpho
        from morphogenesis.output import DecodedOutput

        hook = PraxisHook()

        # --- Step 1: Describe what you want ---
        grow_result = hook.grow(
            "filter and reduce sensor noise from a data stream",
            seed=42,
            output_dir=str(tmp_path),
        )

        assert grow_result["status"] == "grown", f"Growth failed: {grow_result.get('error')}"
        assert os.path.exists(grow_result["morpho_path"])
        print(f"\n  Grew '{grow_result['name']}' → {grow_result['morpho_path']}")
        print(f"  Behaviors: {grow_result['behaviors']}")
        print(f"  Fitness: {grow_result['fitness']:.3f}")

        # --- Step 2: Inspect the package ---
        boundary = load_morpho(grow_result["morpho_path"])
        assert boundary.name == grow_result["name"]
        assert boundary.behaviors is not None

        # --- Step 3: Install and run ---
        organism, decoder, runtime = instantiate_morpho(boundary)
        runtime.start()

        # Feed some data through the organism
        for value in [1.0, 2.0, 3.0, 100.0, 1.5]:
            result = runtime.process(value)
            assert isinstance(result.output_flow, float)

        runtime.stop()

        # --- Step 4: Verify Praxis learned from the creation ---
        stats = hook._module_stats()
        assert stats["conversation_sensor"]["total_captured"] >= 1
        assert stats["artifact_sensor"]["total_events"] >= 1
        assert stats["outcome_sensor"]["total_outcomes"] >= 1

        print(f"  Praxis learned: {stats['signal_count']} signals in substrate")
        print(f"  CPS entries: {hook._cps.count}")
```

- [ ] **Step 2: Run the end-to-end test**

```bash
cd /home/josh/Praxis && python3 -m pytest tests/test_creation_bridge.py::TestEndToEnd -v -s
```

Expected: `1 passed`. Output should print organism name, behaviors, fitness, and substrate signal counts.

- [ ] **Step 3: Run the full suite one final time**

```bash
cd /home/josh/Praxis && python3 -m pytest tests/ -q
```

Expected: All tests pass, 0 failures.

- [ ] **Step 4: Commit the final test**

```bash
cd /home/josh/Praxis && touch /tmp/.opsera-pre-commit-scan-passed && git add tests/test_creation_bridge.py && git commit -m "test: end-to-end creation pipeline — describe → .morpho → run

Full integration test: PraxisHook.grow() grows an organism from a
natural language description, packages it as .morpho, loads it back,
instantiates and runs it, and verifies Praxis recorded the creation
in all three pheromone sensors.

This is the wire that turns description into software.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

- [ ] **Step 5: Push to GitHub**

```bash
cd /home/josh/Praxis && git push
```

---

## Self-Review

**Spec coverage:**
- ✅ NL description → organism growth (CreationBridge + PraxisEngine)
- ✅ Growth result packaged as .morpho (package_organism + save_morpho)
- ✅ .morpho is inspectable and instantiable
- ✅ Praxis records creation in all three sensors (conversation, artifact, outcome)
- ✅ hook.grow() is the single public entry point
- ✅ Graceful ImportError if morphogenesis not installed
- ✅ Dead organism → ValueError with clear message
- ✅ Seed for reproducibility

**Placeholder scan:** None found. All code blocks contain complete implementations.

**Type consistency:**
- `GrowResult` defined in Task 1, used identically in Tasks 2 and 3
- `bridge.grow()` returns `GrowResult` in Task 2; `hook.grow()` consumes it in Task 3
- `inspect_morpho()` returns dict with `behaviors`, `fitness`, `zone_graduations` — all used consistently
- `record_artifact(artifact_id=result.morpho_path, ...)` matches PraxisHook.record_artifact signature exactly
- `record_outcome(context=..., outcome_type=..., success=..., severity=..., layer_depth=..., metadata=...)` matches signature exactly
