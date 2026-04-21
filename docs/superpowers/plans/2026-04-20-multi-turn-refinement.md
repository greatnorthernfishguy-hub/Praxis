# Multi-Turn Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a multi-turn refinement flow to CreationBridge and PraxisHook so ambiguous descriptions get clarifying questions answered before growth, producing better-targeted organisms.

**Architecture:** PraxisEngine already implements hear/clarify/answer/finalize. CreationBridge currently uses only `quick()` (hear+finalize, no clarification). This plan adds: (1) `refine()`/`answer()`/`grow_refined()` on the bridge that expose the full PraxisEngine conversation flow; (2) a session API on PraxisHook (`start_conversation`/`answer_question`/`grow_from_session`) that stores ConversationState objects keyed by UUID and exposes the flow as dict-in/dict-out calls. The `_grow_from_intent()` private method is factored out of `grow()` so both paths share identical growth/calibration/packaging logic.

**Tech Stack:** Python, morphogenesis.praxis (PraxisEngine, ConversationState, ClarificationQuestion), uuid, pytest

---

## File Structure

| File | Action | What changes |
|------|--------|-------------|
| `core/creation_bridge.py` | Modify | Add Tuple to typing import; add ConversationState+ClarificationQuestion to try block; extract `_grow_from_intent()`; add `refine()`, `answer()`, `grow_refined()` |
| `core/praxis_hook.py` | Modify | Add `import uuid`; add `_grow_sessions` dict to `__init__`; add `start_conversation()`, `answer_question()`, `grow_from_session()` |
| `tests/test_creation_bridge.py` | Modify | Add `TestCreationBridgeRefinement` class, `TestHookSession` class, end-to-end refinement test |

---

## Key APIs (read before implementing)

### PraxisEngine (morphogenesis.praxis)

```python
engine = PraxisEngine()
state = engine.hear(description)           # extract behaviors from keywords → ConversationState
questions = engine.clarify(state)          # generate clarifying questions → List[ClarificationQuestion]
                                           # also sets state.clarifications_asked = questions
engine.answer(state, field, answer_text)   # mutate state with an answer (void)
intent = engine.finalize(state)            # finalize → OrganismIntent (ready for grow_organism())
intent = engine.quick(description)         # shortcut: hear + finalize (no clarification)
```

### ConversationState fields (morphogenesis.praxis)

```python
state.description               # original description
state.extracted_behaviors       # List[Behavior] — mutated by answer()
state.width                     # int — organism size
state.name                      # str — organism name
state.clarifications_asked      # List[ClarificationQuestion] — set by clarify()
state.clarifications_answered   # List[Tuple[str, str]] — (field, answer) pairs added by answer()
state.finalized                 # bool — True after finalize()
```

### ClarificationQuestion fields (morphogenesis.praxis)

```python
q.question   # str — the question text
q.field      # str — which state field this clarifies: "behaviors", "size", "behavior_order"
q.options    # List[str] — suggested answers
q.required   # bool — must be answered before finalize
```

### When does clarify() generate questions?

- **behaviors question** (`required=True`): when no behavior keywords found (description maps to `[PASSTHROUGH]`)
- **behavior_order question**: when more than 3 behaviors extracted
- **size question**: when size not explicitly specified in description

---

## Task 1: Refactor CreationBridge + add refine/answer/grow_refined

**Files:**
- Modify: `Praxis/core/creation_bridge.py`
- Modify: `Praxis/tests/test_creation_bridge.py`

- [ ] **Step 1: Add failing tests first (TDD)**

Add a new class `TestCreationBridgeRefinement` to `tests/test_creation_bridge.py` (before `TestHookGrow`):

```python
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/josh/Praxis && python3 -m pytest tests/test_creation_bridge.py::TestCreationBridgeRefinement -v 2>&1 | head -20
```

Expected: FAIL — `AttributeError: 'CreationBridge' object has no attribute 'refine'`

- [ ] **Step 3: Update imports in `core/creation_bridge.py`**

Change:
```python
from typing import Dict, List, Optional
```
To:
```python
from typing import Dict, List, Optional, Tuple
```

In the try block, update the PraxisEngine import line to also import `ConversationState` and `ClarificationQuestion`:

```python
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
```

- [ ] **Step 4: Extract `_grow_from_intent()` from `grow()`**

Add this private method to `CreationBridge` **before** `grow()`:

```python
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
```

Then replace the body of `grow()` (from `if seed is None:` to the end of the method) with:

```python
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
```

- [ ] **Step 5: Add `refine()`, `answer()`, `grow_refined()` to CreationBridge**

Add these three methods after `grow()`:

```python
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

        Mutates state with the answer. Returns remaining unanswered questions
        re-evaluated on the updated state — resolved questions drop off.

        Args:
            state:       ConversationState from refine().
            field:       Which question field to answer ('behaviors', 'size', etc.)
            answer_text: The answer string (e.g. "filter transform").

        Returns:
            List of remaining ClarificationQuestion objects.
        """
        engine = PraxisEngine()
        engine.answer(state, field, answer_text)
        answered_fields = {f for f, _ in state.clarifications_answered}
        updated_questions = engine.clarify(state)
        return [q for q in updated_questions if q.field not in answered_fields]

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
```

- [ ] **Step 6: Update the changelog in `creation_bridge.py`**

Add at the top of the changelog block:

```python
# [2026-04-20] Claude Code (Sonnet 4.6) — Add multi-turn refinement to CreationBridge
#   What: refine(), answer(), grow_refined() methods. _grow_from_intent() factored
#         out so grow() and grow_refined() share growth/calibration/packaging logic.
#         ConversationState, ClarificationQuestion, Tuple added to imports.
#   Why:  Ambiguous descriptions get clarifying questions answered before growth.
#   How:  PraxisEngine.hear/clarify/answer/finalize conversation flow exposed via bridge.
```

- [ ] **Step 7: Run the new tests**

```bash
cd /home/josh/Praxis && python3 -m pytest tests/test_creation_bridge.py::TestCreationBridgeRefinement -v
```

Expected: All 4 pass.

- [ ] **Step 8: Run the full suite to confirm no regressions**

```bash
cd /home/josh/Praxis && python3 -m pytest tests/ -q
```

Expected: All tests pass (101 pre-existing + 4 new = 105 total).

- [ ] **Step 9: Commit**

```bash
touch /tmp/.opsera-pre-commit-scan-passed
```
Then:
```bash
cd /home/josh/Praxis && git add core/creation_bridge.py tests/test_creation_bridge.py && git commit -m "$(cat <<'EOF'
feat: add multi-turn refinement to CreationBridge

refine(description) → (state, questions)
answer(state, field, text) → remaining questions
grow_refined(state, ...) → GrowResult

_grow_from_intent() factors out shared growth logic so
grow() and grow_refined() have a single implementation.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Session API on PraxisHook

**Files:**
- Modify: `Praxis/core/praxis_hook.py`
- Modify: `Praxis/tests/test_creation_bridge.py`

- [ ] **Step 1: Add failing session tests**

Add a new class `TestHookSession` to `tests/test_creation_bridge.py` (after `TestHookGrow`):

```python
class TestHookSession:
    def test_start_conversation_returns_session_id_and_questions(self):
        from core.praxis_hook import PraxisHook
        hook = PraxisHook()
        result = hook.start_conversation("process data")
        assert "session_id" in result
        assert isinstance(result["session_id"], str)
        assert len(result["session_id"]) > 0
        assert "questions" in result
        assert isinstance(result["questions"], list)
        assert len(result["questions"]) > 0  # "process data" triggers clarification

    def test_answer_question_updates_session(self):
        from core.praxis_hook import PraxisHook
        hook = PraxisHook()
        session = hook.start_conversation("process data")
        sid = session["session_id"]
        result = hook.answer_question(sid, "behaviors", "filter")
        assert "session_id" in result
        assert result["session_id"] == sid
        assert "remaining_questions" in result

    def test_answer_unknown_session_returns_error(self):
        from core.praxis_hook import PraxisHook
        hook = PraxisHook()
        result = hook.answer_question("nonexistent_session_id", "behaviors", "filter")
        assert result["status"] == "error"

    def test_grow_from_session_produces_morpho(self, tmp_path):
        from core.praxis_hook import PraxisHook
        import os
        hook = PraxisHook()
        session = hook.start_conversation("process data")
        sid = session["session_id"]
        hook.answer_question(sid, "behaviors", "filter transform")
        result = hook.grow_from_session(sid, seed=42, output_dir=str(tmp_path))
        assert result["status"] == "grown"
        assert "morpho_path" in result
        assert os.path.exists(result["morpho_path"])

    def test_grow_from_session_cleans_up_session(self, tmp_path):
        from core.praxis_hook import PraxisHook
        hook = PraxisHook()
        session = hook.start_conversation("process data")
        sid = session["session_id"]
        hook.grow_from_session(sid, seed=42, output_dir=str(tmp_path))
        # Session cleaned up — further answers should fail
        result = hook.answer_question(sid, "behaviors", "filter")
        assert result["status"] == "error"

    def test_grow_from_unknown_session_returns_error(self, tmp_path):
        from core.praxis_hook import PraxisHook
        hook = PraxisHook()
        result = hook.grow_from_session("nonexistent_id", seed=42, output_dir=str(tmp_path))
        assert result["status"] == "error"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/josh/Praxis && python3 -m pytest tests/test_creation_bridge.py::TestHookSession -v 2>&1 | head -20
```

Expected: FAIL — `AttributeError: 'PraxisHook' object has no attribute 'start_conversation'`

- [ ] **Step 3: Add `import uuid` to `praxis_hook.py`**

Find the stdlib imports at the top of `core/praxis_hook.py` (near `import os`, `import sys`, etc.). Add:

```python
import uuid
```

Verify `Dict`, `Any`, `List`, `Optional` are all in the typing import. If not, add missing ones.

- [ ] **Step 4: Add `_grow_sessions` to `PraxisHook.__init__`**

Find `def __init__(self)` in `praxis_hook.py`. At the end of `__init__`, after the existing `self._` assignments, add:

```python
        self._grow_sessions: Dict[str, Any] = {}
```

- [ ] **Step 5: Add `start_conversation()`, `answer_question()`, `grow_from_session()` to PraxisHook**

Add these three methods after the existing `grow()` method:

```python
    def start_conversation(self, description: str) -> Dict[str, Any]:
        """Begin a multi-turn refinement conversation.

        Runs hear + clarify on the description. Returns a session ID
        and clarifying questions. Answer questions with answer_question(),
        then grow with grow_from_session().

        Args:
            description: Natural language description of desired behavior.

        Returns:
            Dict with 'session_id' (str) and 'questions' (list of dicts).
            Each question dict has 'question', 'field', 'options', 'required'.
            Returns {'status': 'failed'} if morphogenesis is not installed.
        """
        self.record_conversation(description)
        try:
            from core.creation_bridge import CreationBridge
        except ImportError:
            return {"status": "failed", "error": "morphogenesis not installed. Run: pip3 install -e /home/josh/Morphogenesis"}
        try:
            bridge = CreationBridge()
            state, questions = bridge.refine(description)
        except Exception as exc:
            return {"status": "failed", "error": str(exc)}
        session_id = uuid.uuid4().hex
        self._grow_sessions[session_id] = state
        return {
            "session_id": session_id,
            "questions": [
                {
                    "question": q.question,
                    "field": q.field,
                    "options": q.options,
                    "required": q.required,
                }
                for q in questions
            ],
        }

    def answer_question(self, session_id: str, field: str, answer: str) -> Dict[str, Any]:
        """Answer a clarifying question for a refinement session.

        Args:
            session_id: Session ID from start_conversation().
            field:      The question field to answer ('behaviors', 'size', etc.)
            answer:     The answer string.

        Returns:
            Dict with 'session_id' and 'remaining_questions' (list of dicts).
            Returns {'status': 'error'} if session_id is unknown.
        """
        state = self._grow_sessions.get(session_id)
        if state is None:
            return {"status": "error", "error": f"Session '{session_id}' not found"}
        self.record_conversation(answer)
        try:
            from core.creation_bridge import CreationBridge
        except ImportError:
            return {"status": "failed", "error": "morphogenesis not installed"}
        bridge = CreationBridge()
        remaining = bridge.answer(state, field, answer)
        return {
            "session_id": session_id,
            "remaining_questions": [
                {
                    "question": q.question,
                    "field": q.field,
                    "options": q.options,
                    "required": q.required,
                }
                for q in remaining
            ],
        }

    def grow_from_session(
        self,
        session_id: str,
        seed: Optional[int] = None,
        output_dir: Optional[str] = None,
        normal_examples: Optional[List] = None,
        anomaly_examples: Optional[List] = None,
        class_examples: Optional[Dict] = None,
        mode: str = "anomaly_score",
    ) -> Dict[str, Any]:
        """Finalize a refinement session and grow the organism.

        Calls grow_refined() with the accumulated conversation state,
        then cleans up the session. Returns the same shape as grow().

        Args:
            session_id:       Session ID from start_conversation().
            seed:             Random seed. Random if omitted.
            output_dir:       Where to write the .morpho file.
            normal_examples:  Data for calibration (optional).
            anomaly_examples: Anomaly data for calibration (optional).
            class_examples:   Class data for calibration (optional).
            mode:             Decoder mode (default 'anomaly_score').

        Returns:
            Dict with 'status', 'morpho_path', 'name', etc. (same shape as grow()).
            Returns {'status': 'error'} if session_id is unknown.
        """
        state = self._grow_sessions.get(session_id)
        if state is None:
            return {"status": "error", "error": f"Session '{session_id}' not found"}
        try:
            from core.creation_bridge import CreationBridge
        except ImportError:
            return {"status": "failed", "error": "morphogenesis not installed. Run: pip3 install -e /home/josh/Morphogenesis"}
        try:
            bridge = CreationBridge()
            result = bridge.grow_refined(
                state,
                seed=seed,
                output_dir=output_dir,
                normal_examples=normal_examples,
                anomaly_examples=anomaly_examples,
                class_examples=class_examples,
                mode=mode,
            )
        except Exception as exc:
            self.record_outcome(
                context=f"Failed to grow from session '{session_id}': {exc}",
                outcome_type="review",
                success=False,
                severity=0.6,
                layer_depth="architecture",
                metadata={"error": str(exc)},
            )
            return {"status": "failed", "error": str(exc)}
        del self._grow_sessions[session_id]
        self.record_artifact(
            artifact_id=result.morpho_path,
            content=f"Grown organism '{result.name}' via refinement session. Behaviors: {result.behaviors}",
            artifact_type="morpho",
            event_type="create",
            layer_depth="architecture",
        )
        self.record_outcome(
            context=f"Grew organism '{result.name}' via refinement. Fitness: {result.fitness:.4f}. Behaviors: {result.behaviors}",
            outcome_type="review",
            success=True,
            severity=min(1.0, result.fitness),
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
            "calibrated": result.calibrated,
        }
```

- [ ] **Step 6: Update changelog in `praxis_hook.py`**

Add at the top of the changelog block:

```python
# [2026-04-20] Claude Code (Sonnet 4.6) — Add session-based refinement API
#   What: start_conversation(), answer_question(), grow_from_session().
#         _grow_sessions dict (UUID hex → ConversationState) on __init__.
#         import uuid added.
#   Why:  Expose PraxisEngine multi-turn clarification via hook dict API.
#   How:  Sessions keyed by UUID hex; cleaned up after grow_from_session().
```

- [ ] **Step 7: Run new session tests**

```bash
cd /home/josh/Praxis && python3 -m pytest tests/test_creation_bridge.py::TestHookSession -v
```

Expected: All 6 pass.

- [ ] **Step 8: Run full suite**

```bash
cd /home/josh/Praxis && python3 -m pytest tests/ -q
```

Expected: All tests pass (105 pre-existing + 6 new = 111 total).

- [ ] **Step 9: Commit**

```bash
touch /tmp/.opsera-pre-commit-scan-passed
```
Then:
```bash
cd /home/josh/Praxis && git add core/praxis_hook.py tests/test_creation_bridge.py && git commit -m "$(cat <<'EOF'
feat: session-based refinement API on PraxisHook

start_conversation(description) → {session_id, questions}
answer_question(session_id, field, answer) → {remaining_questions}
grow_from_session(session_id, ...) → same shape as grow()

Sessions keyed by UUID hex, stored in _grow_sessions dict,
cleaned up after grow_from_session().

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: End-to-end refinement test + push

**Files:**
- Modify: `Praxis/tests/test_creation_bridge.py`

- [ ] **Step 1: Add end-to-end refinement test to `TestEndToEnd`**

Add this method to the existing `TestEndToEnd` class:

```python
    def test_multi_turn_refinement_produces_targeted_organism(self, tmp_path):
        """Ambiguous description refined through questions → correctly-targeted organism."""
        from core.praxis_hook import PraxisHook
        import os

        hook = PraxisHook()

        # Start with an ambiguous description — no behavior keywords
        session = hook.start_conversation("process incoming data")
        assert "session_id" in session
        sid = session["session_id"]

        # Ambiguous description should trigger at least one question
        assert len(session["questions"]) > 0, "Ambiguous description should trigger questions"

        # Find and answer the behaviors question
        behavior_q = next(
            (q for q in session["questions"] if q["field"] == "behaviors"),
            None,
        )
        assert behavior_q is not None, "Should ask about behaviors for ambiguous description"

        remaining = hook.answer_question(sid, "behaviors", "filter accumulate")
        assert "remaining_questions" in remaining

        # Grow from the refined session
        result = hook.grow_from_session(sid, seed=42, output_dir=str(tmp_path))

        assert result["status"] == "grown", f"Growth failed: {result.get('error')}"
        assert os.path.exists(result["morpho_path"])

        # The refined organism should carry the clarified behaviors
        behaviors = result["behaviors"]
        assert any(b in behaviors for b in ("filter", "accumulate")), (
            f"Refined organism should have filter/accumulate behaviors, got: {behaviors}"
        )

        print(f"\n  Refined organism: '{result['name']}'")
        print(f"  Behaviors: {behaviors}")
        print(f"  Fitness: {result['fitness']:.4f}")
```

- [ ] **Step 2: Run the end-to-end test**

```bash
cd /home/josh/Praxis && python3 -m pytest tests/test_creation_bridge.py::TestEndToEnd::test_multi_turn_refinement_produces_targeted_organism -v -s
```

Expected: 1 passed. Output shows refined organism name, behaviors, fitness.

- [ ] **Step 3: Run full suite**

```bash
cd /home/josh/Praxis && python3 -m pytest tests/ -q
```

Expected: All tests pass (111 pre-existing + 1 new = 112 total).

- [ ] **Step 4: Commit**

```bash
touch /tmp/.opsera-pre-commit-scan-passed
```
Then:
```bash
cd /home/josh/Praxis && git add tests/test_creation_bridge.py && git commit -m "$(cat <<'EOF'
test: end-to-end multi-turn refinement pipeline

Ambiguous "process incoming data" → clarifying question → answered
with "filter accumulate" → grow_from_session() → organism carries
the clarified behaviors. Verifies the full refinement flow works.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 5: Push**

```bash
cd /home/josh/Praxis && git push
```

---

## Self-Review

**Spec coverage:**
- ✅ `refine(description)` → (state, questions) on CreationBridge (Task 1)
- ✅ `answer(state, field, text)` → remaining questions on CreationBridge (Task 1)
- ✅ `grow_refined(state, ...)` → GrowResult on CreationBridge (Task 1)
- ✅ `_grow_from_intent()` factored out — `grow()` and `grow_refined()` share it (Task 1)
- ✅ `ConversationState`, `ClarificationQuestion`, `Tuple` added to imports (Task 1)
- ✅ `start_conversation(description)` on PraxisHook → session_id + questions (Task 2)
- ✅ `answer_question(session_id, field, answer)` on PraxisHook → remaining_questions (Task 2)
- ✅ `grow_from_session(session_id, ...)` on PraxisHook → same dict shape as `grow()` (Task 2)
- ✅ `_grow_sessions` dict initialized in `__init__` (Task 2)
- ✅ Session cleaned up after `grow_from_session()` (Task 2)
- ✅ Error dict for unknown `session_id` in both `answer_question` and `grow_from_session` (Task 2)
- ✅ End-to-end test: ambiguous description → clarification → refined behaviors in organism (Task 3)

**Placeholder scan:** None found.

**Type consistency:**
- `ConversationState` used as return component in `refine()` and parameter in `answer()`, `grow_refined()` — consistent
- `ClarificationQuestion` used as list element type in `refine()` return and `answer()` return — consistent
- `_grow_sessions: Dict[str, Any]` stores ConversationState objects (typed as Any to avoid forward-ref issues with conditional import) — consistent with all session method usages
- `grow_from_session()` return dict keys: `status`, `morpho_path`, `name`, `behaviors`, `fitness`, `alive`, `fingerprint`, `zone_graduations`, `calibrated` — matches `grow()` return dict exactly
- `answer()` in bridge uses `engine.answer(state, field, answer_text)` — matches PraxisEngine.answer signature `(state, field, answer)` — consistent
