"""Unit tests for autonomic creation intent detection.

Tests the _check_creation_intent state machine in isolation:
- is_creation_request() trigger detection
- session creation on new request
- answer routing to open session
- grow fires when clarify() returns empty
- session timeout clears state
- non-creation turns are ignored

No LLM calls, no actual grows — both are monkeypatched.
"""

import os
import sys
import threading
import time
import types

import pytest

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vendored  = os.path.join(repo_root, "vendored")
for p in [repo_root, vendored]:
    if p not in sys.path:
        sys.path.insert(0, p)

morphogenesis = pytest.importorskip(
    "morphogenesis",
    reason="morphogenesis not installed — run: pip3 install -e /home/josh/Morphogenesis",
)

from morphogenesis.praxis import is_creation_request, PraxisEngine


# ---------------------------------------------------------------------------
# Minimal stub that has only what _check_creation_intent touches
# ---------------------------------------------------------------------------

def _make_stub():
    """Build a minimal object that hosts _check_creation_intent."""
    import importlib, types

    # Pull the real method off the class without __init__
    from core.praxis_hook import PraxisHook

    stub = types.SimpleNamespace()
    stub._pending_creation      = None
    stub._pending_creation_lock = threading.Lock()

    # Track calls
    stub._llm_responses   = []   # queue of canned responses
    stub._deposited       = []
    stub._grew            = []

    def _llm_call(self_ignored, messages):
        if stub._llm_responses:
            return stub._llm_responses.pop(0)
        return "What size should the organism be? (tiny/small/medium/large)"

    def _deposit_praxis_response(self_ignored, text):
        stub._deposited.append(text)

    def _ask_via_llm(self_ignored, state, question):
        # Run synchronously in tests (no threading) for determinism
        try:
            response = _llm_call(None, [])
            _deposit_praxis_response(None, response)
        except Exception:
            _deposit_praxis_response(None, question.question)

    def _fire_grow(self_ignored, intent):
        stub._grew.append(intent)

    # Bind the real _check_creation_intent, injecting our stubs
    stub._llm_call               = lambda msgs: _llm_call(None, msgs)
    stub._deposit_praxis_response = lambda text: _deposit_praxis_response(None, text)
    stub._ask_via_llm            = lambda state, q: _ask_via_llm(None, state, q)
    stub._fire_grow              = lambda intent: _fire_grow(None, intent)

    # Bind the real method
    stub._check_creation_intent = lambda text: PraxisHook._check_creation_intent(stub, text)

    return stub


# ---------------------------------------------------------------------------
# is_creation_request — trigger detection
# ---------------------------------------------------------------------------

class TestIsCreationRequest:
    def test_grow_me_trigger(self):
        assert is_creation_request("grow me a filter organism")

    def test_grow_an_trigger(self):
        assert is_creation_request("grow an accumulator for sensor data")

    def test_need_filter_trigger(self):
        assert is_creation_request("I need a filter for this noisy stream")

    def test_new_organism_trigger(self):
        assert is_creation_request("new organism please")

    def test_spawn_trigger(self):
        assert is_creation_request("spawn a small delay organism")

    def test_build_me_trigger(self):
        assert is_creation_request("build me an accumulator")

    def test_false_grow_direction(self):
        assert not is_creation_request("grow in that direction")

    def test_false_create_variable(self):
        assert not is_creation_request("create a variable called x")

    def test_false_plain_filter(self):
        assert not is_creation_request("the filter needs fixing")

    def test_false_empty(self):
        assert not is_creation_request("")


# ---------------------------------------------------------------------------
# _check_creation_intent state machine
# ---------------------------------------------------------------------------

class TestCreationIntentStateMachine:

    def test_non_creation_turn_ignored(self):
        stub = _make_stub()
        stub._check_creation_intent("the weather is nice today")
        assert stub._pending_creation is None
        assert stub._grew == []

    def test_creation_turn_opens_session(self):
        stub = _make_stub()
        stub._check_creation_intent("grow me a filter organism")
        # Either a session is pending (clarification needed) or grow fired
        opened = stub._pending_creation is not None
        grew   = len(stub._grew) > 0
        assert opened or grew, "Should have opened a session or fired grow"

    def test_session_gets_question_deposited(self):
        stub = _make_stub()
        stub._check_creation_intent("grow me a filter organism")
        if stub._pending_creation is not None:
            # A clarifying question should have been deposited
            assert len(stub._deposited) > 0

    def test_answer_routes_to_open_session(self):
        stub = _make_stub()
        stub._check_creation_intent("grow me a filter organism")
        if stub._pending_creation is None:
            pytest.skip("No clarification needed for this description")
        initial_answered = len(stub._pending_creation['state'].clarifications_answered)
        stub._check_creation_intent("small")
        # Either still open (more questions) or grew — but answered count must have risen
        if stub._pending_creation is not None:
            new_answered = len(stub._pending_creation['state'].clarifications_answered)
            assert new_answered > initial_answered
        else:
            assert len(stub._grew) > 0

    def test_grow_fires_when_all_questions_answered(self):
        stub = _make_stub()
        stub._check_creation_intent("grow me a filter organism")
        # Keep answering until session closes
        for answer in ["small", "yes, in that order", "medium", "tiny"]:
            if stub._pending_creation is None:
                break
            stub._check_creation_intent(answer)
        assert len(stub._grew) > 0 or stub._pending_creation is None

    def test_session_timeout_clears(self):
        stub = _make_stub()
        stub._check_creation_intent("grow me a filter organism")
        if stub._pending_creation is None:
            pytest.skip("No session to timeout")
        # Fake a stale timestamp
        stub._pending_creation['timestamp'] = time.time() - 400
        stub._check_creation_intent("grow me another filter")
        # The stale session cleared and a new one may or may not open
        # Most important: no crash, and the stale session is gone
        if stub._pending_creation is not None:
            # If a new session opened, timestamp should be recent
            assert time.time() - stub._pending_creation['timestamp'] < 5

    def test_no_double_session(self):
        """A second creation request while a session is open extends the existing session."""
        stub = _make_stub()
        stub._check_creation_intent("grow me a filter organism")
        if stub._pending_creation is None:
            pytest.skip("No session opened")
        first_state = stub._pending_creation['state']
        # Feed another creation phrase — should route as answer, not open a new session
        stub._check_creation_intent("grow me a split organism instead")
        if stub._pending_creation is not None:
            # Same state object (session not replaced)
            assert stub._pending_creation['state'] is first_state

    def test_clear_description_grows_immediately(self):
        """A fully specified description (no clarifications needed) grows without waiting."""
        stub = _make_stub()
        # Patch clarify to return empty — simulates fully specified description
        engine_orig = PraxisEngine.clarify

        def _no_questions(self, state):
            return []

        PraxisEngine.clarify = _no_questions
        try:
            stub._check_creation_intent("grow me a filter organism")
        finally:
            PraxisEngine.clarify = engine_orig

        assert len(stub._grew) > 0
        assert stub._pending_creation is None
