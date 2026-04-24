# ---- Changelog ----
# [2026-04-24] Claude Code (Sonnet 4.6) — Add self-contained intent extraction for autonomic creation
# What: Behavior enum, OrganismIntent, PraxisEngine, CREATION_TRIGGERS, is_creation_request
# Why: _check_creation_intent in praxis_hook.py was importing from the Morphogenesis research
#      repo (from morphogenesis.praxis import ...). Research-stage code has no place coupling
#      into the live ecosystem. This module is a clean, Praxis-local copy that depends on
#      nothing outside the stdlib.
# How: Copied the minimal required definitions from Morphogenesis/morphogenesis/intent.py
#      and Morphogenesis/morphogenesis/praxis.py. praxis_hook.py imports from here instead.
#      When Morphogenesis formally integrates (Praxis→Morphogenesis direction per CLAUDE.md),
#      this module gets replaced by that integration — it is not a permanent fork.
# -------------------

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class Behavior(Enum):
    PASSTHROUGH = "passthrough"
    FILTER      = "filter"
    ACCUMULATE  = "accumulate"
    TRANSFORM   = "transform"
    REDUCE      = "reduce"
    SPLIT       = "split"
    COMBINE     = "combine"
    COMPARE     = "compare"
    DELAY       = "delay"


@dataclass
class OrganismIntent:
    name:        str
    description: str            = ""
    behaviors:   List[Behavior] = field(default_factory=lambda: [Behavior.PASSTHROUGH])
    width:       int            = 15


BEHAVIOR_KEYWORDS = {
    'filter': Behavior.FILTER, 'filters': Behavior.FILTER, 'filtering': Behavior.FILTER,
    'gate': Behavior.FILTER, 'block': Behavior.FILTER, 'attenuate': Behavior.FILTER,
    'screen': Behavior.FILTER, 'reject': Behavior.FILTER,
    'accumulate': Behavior.ACCUMULATE, 'aggregate': Behavior.ACCUMULATE,
    'collect': Behavior.ACCUMULATE, 'sum': Behavior.ACCUMULATE,
    'tally': Behavior.ACCUMULATE, 'count': Behavior.ACCUMULATE, 'total': Behavior.ACCUMULATE,
    'transform': Behavior.TRANSFORM, 'reshape': Behavior.TRANSFORM,
    'convert': Behavior.TRANSFORM, 'change': Behavior.TRANSFORM,
    'reformat': Behavior.TRANSFORM, 'map': Behavior.TRANSFORM,
    'reduce': Behavior.REDUCE, 'summarize': Behavior.REDUCE, 'condense': Behavior.REDUCE,
    'compress': Behavior.REDUCE, 'distill': Behavior.REDUCE, 'average': Behavior.REDUCE,
    'split': Behavior.SPLIT, 'branch': Behavior.SPLIT, 'fork': Behavior.SPLIT,
    'distribute': Behavior.SPLIT, 'divide': Behavior.SPLIT,
    'combine': Behavior.COMBINE, 'merge': Behavior.COMBINE, 'join': Behavior.COMBINE,
    'unify': Behavior.COMBINE, 'mix': Behavior.COMBINE,
    'compare': Behavior.COMPARE, 'rank': Behavior.COMPARE, 'sort': Behavior.COMPARE,
    'order': Behavior.COMPARE, 'prioritize': Behavior.COMPARE,
    'delay': Behavior.DELAY, 'buffer': Behavior.DELAY, 'queue': Behavior.DELAY,
    'hold': Behavior.DELAY, 'wait': Behavior.DELAY,
    'pass': Behavior.PASSTHROUGH, 'forward': Behavior.PASSTHROUGH,
    'relay': Behavior.PASSTHROUGH, 'pipe': Behavior.PASSTHROUGH,
}

SIZE_KEYWORDS = {'tiny': 8, 'small': 12, 'medium': 18, 'large': 25, 'big': 25}

CREATION_TRIGGERS = [
    r'\bgrow (a|an|me|something|one)\b',
    r'\bcreate (an organism|something that|a .{0,30}organism)\b',
    r'\bspawn (a|an)\b',
    r'\bbuild me (a|an)\b',
    r'\bmake me (a|an)\b',
    r'\bi need (a|an) .{0,40}(filter|transform|reduce|split|accumulate|delay|emit|organism)\b',
    r'\bnew organism\b',
]


def is_creation_request(text: str) -> bool:
    lower = text.lower()
    for pattern in CREATION_TRIGGERS:
        if re.search(pattern, lower):
            return True
    return False


@dataclass
class ClarificationQuestion:
    question: str
    field:    str
    options:  List[str]
    required: bool = False


@dataclass
class ConversationState:
    description:              str                       = ""
    extracted_behaviors:      List[Behavior]            = field(default_factory=list)
    width:                    Optional[int]             = None
    name:                     Optional[str]             = None
    clarifications_asked:     List[ClarificationQuestion] = field(default_factory=list)
    clarifications_answered:  List[Tuple[str, str]]     = field(default_factory=list)
    finalized:                bool                      = False


class PraxisEngine:
    """Conversational intent extraction — self-contained Praxis-local copy.

    Source of truth is Morphogenesis/morphogenesis/praxis.py. This copy exists
    only to avoid a live-ecosystem dependency on a research repo. When Morphogenesis
    formally integrates (Praxis→Morphogenesis direction), replace this with that import.
    """

    def hear(self, description: str) -> ConversationState:
        state = ConversationState(description=description)
        lower = description.lower()
        found = []
        for keyword, behavior in BEHAVIOR_KEYWORDS.items():
            if re.search(r'\b' + re.escape(keyword) + r'\b', lower):
                if behavior not in found:
                    found.append(behavior)
        state.extracted_behaviors = found if found else [Behavior.PASSTHROUGH]
        for keyword, size in SIZE_KEYWORDS.items():
            if keyword in lower:
                state.width = size
                break
        if state.width is None:
            state.width = max(12, 6 + len(found) * 4)
        words = re.findall(r'[a-z]+', lower)
        meaningful = [w for w in words if len(w) > 3 and w not in
                      ('that', 'this', 'need', 'want', 'something', 'which',
                       'then', 'from', 'into', 'with')]
        state.name = '_'.join(meaningful[:3]) if meaningful else 'organism'
        return state

    def clarify(self, state: ConversationState) -> List[ClarificationQuestion]:
        questions = []
        if state.extracted_behaviors == [Behavior.PASSTHROUGH] and state.description:
            questions.append(ClarificationQuestion(
                question="I didn't detect specific behaviors. What should this organism do?",
                field="behaviors",
                options=["filter", "transform", "accumulate", "reduce", "combine"],
                required=True,
            ))
        if len(state.extracted_behaviors) > 3:
            questions.append(ClarificationQuestion(
                question=f"I found {len(state.extracted_behaviors)} behaviors. Should they be applied in the order mentioned?",
                field="behavior_order",
                options=["yes, in that order", "no, optimize for best flow"],
            ))
        if state.width is None or state.width == max(12, 6 + len(state.extracted_behaviors) * 4):
            questions.append(ClarificationQuestion(
                question="How complex should this organism be?",
                field="size",
                options=["tiny (simple)", "small", "medium", "large (complex)"],
            ))
        state.clarifications_asked = questions
        return questions

    def answer(self, state: ConversationState, field: str, answer: str) -> None:
        state.clarifications_answered.append((field, answer))
        lower = answer.lower()
        if field == "behaviors":
            for keyword, behavior in BEHAVIOR_KEYWORDS.items():
                if keyword in lower and behavior not in state.extracted_behaviors:
                    state.extracted_behaviors.append(behavior)
            if len(state.extracted_behaviors) > 1 and Behavior.PASSTHROUGH in state.extracted_behaviors:
                state.extracted_behaviors.remove(Behavior.PASSTHROUGH)
        elif field == "size":
            for keyword, size in SIZE_KEYWORDS.items():
                if keyword in lower:
                    state.width = size
                    break

    def finalize(self, state: ConversationState) -> OrganismIntent:
        state.finalized = True
        return OrganismIntent(
            name=state.name or 'organism',
            description=state.description,
            behaviors=state.extracted_behaviors,
            width=state.width or 15,
        )
