# Praxis Repository
## Claude Code Onboarding — Repo-Specific

**You have already read the global `CLAUDE.md` and `ARCHITECTURE.md`.**
**If you have not, stop. Go read them. The Laws defined there govern this repo.**
**This document adds Praxis-specific rules on top of those Laws.**

---

## What This Repo Is

Praxis (formerly The Creation Engine / TCE) is the sensory integration cortex of the E-T Systems digital organism. It captures three continuous signal streams — conversation, artifacts, and outcomes — and feeds them into an NG-Lite learning substrate. The substrate's learned topology handles classification, connection, surfacing, and learning automatically.

**Praxis does not generate code.** It provides context intelligence that makes vibe coding reliable, repeatable, and self-improving. It bridges knowing and doing — turns intent into practice. It provides session continuity.

**Status: v0.8.0 Alpha — all 8 phases complete.** Vendored files synced from NeuroGraph canonical (2026-03-18). All three pheromone sensors implemented. CPS with substrate-routed retrieval. Session Bridge solving context death. 79 tests passing.

---

## 1. Repository Structure

```
~/Praxis/
├── main.py                        # Module entry point
├── et_module.json                 # Module manifest (v2 schema)
├── config.yaml                    # All configuration (PRD §13)
├── SKILL.md                       # OpenClaw skill discovery
├── requirements.txt               # Python dependencies
├── LICENSE                        # AGPL-3.0
├── core/                          # Core domain logic
│   ├── config.py                  # PraxisConfig — all settings from config.yaml
│   ├── signals.py                 # WorkflowSignal, ConversationMeta, ArtifactMeta, OutcomeMeta
│   ├── praxis_hook.py             # OpenClaw skill entry point (PraxisHook singleton)
│   └── session_bridge.py          # Session continuity (stub — Phase 4)
├── sensors/                       # The three pheromones (PRD §4)
│   ├── base.py                    # SensorBase ABC
│   ├── conversation.py            # Pheromone 1: Conversation Stream
│   ├── artifact.py                # Pheromone 2: Artifact Stream (stub — Phase 5)
│   └── outcome.py                 # Pheromone 3: Outcome Stream (stub — Phase 6)
├── store/                         # Context Persistence Store (PRD §8)
│   └── cps.py                     # CPS + CPSEntry (stub — Phase 3)
├── vendored/                      # NEVER modify (Law 2)
│   ├── ng_lite.py
│   ├── ng_peer_bridge.py
│   ├── ng_ecosystem.py
│   ├── openclaw_adapter.py
│   └── ng_autonomic.py
├── tests/                         # Test suite (79 tests)
│   ├── test_config.py
│   ├── test_signals.py
│   ├── test_hook.py
│   ├── test_conversation.py
│   ├── test_cps.py
│   ├── test_session_bridge.py
│   ├── test_artifact.py
│   ├── test_outcome.py
│   └── test_integration.py
└── docs/
    └── ETHICS.md                  # Ethics doc (planned)
```

---

## 2. Key Architectural Constraints

### Praxis Does NOT Write Autonomic State

Praxis reads `ng_autonomic.py` to adjust capture granularity. It does NOT write to it. Only security modules (Immunis, TrollGuard, Cricket) have write permission. This is a hard boundary — PRD §9.

### Praxis Does NOT Generate Code

Praxis is a context substrate, not a code generator. Code generation is the external LLM's job, selected and routed by TID. If you find yourself writing code that generates code, you are out of domain.

### Raw Experience Enters the Substrate (Law 7)

Signals enter the substrate as raw embeddings with fair starting values. No classification before the substrate sees the data. Classification emerges from learned topology.

### The Substrate Is the Pipeline (PRD §2.3)

There is no explicit routing, classification, or pipeline staged processing. The substrate handles classification through learned associations, connection through synaptic propagation, surfacing through voltage thresholds, and learning through STDP.

---

## 3. The Three Pheromones

| Sensor | File | Status |
|--------|------|--------|
| Conversation Stream | `sensors/conversation.py` | Full — temporal binding, both directions, multi-speaker |
| Artifact Stream | `sensors/artifact.py` | Full — lifecycle tracking, reference counting, stale detection |
| Outcome Stream | `sensors/outcome.py` | Full — reward signals, severity scaling, intent tracing |

All sensors extend `sensors/base.py` (SensorBase ABC).

---

## 4. Vendored Files

All five vendored files in `vendored/` directory, synced from NeuroGraph canonical on 2026-03-18:

| File | Purpose |
|------|---------|
| `ng_lite.py` | Tier 1 learning substrate |
| `ng_peer_bridge.py` | Tier 2 cross-module learning |
| `ng_ecosystem.py` | Tier management lifecycle |
| `ng_autonomic.py` | Autonomic state (**Praxis is read-only**) |
| `openclaw_adapter.py` | OpenClaw skill base class |

---

## 5. Implementation Roadmap

| Phase | Version | What | Status |
|-------|---------|------|--------|
| 1 | v0.1 | Foundation — module structure, config, hook, sensor interfaces | Done |
| 2 | v0.2 | Conversation Stream Sensor — temporal binding, both directions | Done |
| 3 | v0.3 | Context Persistence Store — msgpack, substrate-routed retrieval | Done |
| 4 | v0.4 | Session Bridge — auto context surfacing, usage tracking | Done |
| 5 | v0.5 | Artifact Stream Sensor — lifecycle tracking, reference counting | Done |
| 6 | v0.6 | Outcome Stream Sensor — reward injection, intent tracing | Done |
| 7 | v0.7 | Autonomic & Ecosystem Integration — cold start, integration tests | Done |
| 8 | v0.8 | Hardening & Alpha Release — 79 tests, full validation | **Current** |

---

## 6. What Praxis Does NOT Do

- Text-level AI security (TrollGuard's domain)
- Host-level threat detection (Immunis's domain)
- System repair (THC's domain)
- Substrate maintenance, identity coherence (Elmer's domain)
- Model routing (TID's domain)
- Causal event logging (Bunyan's domain)
- Code generation (external LLM's job)
- Behavioral constraint enforcement (Cricket's domain)

When Praxis detects a signal that requires another module's capability, it records the observation on the substrate. The relevant module absorbs it through the River.

---

## 7. What Claude Code May and May Not Do

### Without Josh's Approval

**Permitted:**
- Read any file in the repo
- Run the test suite
- Edit Praxis-specific files (core/, sensors/, store/)
- Add or modify tests
- Update documentation

**Not permitted without explicit Josh approval:**
- Modify any vendored file
- Delete any file
- Add autonomic state write logic
- Add code generation capabilities
- Add direct inter-module communication

---

## 8. Environment and Paths

| What | Where |
|------|-------|
| Repo root | `~/Praxis/` |
| Configuration | `~/Praxis/config.yaml` |
| Module manifest | `~/Praxis/et_module.json` |
| Module data (runtime) | `~/.et_modules/praxis/` |
| CPS storage | `~/.et_modules/praxis/cps.msgpack` |
| NG-Lite state | `~/.et_modules/praxis/ng_lite_state.json` |
| Shared learning JSONL | `~/.et_modules/shared_learning/praxis.jsonl` |
| OpenClaw workspace | `~/.openclaw/praxis/` |

---

*E-T Systems / Praxis*
*Last updated: 2026-03-18*
*Maintained by Josh — do not edit without authorization*
*Parent documents: `~/.claude/CLAUDE.md` (global), `~/.claude/ARCHITECTURE.md`*
