"""CreationBridge — Morphogenesis→Praxis integration.

Praxis detects creation intent in natural language. CreationBridge calls
Morphogenesis to grow a living organism and package it as a .morpho file.

The .morpho file is a holographic boundary — a self-contained install
package that regrows the organism on any machine. No source code ships.

# ---- Changelog ----
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
from typing import List, Optional

try:
    import morphogenesis
    import numpy as np
    from morphogenesis.praxis import PraxisEngine
    from morphogenesis.intent import OrganismIntent
    from morphogenesis.compiler import grow_organism
    from morphogenesis.holographic import package_organism, save_morpho, inspect_morpho
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

    def grow(
        self,
        description: str,
        seed: Optional[int] = None,
        output_dir: Optional[str] = None,
        _override_intent: dict = None,
    ) -> GrowResult:
        """Grow a living organism from a natural language description.

        Extracts intent using PraxisEngine, grows the organism, and packages
        it as a .morpho holographic boundary file.

        Args:
            description:      Natural language description of desired behavior.
            seed:             Random seed for reproducible growth.
            output_dir:       Directory to write the .morpho file.
                              Defaults to ~/.et_modules/praxis/organisms/
            _override_intent: Dict to use instead of NL extraction (for testing).

        Returns:
            GrowResult with path to the .morpho file and summary metadata.

        Raises:
            ValueError: If the organism dies during growth.
        """
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

        # Packaging
        boundary = package_organism(organism)

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
        )
