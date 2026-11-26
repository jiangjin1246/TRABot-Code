"""Seeded random Planner interface for the first multi-Atom dialogue MVP."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

from .atom_catalog import AtomCatalog, CatalogAtom


RANDOM_STRATEGY = "random_v0"


@dataclass(frozen=True, slots=True)
class TurnPlan:
    turn_id: str
    strategy: str
    seed: int
    atoms: tuple[CatalogAtom, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the intentionally small future-Planner wire projection."""

        return {
            "turn_id": self.turn_id,
            "strategy": self.strategy,
            "seed": self.seed,
            "atoms": [
                {"atom_id": atom.atom_id, "revision": atom.revision}
                for atom in self.atoms
            ],
        }

    def to_log_record(self) -> dict[str, Any]:
        """Return exact bindings suitable for the per-turn audit log."""

        return {
            "turn_id": self.turn_id,
            "strategy": self.strategy,
            "seed": self.seed,
            "atoms": [
                {
                    "atom_id": atom.atom_id,
                    "version": atom.version,
                    "revision": atom.revision,
                    "content_sha256": atom.content_sha256,
                }
                for atom in self.atoms
            ],
        }


class RandomSequencePlanner:
    """Choose a reproducible sequence without repeating an Atom in one turn."""

    strategy = RANDOM_STRATEGY

    def plan(
        self,
        *,
        turn_id: str,
        atom_catalog: AtomCatalog,
        count: int = 3,
        seed: int,
    ) -> TurnPlan:
        if not isinstance(turn_id, str) or not turn_id.strip():
            raise ValueError("turn_id must be a non-empty string")
        if not isinstance(atom_catalog, AtomCatalog):
            raise TypeError("atom_catalog must be an AtomCatalog")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("count must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        approved = atom_catalog.approved_atoms()
        if count > len(approved):
            raise ValueError(
                f"count={count} exceeds {len(approved)} approved catalog Atoms"
            )
        selected = tuple(random.Random(seed).sample(list(approved), count))
        return TurnPlan(
            turn_id=turn_id,
            strategy=self.strategy,
            seed=seed,
            atoms=selected,
        )


random_planner = RandomSequencePlanner()


__all__ = [
    "RANDOM_STRATEGY",
    "RandomSequencePlanner",
    "TurnPlan",
    "random_planner",
]
