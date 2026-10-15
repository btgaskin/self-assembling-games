"""Operative relations among components, participants and populations."""

from __future__ import annotations

import copy
import math
import numpy as np
from .finite import FiniteGame, distribution
from typing import Any


class Organisation:
    """One backing store for component ownership and functional roles.

    Encounter slots identify independently sampled participants. A population
    layout specifies joint inheritance and matching through ``PopulationSpec``.
    Neither relation silently supplies an organism identity.
    """

    def __init__(self):
        self.owners: dict[str, str] = {}
        self.module_roles: dict[str, str] = {}
        self._game = None
        self.population = None

    def declare(self, module: str, owner: str, role: str) -> None:
        self.owners[module] = owner
        self.module_roles[module] = role

    @classmethod
    def for_finite(cls, game):
        organisation = cls()
        organisation._game = game
        return organisation

    @property
    def slots(self):
        """Current bindings from the operative finite decision records."""
        return (
            {}
            if self._game is None
            else {
                slot: tuple(
                    dict.fromkeys(
                        d.module for d in self._game.decisions if d.role == slot
                    )
                )
                for slot in self._game.roles
            }
        )

    def decision(self, slot, context):
        if slot not in self.slots:
            raise ValueError(f"Unknown encounter slot {slot}")
        return self._game.decision(slot, context)

    def policy_address(self, slot, context):
        self.decision(slot, context)
        return slot + "/" + context

    def inventory(self, realisation):
        """Discover live component state without retaining another inventory."""
        if hasattr(realisation, "tables"):
            return {
                module: [
                    address
                    for address, table in realisation.tables.items()
                    if table.module == module
                ]
                for module in self.owners
            }
        if realisation.id == "finite.complete_plan":
            return {slot: [slot] for slot in self.slots}
        if realisation.id == "population.complete_plan":
            return {name: [name] for name in self.population.names}
        if realisation.id == "population.multilocus":
            return {
                "population": ["frequencies"]
                + (["counts"] if realisation.finite else [])
            }
        return {
            slot: [
                self.policy_address(slot, row.context)
                for row in self._game.decisions
                if row.slot == slot
                and self.policy_address(slot, row.context) in realisation.weights
            ]
            for slot in self.slots
        }

    def decision_bindings(self, realisation):
        """Discover the contexts and modules attached to each encounter slot."""
        if hasattr(realisation, "tables"):
            return {
                module: {
                    address: copy.deepcopy(table.context)
                    for address, table in realisation.tables.items()
                    if table.module == module
                }
                for module in self.owners
            }
        return {
            slot: {
                row.context: row.module
                for row in self._game.decisions
                if row.slot == slot
            }
            for slot in self.slots
        }

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(
            {
                "owners": self.owners,
                "module_roles": self.module_roles,
                "encounter_slots": {k: list(v) for k, v in self.slots.items()},
                "population": self.population.to_dict() if self.population else None,
            }
        )


class PopulationSpec:
    """A distinct individual occupies every slot, including shared populations."""

    def __init__(
        self,
        game: FiniteGame,
        *,
        groups: dict[str, list[str]] | None = None,
        role_weights: dict[str, float] | None = None,
        rates: dict[str, float] | None = None,
        frozen: list[str] | None = None,
        matching: str = "independent_individuals",
        max_population_strategies: int = 100_000,
        enumerate_components: bool = True,
    ):
        self.groups = copy.deepcopy(
            groups if groups is not None else {r: [r] for r in game.roles}
        )
        if not self.groups or any(
            not name or not roles for name, roles in self.groups.items()
        ):
            raise ValueError("Populations need nonempty names and inherited role sets")
        roles = [r for rs in self.groups.values() for r in rs]
        if sorted(roles) != sorted(game.roles):
            raise ValueError("Every role must belong to exactly one population")
        if matching != "independent_individuals":
            raise ValueError(
                "Supported matching samples a distinct independent individual per role slot"
            )
        self.names = tuple(sorted(self.groups))
        self.matching = matching
        self.role_population = {r: name for name, rs in self.groups.items() for r in rs}
        default_weights = {r: 1 / len(rs) for rs in self.groups.values() for r in rs}
        self.role_weights = dict(
            default_weights if role_weights is None else role_weights
        )
        if set(self.role_weights) != set(game.roles):
            raise ValueError("role_weights must cover all role slots")
        for rs in self.groups.values():
            distribution([self.role_weights[r] for r in rs], len(rs))
        self.rates = {name: 1.0 for name in self.names}
        if rates is not None:
            if set(rates) - set(self.names):
                raise ValueError("Unknown encounter-rate population")
            self.rates.update(rates)
        if any(not math.isfinite(v) or v <= 0 for v in self.rates.values()):
            raise ValueError("Encounter rates must be positive and finite")
        self.frozen = set(frozen or [])
        if not self.frozen <= set(self.names):
            raise ValueError("Unknown frozen population")
        self.components = {}
        self.counts = {}
        for name in self.names:
            rs = self.groups[name]
            sizes = [game.plan_counts[r] for r in rs]
            count = math.prod(sizes)
            if enumerate_components and count > max_population_strategies:
                raise ValueError(
                    f"{name} needs {count} inherited types; max_population_strategies="
                    f"{max_population_strategies}"
                )
            self.counts[name] = count
            if not enumerate_components:
                continue
            index = np.arange(count)
            self.components[name] = {
                r: (index // math.prod(sizes[k + 1 :])) % sizes[k]
                for k, r in enumerate(rs)
            }

    def to_dict(self) -> dict[str, Any]:
        return {
            "groups": copy.deepcopy(self.groups),
            "role_weights": dict(self.role_weights),
            "rates": dict(self.rates),
            "frozen": sorted(self.frozen),
            "matching": self.matching,
            "strategy_counts": dict(self.counts),
            "inheritance": "joint plans for declared roles; lexicographic role-plan indices",
        }

    def role_marginals(
        self, game: FiniteGame, frequencies: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        return {
            r: np.bincount(
                self.components[self.role_population[r]][r],
                weights=frequencies[self.role_population[r]],
                minlength=len(game.plans[r]),
            )
            for r in game.roles
        }


class MultilocusPopulationSpec:
    """Encounter positions of one haploid population, sampled with replacement.

    Each group shares a genotype across all its slots. Separate groups draw
    independently. Return weights specify their contributions to reproduction.
    """

    def __init__(
        self,
        game,
        *,
        individuals=None,
        return_weights=None,
        sampling="with_replacement",
    ):
        self.individuals = copy.deepcopy(
            individuals if individuals is not None else {s: [s] for s in game.roles}
        )
        if (
            not isinstance(self.individuals, dict)
            or not self.individuals
            or any(
                not isinstance(n, str)
                or not n
                or not isinstance(slots, list)
                or not slots
                for n, slots in self.individuals.items()
            )
        ):
            raise ValueError("Individuals need named, nonempty encounter-slot bindings")
        slots = [s for group in self.individuals.values() for s in group]
        if len(slots) != len(game.roles) or set(slots) != set(game.roles):
            raise ValueError(
                "Individual bindings must partition encounter slots exactly once"
            )
        if sampling != "with_replacement":
            raise ValueError("Multilocus encounter sampling is with_replacement")
        self.sampling = sampling
        self.return_weights = copy.deepcopy(
            return_weights
            if return_weights is not None
            else {s: 1 / len(game.roles) for s in game.roles}
        )
        if self.return_weights.keys() != game.plans.keys():
            raise ValueError("return_weights must cover every encounter slot")
        distribution([self.return_weights[s] for s in game.roles], len(game.roles))
        self.slot_individual = {
            s: name for name, group in self.individuals.items() for s in group
        }

    def to_dict(self):
        return {
            "individuals": copy.deepcopy(self.individuals),
            "return_weights": copy.deepcopy(self.return_weights),
            "sampling": self.sampling,
        }
