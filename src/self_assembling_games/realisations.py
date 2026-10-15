"""Concrete state and operations for finite dispositions and inherited plans."""

from __future__ import annotations

import copy
import math
from typing import Any
import numpy as np
from .finite import DecisionProvider, distribution


def validate_finite_realisation(game, method, *, sampled=False):
    """Shared method/game gate used during resolution and construction."""
    if sampled:
        if method.representation != "conditional_decisions":
            raise ValueError(
                "Sampled finite construction requires conditional decisions"
            )
        if game.payoff_matrices is not None or game.parameters.get("listening_cost", 0):
            raise ValueError(
                "Conditional R1 with strategy-level listening/table costs needs a defined mechanism; SKQ3"
            )
        return
    if method.representation == "conditional_decisions" and (
        game.payoff_matrices is not None
        or any(np.any(c) for c in game.strategy_costs.values())
    ):
        raise ValueError(
            "Conditional R1 with strategy-level listening/table costs needs a defined mechanism; SKQ3"
        )
    game.compile()
    if method.nonnegative_returns:
        minimum = (
            min(float(m.min()) for m in game.payoff_matrices)
            if game.payoff_matrices is not None
            else min(
                float(game.returns[:, i].min() - c.max())
                for i, c in enumerate(game.strategy_costs.values())
            )
        )
        if minimum < 0:
            raise ValueError(
                "R1 requires nonnegative numerical returns; no implicit payoff shift"
                if method.method_class == "reinforcement"
                else "Birth mutation needs nonnegative reproduction returns; preserve/declare background fitness"
            )


def state_alias(name):
    """Legacy access refers to the authoritative realisation state."""

    def get(instance):
        return getattr(instance._realisation, name)

    def set_value(instance, value):
        setattr(instance._realisation, name, value)

    return property(get, set_value)


class _Sampling(DecisionProvider):
    def __init__(
        self, realisation: FiniteReinforcementRealisation, rng: np.random.Generator
    ):
        super().__init__(realisation.game)
        self.realisation = realisation
        self.rng = rng
        if realisation.method.representation == "complete_role_plans":
            self.plan_indices = {
                r: self.chance(realisation.weights[r] / realisation.weights[r].sum())
                for r in realisation.organisation.slots
            }

    def chance(self, probabilities: Any) -> int:
        raw = np.asarray(probabilities, dtype=float)
        if raw.ndim != 1:
            raise ValueError("Chance probabilities must be a vector")
        p = distribution(raw, len(raw))
        nonzero = np.flatnonzero(p)
        if len(nonzero) == 1:
            return int(nonzero[0])
        return min(
            int(np.searchsorted(np.cumsum(p), self.rng.random(), side="right")),
            len(p) - 1,
        )

    def choose(self, role: str, context: str) -> int:
        self.realisation.organisation.decision(role, context)
        if self.plan_indices is not None:
            return int(
                self.game.plans[role][
                    self.plan_indices[role], self.game.row_indices[role, context]
                ]
            )
        weights = self.realisation.weights[
            self.realisation.organisation.policy_address(role, context)
        ]
        return self.chance(weights / weights.sum())


class FiniteReinforcementRealisation:
    def __init__(
        self, game, organisation, method, parameters, seed=0, *, sampled=False
    ):
        self.method, self.parameters = method, parameters
        self.organisation = organisation
        self.id = (
            "finite.complete_plan"
            if method.representation == "complete_role_plans"
            else "finite.conditional"
        )
        validate_finite_realisation(game, method, sampled=sampled)
        initial = parameters["initial_weight"]
        self.weights = (
            {
                r: np.full(len(plans), initial, dtype=float)
                for r, plans in self.game.plans.items()
            }
            if self.method.representation == "complete_role_plans"
            else {
                r + "/" + d.context: np.full(len(d.options), initial, dtype=float)
                for r, rows in self.game.rows.items()
                for d in rows
            }
        )
        self.initial_weights = {k: v.copy() for k, v in self.weights.items()}
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.frozen = set()

    @property
    def game(self):
        return self.organisation._game

    @property
    def lineage(self):
        return (
            getattr(self.game, "construction_state", {})
            .get("barrett_source", {})
            .get("lineage", {})
        )

    def encounter(self, game):
        provider = _Sampling(self, self.rng)
        outcome = game.encounter(provider)
        self.adapt(provider, outcome)
        return outcome

    def adapt(self, provider, outcome):
        if provider.plan_indices is not None:
            credits = [
                (r, provider.plan_indices[r], value)
                for r, value in zip(self.organisation.slots, outcome.returns)
                if self.game.rows[r]
            ]
        else:
            credits = (
                [
                    (self.organisation.policy_address(r, c), option, value)
                    for r, c, option, value in provider.credits
                ]
                if provider.credits is not None
                else [
                    (
                        self.organisation.policy_address(r, c),
                        option,
                        outcome.returns[tuple(self.organisation.slots).index(r)],
                    )
                    for r, c, option in provider.draws
                ]
            )
        # Validate the entire update before applying any credit.
        if any(not math.isfinite(value) or value < 0 for _, _, value in credits):
            raise ValueError("R1 requires finite nonnegative reinforcement credit")
        updated = {}
        for address, option, value in credits:
            if address not in self.frozen:
                if address not in updated:
                    updated[address] = self.weights[address].copy()
                updated[address][option] += value
        if any(
            not np.isfinite(w).all() or not math.isfinite(float(w.sum()))
            for w in updated.values()
        ):
            raise OverflowError("Reinforcement weights overflowed")
        self.weights.update(updated)

    def conditional_disposition(self, address):
        """Current act or complete-plan selection at a declared policy address."""
        weights = self.weights[address]
        return weights / weights.sum()

    def inspect_dispositions(self):
        return {
            "implementation": self.id,
            "adaptive_operation": self.method.id,
            "credit_transaction": "validate whole encounter, then apply addressed credits",
            "weights": {k: w.tolist() for k, w in self.weights.items()},
            "plan_selection_distributions"
            if self.id == "finite.complete_plan"
            else "conditional_act_dispositions": {
                k: self.conditional_disposition(k).tolist() for k in self.weights
            },
            "frozen": sorted(self.frozen),
            "fixed_decisions": copy.deepcopy(self.game.fixed_decisions),
            "plan_coordinates": {r: p.tolist() for r, p in self.game.plans.items()}
            if self.id == "finite.complete_plan"
            else None,
        }


class PopulationRealisation:
    id = "population.complete_plan"

    def __init__(self, game, organisation, method, parameters, seed=0):
        self.organisation = organisation
        self.method, self.parameters = method, parameters
        init = self.parameters["initialization"]
        if not isinstance(init, dict) or init.get("kind") not in {
            "dirichlet",
            "uniform",
            "explicit",
        }:
            raise ValueError("Initialization is dirichlet, uniform, or explicit")
        rng = np.random.Generator(np.random.PCG64(seed))
        self.frequencies = {}
        if init["kind"] == "explicit" and set(init.get("frequencies", {})) != set(
            self.population.names
        ):
            raise ValueError("Explicit initialization must supply every population")
        for name in self.population.names:
            count = self.population.counts[name]
            if init["kind"] == "explicit":
                values = distribution(init["frequencies"][name], count)
            elif init["kind"] == "uniform":
                values = np.full(count, 1 / count)
            else:
                concentration = init.get("concentration", 1.0)
                if (
                    not isinstance(concentration, (int, float))
                    or not math.isfinite(concentration)
                    or concentration <= 0
                ):
                    raise ValueError(
                        "Dirichlet concentration must be positive and finite"
                    )
                values = rng.dirichlet(np.full(count, concentration))
                if np.any(values <= 0):
                    raise ValueError(
                        "Dirichlet initialization underflowed; increase concentration"
                    )
            self.frequencies[name] = values
        self.initial_frequencies = {n: x.copy() for n, x in self.frequencies.items()}

    @property
    def population(self):
        return self.organisation.population

    @property
    def game(self):
        return self.organisation._game

    @property
    def lineage(self):
        return (
            getattr(self.game, "construction_state", {})
            .get("barrett_source", {})
            .get("lineage", {})
        )

    def _validate_birth_domain(self) -> None:
        if self.method.id != "replicator_mutator.birth_uniform":
            return
        if self.game.payoff_matrices is not None:
            minimum = min(float(m.min()) for m in self.game.payoff_matrices)
        else:
            minimum = min(
                float(self.game.returns[:, i].min() - cost.max())
                for i, cost in enumerate(self.game.strategy_costs.values())
            )
        if minimum < 0:
            raise ValueError(
                "Birth mutation needs nonnegative reproduction returns; preserve/declare background fitness"
            )

    def _expectations(self, frequencies: dict[str, np.ndarray]) -> dict[str, Any]:
        role_frequencies = self.population.role_marginals(self.game, frequencies)
        result = self.game.expectations(role_frequencies)
        result["population_fitness"] = {
            name: sum(
                self.population.role_weights[r]
                * result["fitness"][r][self.population.components[name][r]]
                for r in self.population.groups[name]
            )
            for name in self.population.names
        }
        return result

    def vector_field(
        self, frequencies: dict[str, np.ndarray] | None = None
    ) -> dict[str, np.ndarray]:
        values = self.frequencies if frequencies is None else frequencies
        for name in self.population.names:
            distribution(values[name], self.population.counts[name])
        result = self._expectations(values)
        return self.field_changes(values, result)

    def field_changes(self, values, result):
        """Selection/transmission law used by diagnostics and the ODE solver."""
        return {
            name: np.zeros_like(values[name])
            if name in self.population.frozen
            else self.population.rates[name]
            * self.method.field(
                values[name],
                result["population_fitness"][name],
                self.parameters["mutation_rate"],
            )
            for name in self.population.names
        }

    def encounter_plan_distributions(self):
        return self.population.role_marginals(self.game, self.frequencies)

    def inspect_dispositions(self):
        return {
            "implementation": self.id,
            "adaptive_operation": self.method.id,
            "mutation_rate": self.parameters["mutation_rate"],
            "frequencies": {k: v.tolist() for k, v in self.frequencies.items()},
            "encounter_plan_distributions": {
                k: v.tolist() for k, v in self.encounter_plan_distributions().items()
            },
            "plan_coordinates": {k: v.tolist() for k, v in self.game.plans.items()},
        }
