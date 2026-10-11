"""Conditional and complete-plan reinforcement of shared finite interactions."""

from __future__ import annotations

import copy
import math
from collections import deque
from typing import Any

import numpy as np

from .adaptive import adaptive_method
from .finite import DecisionProvider, FiniteGame, distribution
from .finite_profiles import AdaptiveProfile, finite_game, load_adaptive_profile


class _Sampling(DecisionProvider):
    def __init__(self, simulation: ReinforcementSimulation, rng: np.random.Generator):
        super().__init__(simulation.game)
        self.simulation = simulation
        self.rng = rng
        if simulation.method.representation == "complete_role_plans":
            self.plan_indices = {
                r: self.chance(simulation.weights[r] / simulation.weights[r].sum())
                for r in self.game.roles
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
        if self.plan_indices is not None:
            return int(
                self.game.plans[role][
                    self.plan_indices[role], self.game.row_indices[role, context]
                ]
            )
        weights = self.simulation.weights[role + "/" + context]
        return self.chance(weights / weights.sum())


class ReinforcementSimulation:
    def __init__(
        self,
        profile: AdaptiveProfile | str | dict[str, Any],
        seed: int = 0,
        *,
        game: FiniteGame | None = None,
    ):
        from .simulation import digest, provenance

        self.profile = load_adaptive_profile(profile)
        self.method = adaptive_method(self.profile.method)
        if self.method.method_class != "reinforcement":
            raise ValueError("ReinforcementSimulation needs a reinforcement method")
        self.game = game if game is not None else finite_game(self.profile)
        self.seed = seed
        self.protocol = {
            "plays": 10_000,
            "initial_weight": 1.0,
            "window": 1000,
            **copy.deepcopy(self.profile.protocol),
        }
        if self.method.representation == "conditional_decisions" and (
            self.game.payoff_matrices is not None
            or any(np.any(c) for c in self.game.strategy_costs.values())
        ):
            raise ValueError(
                "Conditional R1 with strategy-level listening/table costs needs a defined mechanism; SKQ3"
            )
        self.game.compile()
        if self.game.payoff_matrices is not None:
            minimum = min(float(m.min()) for m in self.game.payoff_matrices)
        else:
            minimum = min(
                float(self.game.returns[:, i].min() - c.max())
                for i, c in enumerate(self.game.strategy_costs.values())
            )
        if minimum < 0:
            raise ValueError(
                "R1 requires nonnegative numerical returns; no implicit payoff shift"
            )
        initial = self.protocol["initial_weight"]
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
        self.frozen: set[str] = set()
        self.plays = 0
        self.successes = 0
        self.return_totals = {r: 0.0 for r in self.game.roles}
        self.terms = 0
        self.exchanges = 0
        self.recent: deque[bool] = deque(maxlen=self.protocol["window"])
        self.history: list[dict[str, Any]] = [
            {"phase": "initialization", "play": 0, "initial_weight": initial}
        ]
        self.status = "ready"
        self.metadata = provenance()
        self.configuration_hash = digest(self.profile.to_dict())
        self.game_hash = digest(self.game.to_dict())
        self.fidelity = {
            **copy.deepcopy(self.game.fidelity),
            "population": {
                "status": "role_participants",
                "matching": "one participant per role",
            },
            "adaptive_law": {
                "status": "library_instantiation",
                "method": self.method.id,
                "source": self.method.source,
            },
            "protocol": {
                "status": "library_demonstration",
                "historical_reproduction": "gated_SKQ1",
                "resume": "bitwise continuation under pinned implementation",
            },
        }

    def step(self):
        provider = _Sampling(self, self.rng)
        outcome = self.game.encounter(provider)
        if provider.plan_indices is not None:
            credits = [
                (r, provider.plan_indices[r], value)
                for r, value in zip(self.game.roles, outcome.returns)
                if self.game.rows[r]
            ]
        else:
            credits = (
                [
                    (r + "/" + c, option, value)
                    for r, c, option, value in provider.credits
                ]
                if provider.credits is not None
                else [
                    (r + "/" + c, option, outcome.returns[self.game.roles.index(r)])
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
        self.plays += 1
        self.successes += int(outcome.success)
        self.recent.append(outcome.success)
        for r, value in zip(self.game.roles, outcome.returns):
            self.return_totals[r] += value
        self.terms += outcome.terms
        self.exchanges += outcome.exchanges
        return outcome

    def run(
        self, *, plays: int | None = None, output=None, **invalid
    ) -> dict[str, Any]:
        if invalid:
            raise ValueError(
                "Reinforcement runs advance plays, not evolutionary duration"
            )
        plays = self.protocol["plays"] if plays is None else plays
        if not isinstance(plays, int) or isinstance(plays, bool) or plays < 0:
            raise ValueError("plays must be a nonnegative integer")
        start = self.plays
        for _ in range(plays):
            self.step()
        self.status = "complete"
        self.history.append(
            {"phase": "learning_segment", "start": start, "end": self.plays}
        )
        if output:
            self.checkpoint(output)
        return self.summary()

    def evaluate(
        self, *, protocol: str = "exact", samples: int = 1000, seed: int = 0
    ) -> dict[str, Any]:
        if protocol == "sampled":
            if not isinstance(samples, int) or samples <= 0:
                raise ValueError("samples must be a positive integer")
            rng = np.random.Generator(np.random.PCG64(seed))
            results = [
                self.game.encounter(_Sampling(self, rng)) for _ in range(samples)
            ]
            return {
                "protocol": "sampled_frozen",
                "samples": samples,
                "seed": seed,
                "success_probability": sum(o.success for o in results) / samples,
                "expected_returns": {
                    r: sum(o.returns[i] for o in results) / samples
                    for i, r in enumerate(self.game.roles)
                },
            }
        if protocol != "exact":
            raise ValueError("Finite reinforcement evaluation is exact or sampled")
        normalized = {k: w / w.sum() for k, w in self.weights.items()}
        result = (
            self.game.expectations(normalized)
            if self.method.representation == "complete_role_plans"
            else self.game.conditional_expectations(normalized)
        )
        return {
            "protocol": "exact_finite",
            **{k: v for k, v in result.items() if k != "fitness"},
        }

    def summary(self) -> dict[str, Any]:
        return {
            "clock": "plays",
            "attempts": self.plays,
            "completed": self.plays,
            "successes": self.successes,
            "status": self.status,
            "cumulative_accuracy": self.successes / self.plays if self.plays else None,
            "window_accuracy": sum(self.recent) / len(self.recent)
            if self.recent
            else None,
            "window_completed": len(self.recent),
            "mean_returns": {
                r: total / self.plays if self.plays else None
                for r, total in self.return_totals.items()
            },
            "mean_terms": self.terms / self.plays if self.plays else None,
            "mean_exchanges": self.exchanges / self.plays if self.plays else None,
            "final_policy": self.evaluate(),
        }

    def set_policy(
        self, address: str, probabilities: Any, *, frozen: bool = True
    ) -> None:
        if address not in self.weights:
            raise ValueError(f"Unknown policy address {address}")
        p = distribution(probabilities, len(self.weights[address]))
        self.weights[address] = p * self.weights[address].sum()
        if frozen:
            self.frozen.add(address)
        else:
            self.frozen.discard(address)
        self.history.append(
            {
                "phase": "policy_intervention",
                "play": self.plays,
                "address": address,
                "probabilities": p.tolist(),
                "frozen": frozen,
            }
        )

    def freeze_role(self, role: str, *, frozen: bool = True) -> None:
        if role not in self.game.roles:
            raise ValueError(f"Unknown role {role}")
        addresses = (
            [role]
            if self.method.representation == "complete_role_plans"
            else [role + "/" + row.context for row in self.game.rows[role]]
        )
        for address in addresses:
            if frozen:
                self.frozen.add(address)
            else:
                self.frozen.discard(address)
        self.history.append(
            {"phase": "freeze_role", "play": self.plays, "role": role, "frozen": frozen}
        )

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(
            {
                "schema_version": 1,
                "engine": "adaptive",
                "method_class": "reinforcement",
                "profile": self.profile.to_dict(),
                "configuration_hash": self.configuration_hash,
                "game_hash": self.game_hash,
                "game": self.game.to_dict(),
                "fidelity": self.fidelity,
                "seed": self.seed,
                "metadata": self.metadata,
                "protocol": self.protocol,
                "weights": {k: w.tolist() for k, w in self.weights.items()},
                "initial_weights": {
                    k: w.tolist() for k, w in self.initial_weights.items()
                },
                "rng_state": self.rng.bit_generator.state,
                "frozen": sorted(self.frozen),
                "plays": self.plays,
                "successes": self.successes,
                "return_totals": self.return_totals,
                "terms": self.terms,
                "exchanges": self.exchanges,
                "recent": list(self.recent),
                "history": self.history,
                "status": self.status,
            }
        )

    def checkpoint(self, path):
        from .adaptive_runtime import checkpoint

        return checkpoint(self.snapshot(), path)

    def inspect(self) -> dict[str, Any]:
        return {**self.snapshot(), "assessment": self.summary()}
