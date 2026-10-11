"""Conditional and complete-plan reinforcement of shared finite interactions."""

from __future__ import annotations

import copy
from collections import deque
from typing import Any

import numpy as np

from .adaptive import adaptive_method
from .finite import FiniteGame, distribution
from .finite_profiles import AdaptiveProfile
from .specification import ModelSpecification
from .model import ModelInstance
from .organisation import Organisation
from .realisations import FiniteReinforcementRealisation, _Sampling, state_alias


class ReinforcementSimulation:
    weights = state_alias("weights")
    initial_weights = state_alias("initial_weights")
    rng = state_alias("rng")
    frozen = state_alias("frozen")
    method = state_alias("method")

    @property
    def game(self):
        return (
            self.model_instance.game if hasattr(self, "model_instance") else self._game
        )

    @game.setter
    def game(self, value):
        if hasattr(self, "model_instance"):
            raise ValueError(
                "Construct a game override through game= to resolve its specification"
            )
        self._game = value

    def __init__(
        self,
        profile: AdaptiveProfile | str | dict[str, Any],
        seed: int = 0,
        *,
        game: FiniteGame | None = None,
        protocol=None,
    ):
        from .simulation import digest, provenance

        specification = (
            profile
            if isinstance(profile, ModelSpecification)
            else ModelSpecification.from_profile(profile, game=game)
        )
        if isinstance(profile, ModelSpecification) and game is not None:
            specification = specification.with_game(game)
        self.profile = specification.to_profile(protocol)
        method = adaptive_method(specification.realisation_parameters["method"])
        if method.method_class != "reinforcement":
            raise ValueError("ReinforcementSimulation needs a reinforcement method")
        self.game = specification.construct_finite_game()
        self.seed = seed
        self.protocol = specification.execution_parameters(protocol)
        organisation = Organisation.for_finite(self.game)
        self._realisation = FiniteReinforcementRealisation(
            self.game, organisation, method, self.protocol, seed
        )
        self.model_instance = ModelInstance(
            specification, self.game, organisation, self._realisation
        )
        self.plays = 0
        self.successes = 0
        self.return_totals = {r: 0.0 for r in self.game.roles}
        self.terms = 0
        self.exchanges = 0
        self.recent: deque[bool] = deque(maxlen=self.protocol["window"])
        self.history: list[dict[str, Any]] = [
            {
                "phase": "initialization",
                "play": 0,
                "initial_weight": self.protocol["initial_weight"],
            }
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
        outcome = self.model_instance.encounter()
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
                self.game.encounter(_Sampling(self._realisation, rng))
                for _ in range(samples)
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
        from .simulation import digest

        return copy.deepcopy(
            {
                "schema_version": 1,
                "model_specification": self.model_instance.specification.to_dict(),
                "model_specification_hash": digest(
                    self.model_instance.specification.to_dict()
                ),
                "construction_state": getattr(self.game, "construction_state", {}),
                "model_protocol": {
                    k: v for k, v in self.protocol.items() if k != "initial_weight"
                },
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
        return {
            **self.snapshot(),
            "model_instance": self.model_instance.inspect(),
            "assessment": self.summary(),
        }
