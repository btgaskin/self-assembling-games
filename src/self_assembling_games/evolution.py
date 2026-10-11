"""Complete inherited-plan populations and continuous evolutionary dynamics."""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

from .adaptive import adaptive_method
from .finite import FiniteGame, distribution
from .finite_profiles import AdaptiveProfile
from .specification import ModelSpecification
from .model import ModelInstance
from .organisation import Organisation
from .realisations import PopulationRealisation, state_alias


from .organisation import PopulationSpec


class IntegrationDomainError(ValueError):
    """Integration entered an invalid simplex; no state is silently repaired."""


class EvolutionResourceError(ValueError):
    """An explicitly configured evolutionary work budget was exhausted."""


class EvolutionSimulation:
    frequencies = state_alias("frequencies")
    initial_frequencies = state_alias("initial_frequencies")
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

    @property
    def population(self):
        return self._realisation.organisation.population

    @population.setter
    def population(self, value):
        self._realisation.organisation.population = value

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
        if method.method_class != "population_evolution":
            raise ValueError("EvolutionSimulation needs a population evolution method")
        self.game = specification.construct_finite_game()
        organisation = Organisation.for_finite(self.game)
        organisation.population = PopulationSpec(
            self.game,
            **specification.organisation_parameters,
            max_population_strategies=specification.realisation_parameters[
                "resources"
            ].get("max_population_strategies", 100_000),
        )
        self.resources = specification.realisation_parameters["resources"]
        self.seed = seed
        self.protocol = specification.execution_parameters(protocol)
        if self.protocol["solver"] not in {"DOP853", "RK45", "RK23"}:
            raise ValueError(
                "Supported simplex solvers are DOP853, RK45, and RK23; implicit solvers require a tangent-coordinate or analytic-Jacobian implementation"
            )
        self._realisation = PopulationRealisation(
            self.game, organisation, method, self.protocol, seed
        )
        self.model_instance = ModelInstance(
            specification, self.game, organisation, self._realisation
        )
        self.time = 0.0
        # Integrated expected success, returns by role, terms, exchanges.
        self.integrals = np.zeros(len(self.game.roles) + 3)
        self.history: list[dict[str, Any]] = [
            {
                "phase": "initialization",
                "time": 0.0,
                "protocol": copy.deepcopy(self.protocol["initialization"]),
            }
        ]
        self.measurements: list[dict[str, Any]] = []
        self.status = "ready"
        self.metadata = provenance()
        self.configuration_hash = digest(self.profile.to_dict())
        self.game_hash = digest(self.game.to_dict())
        self.fidelity = {
            **copy.deepcopy(self.game.fidelity),
            "population": {
                "status": "explicit_construction",
                "specification": self.population.to_dict(),
                "source_note": "Paper discusses separate/grouped/shared populations; role weights and rates are explicit library choices.",
            },
            "adaptive_law": {
                "status": "paper_defined",
                "method": self.method.id,
                "source": self.method.source,
            },
            "protocol": {
                "status": "library_demonstration",
                "historical_reproduction": "gated_SKQ1",
                "resume": "restart solver; numerical agreement within declared tolerances",
            },
        }
        self.game.compile()
        self._validate_birth_domain()
        self.measurements.append(self.evaluate())

    def _validate_birth_domain(self):
        return self._realisation._validate_birth_domain()

    def _expectations(self, frequencies):
        return self._realisation._expectations(frequencies)

    def vector_field(self, frequencies=None):
        return self._realisation.vector_field(frequencies)

    def evaluate(self, *, tolerance: float = 1e-8) -> dict[str, Any]:
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Diagnostic tolerance must be finite and nonnegative")
        result = self._expectations(self.frequencies)
        residuals, absent, regret, means = {}, {}, {}, {}
        for name in self.population.names:
            x, f = self.frequencies[name], result["population_fitness"][name]
            means[name] = float(x @ f)
            field = (
                np.zeros_like(x)
                if name in self.population.frozen
                else self.population.rates[name]
                * self.method.field(x, f, self.protocol["mutation_rate"])
            )
            residuals[name] = float(np.max(np.abs(field)))
            absent[name] = (
                float(np.max(f[x == 0] - means[name])) if np.any(x == 0) else None
            )
            regret[name] = max(0.0, float(np.max(f) - means[name]))
        stationary = max(residuals.values()) <= tolerance
        return {
            "protocol": "exact_finite",
            "time": self.time,
            "success_probability": result["success_probability"],
            "expected_returns": result["expected_returns"],
            "expected_terms": result["expected_terms"],
            "expected_exchanges": result["expected_exchanges"],
            "completed_mass": result["completed_mass"],
            "mean_reproduction_fitness": means,
            "stationarity_residual": residuals,
            "diagnostic_tolerance": tolerance,
            "absent_strategy_invasion_advantage": absent,
            "selection_regret": regret,
            "candidate_classification": (
                "mutation_selection_stationary_candidate"
                if self.method.id == "replicator_mutator.birth_uniform"
                else "selection_stationary_candidate"
            )
            if stationary
            else "nonstationary",
            "stability_assessed": False,
        }

    def run(
        self, *, duration: float | None = None, output=None, **invalid
    ) -> dict[str, Any]:
        from scipy.integrate import solve_ivp
        from importlib.metadata import version

        if invalid:
            raise ValueError(
                "Evolution runs advance duration in evolutionary time, not plays"
            )
        duration = self.protocol["duration"] if duration is None else duration
        if (
            not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration < 0
        ):
            raise ValueError("duration must be finite and nonnegative")
        self.metadata["scipy"] = version("scipy")
        names = self.population.names
        sizes = [self.population.counts[n] for n in names]
        ends = np.cumsum([0] + sizes)
        base_size = int(ends[-1])
        y = np.concatenate([*(self.frequencies[n] for n in names), self.integrals])
        target = self.time + duration
        if not math.isfinite(target) or (duration > 0 and target == self.time):
            raise ValueError(
                "Evolution duration cannot be represented by this time clock"
            )
        accepted_max_step = self.protocol["max_step"]
        calls = 0

        def field(t, state):
            nonlocal calls
            calls += 1
            if calls > self.resources.get("max_field_evaluations", 200_000):
                raise EvolutionResourceError("Evolution exceeded max_field_evaluations")
            values = {n: state[ends[i] : ends[i + 1]] for i, n in enumerate(names)}
            for n, x in values.items():
                if (
                    not np.isfinite(x).all()
                    or np.any(x < 0)
                    or abs(float(x.sum()) - 1) > max(1e-8, 100 * self.protocol["atol"])
                ):
                    raise IntegrationDomainError(
                        f"Solver left the {n} simplex; reduce max_step or tighten tolerances"
                    )
            result = self._expectations(values)
            changes = list(self._realisation.field_changes(values, result).values())
            accumulators = [
                result["success_probability"],
                *(result["expected_returns"][r] for r in self.game.roles),
                result["expected_terms"],
                result["expected_exchanges"],
            ]
            return np.concatenate([*changes, np.array(accumulators)])

        start = self.time
        if len(self.measurements) + math.ceil(
            duration / self.protocol["sample_interval"]
        ) > self.resources.get("max_measurements", 10_000):
            raise ValueError("Evolution report samples exceed max_measurements")
        try:
            while self.time < target:
                end = min(target, self.time + self.protocol["sample_interval"])
                for attempt in range(12):
                    try:
                        solution = solve_ivp(
                            field,
                            (self.time, end),
                            y,
                            method=self.protocol["solver"],
                            rtol=self.protocol["rtol"],
                            atol=self.protocol["atol"],
                            max_step=accepted_max_step,
                        )
                        if not solution.success:
                            raise IntegrationDomainError(solution.message)
                        candidate = solution.y[:, -1]
                        if not np.isfinite(candidate).all():
                            raise IntegrationDomainError("Nonfinite integrated state")
                        for i, n in enumerate(names):
                            distribution(candidate[ends[i] : ends[i + 1]], sizes[i])
                        break
                    except (IntegrationDomainError, ValueError) as error:
                        if isinstance(error, EvolutionResourceError):
                            raise
                        if attempt == 11:
                            raise IntegrationDomainError(str(error)) from error
                        accepted_max_step /= 2
                y = candidate
                self.frequencies = {
                    n: y[ends[i] : ends[i + 1]].copy() for i, n in enumerate(names)
                }
                self.integrals = y[base_size:].copy()
                self.time = end
                self.measurements.append(self.evaluate())
            self.status = "complete"
        except (IntegrationDomainError, EvolutionResourceError) as error:
            self.status = (
                "resource_limit"
                if isinstance(error, EvolutionResourceError)
                else "integration_failed"
            )
            raise
        finally:
            self.history.append(
                {
                    "phase": "integration_segment",
                    "start": start,
                    "end": self.time,
                    "requested_end": target,
                    "solver": self.protocol["solver"],
                    "rtol": self.protocol["rtol"],
                    "atol": self.protocol["atol"],
                    "configured_max_step": self.protocol["max_step"],
                    "accepted_max_step": accepted_max_step,
                    "field_evaluations": calls,
                    "status": self.status,
                }
            )
            if output:
                self.checkpoint(output)
        return self.summary()

    def summary(self) -> dict[str, Any]:
        return {
            **self.evaluate(),
            "clock": "evolutionary_time",
            "status": self.status,
            "time_average_expected_success": float(self.integrals[0] / self.time)
            if self.time
            else None,
            "time_average_expected_returns": {
                r: float(self.integrals[1 + i] / self.time) if self.time else None
                for i, r in enumerate(self.game.roles)
            },
            "time_average_expected_terms": float(self.integrals[-2] / self.time)
            if self.time
            else None,
            "time_average_expected_exchanges": float(self.integrals[-1] / self.time)
            if self.time
            else None,
        }

    def set_population(
        self, name: str, frequencies: Any, *, frozen: bool | None = None
    ) -> None:
        if name not in self.population.names:
            raise ValueError(f"Unknown population {name}")
        self.frequencies[name] = distribution(frequencies, self.population.counts[name])
        if frozen is True:
            self.population.frozen.add(name)
        elif frozen is False:
            self.population.frozen.discard(name)
        self.history.append(
            {
                "phase": "population_intervention",
                "time": self.time,
                "population": name,
                "frequencies": self.frequencies[name].tolist(),
                "frozen": name in self.population.frozen,
            }
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
                    k: v
                    for k, v in self.protocol.items()
                    if k not in {"initialization", "mutation_rate"}
                },
                "engine": "adaptive",
                "method_class": "population_evolution",
                "profile": self.profile.to_dict(),
                "configuration_hash": self.configuration_hash,
                "game_hash": self.game_hash,
                "game": self.game.to_dict(),
                "fidelity": self.fidelity,
                "seed": self.seed,
                "metadata": self.metadata,
                "protocol": self.protocol,
                "population": self.population.to_dict(),
                "time": self.time,
                "frequencies": {n: x.tolist() for n, x in self.frequencies.items()},
                "initial_frequencies": {
                    n: x.tolist() for n, x in self.initial_frequencies.items()
                },
                "integrals": self.integrals.tolist(),
                "history": self.history,
                "measurements": self.measurements,
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
