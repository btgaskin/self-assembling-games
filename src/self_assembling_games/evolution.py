"""Complete inherited-plan populations and continuous evolutionary dynamics."""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

from .adaptive import adaptive_method
from .finite import FiniteGame, distribution
from .finite_profiles import AdaptiveProfile, finite_game, load_adaptive_profile


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
            sizes = [len(game.plans[r]) for r in rs]
            count = math.prod(sizes)
            if count > max_population_strategies:
                raise ValueError(
                    f"{name} needs {count} inherited types; max_population_strategies="
                    f"{max_population_strategies}"
                )
            index = np.arange(count)
            self.counts[name] = count
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


class IntegrationDomainError(ValueError):
    """Integration entered an invalid simplex; no state is silently repaired."""


class EvolutionResourceError(ValueError):
    """An explicitly configured evolutionary work budget was exhausted."""


class EvolutionSimulation:
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
        if self.method.method_class != "population_evolution":
            raise ValueError("EvolutionSimulation needs a population evolution method")
        self.game = game if game is not None else finite_game(self.profile)
        self.population = PopulationSpec(
            self.game,
            **self.profile.population,
            max_population_strategies=self.profile.resources.get(
                "max_population_strategies", 100_000
            ),
        )
        self.seed = seed
        self.protocol = {
            "duration": 50.0,
            "initialization": {"kind": "dirichlet", "concentration": 1.0},
            "solver": "DOP853",
            "rtol": 1e-9,
            "atol": 1e-12,
            "max_step": 1.0,
            "sample_interval": 1.0,
            "mutation_rate": 0.0,
            **copy.deepcopy(self.profile.protocol),
        }
        if self.protocol["solver"] not in {
            "DOP853",
            "RK45",
            "RK23",
        }:
            raise ValueError(
                "Supported simplex solvers are DOP853, RK45, and RK23; implicit solvers require a tangent-coordinate or analytic-Jacobian implementation"
            )
        init = self.protocol["initialization"]
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
        self.time = 0.0
        # Integrated expected success, returns by role, terms, exchanges.
        self.integrals = np.zeros(len(self.game.roles) + 3)
        self.history: list[dict[str, Any]] = [
            {"phase": "initialization", "time": 0.0, "protocol": copy.deepcopy(init)}
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
        return {
            name: np.zeros_like(values[name])
            if name in self.population.frozen
            else self.population.rates[name]
            * self.method.field(
                values[name],
                result["population_fitness"][name],
                self.protocol["mutation_rate"],
            )
            for name in self.population.names
        }

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
            if calls > self.profile.resources.get("max_field_evaluations", 200_000):
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
            changes = [
                np.zeros_like(values[n])
                if n in self.population.frozen
                else self.population.rates[n]
                * self.method.field(
                    values[n],
                    result["population_fitness"][n],
                    self.protocol["mutation_rate"],
                )
                for n in names
            ]
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
        ) > self.profile.resources.get("max_measurements", 10_000):
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
        return copy.deepcopy(
            {
                "schema_version": 1,
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
        return {**self.snapshot(), "assessment": self.summary()}
