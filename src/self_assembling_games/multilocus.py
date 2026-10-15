"""Haplotype recursions and Wright–Fisher censuses for realised finite games."""

from __future__ import annotations

import copy
import math

import numpy as np

from .adaptive import adaptive_method
from .finite import distribution
from .genetics import GenomeSpec, ExpressionMap, TransmissionSpec, positive_integer
from .model import ModelInstance
from .organisation import Organisation, MultilocusPopulationSpec


GENETIC_FIELDS = {"genome", "expression", "transmission", "fitness", "population_size"}
RESOURCE_DEFAULTS = {
    "max_haplotypes": 4096,
    "max_recombination_patterns": 65536,
    "max_transmission_work": 10_000_000,
    "max_genotype_mask_entries": 10_000_000,
    "max_measurements": 100_000,
    "max_history_entries": 100_000,
    "max_generation_work": 50_000_000,
}
LIFE_CYCLE = (
    "expression_and_assay",
    "selection",
    "recombination",
    "mutation",
    "census",
)


def resolve_genetics(
    game,
    method,
    genetics,
    organisation,
    initialization,
    resources,
    *,
    declaration_only=False,
):
    """Shared construction gates, with declared orders and effective defaults."""
    genetics = copy.deepcopy(genetics)
    if (
        genetics.keys() - GENETIC_FIELDS
        or not {"genome", "expression"} <= genetics.keys()
    ):
        raise ValueError("Multilocus genetics requires genome and expression")
    limits = {**RESOURCE_DEFAULTS, **resources}
    for k, v in limits.items():
        positive_integer(v, k)
    genome = GenomeSpec.from_dict(
        genetics["genome"], max_haplotypes=limits["max_haplotypes"]
    )
    if not declaration_only:
        game.compile()
    if organisation.keys() - {"individuals", "return_weights", "sampling"}:
        raise ValueError(
            "Unknown multilocus organisation fields: "
            + ", ".join(
                sorted(
                    organisation.keys() - {"individuals", "return_weights", "sampling"}
                )
            )
        )
    binding = MultilocusPopulationSpec(game, **organisation)
    if (
        not declaration_only
        and len(game.chance_mass) * genome.count * len(binding.individuals)
        > limits["max_genotype_mask_entries"]
    ):
        raise ValueError("Expression exceeds max_genotype_mask_entries")
    expression = ExpressionMap(genome, game, genetics["expression"])
    transmission = TransmissionSpec(
        genome,
        genetics.get("transmission"),
        max_recombination_patterns=limits["max_recombination_patterns"],
        max_transmission_work=limits["max_transmission_work"],
    )
    if not declaration_only:
        work = (
            genome.count
            * len(game.chance_mass)
            * (2 * len(game.roles) + len(binding.individuals))
            + transmission.estimated_work
        )
        if game.payoff_matrices is not None:
            work += sum(matrix.size for matrix in game.payoff_matrices)
        if work > limits["max_generation_work"]:
            raise ValueError("Genotype contractions exceed max_generation_work")
    fitness = copy.deepcopy(genetics.get("fitness", {"background": 1.0, "scale": 1.0}))
    if set(fitness) != {"background", "scale"} or any(
        not isinstance(v, (int, float))
        or isinstance(v, bool)
        or not math.isfinite(v)
        or v < 0
        for v in fitness.values()
    ):
        raise ValueError("Fitness background and scale must be finite and nonnegative")
    finite = method.id == "wright_fisher.haploid"
    if finite:
        size = positive_integer(genetics.get("population_size"), "population_size")
        if size > np.iinfo(np.int64).max:
            raise ValueError("population_size exceeds integer census range")
    elif "population_size" in genetics:
        raise ValueError("Deterministic haplotypes have no population_size")
    init = copy.deepcopy(
        initialization
        if initialization is not None
        else (
            {
                "kind": "multinomial_from_frequencies",
                "frequencies": [1 / genome.count] * genome.count,
            }
            if finite
            else {
                "kind": "frequencies",
                "frequencies": [1 / genome.count] * genome.count,
            }
        )
    )
    if finite and set(init) == {"kind", "counts"} and init["kind"] == "explicit_counts":
        validate_counts(init["counts"], genome.count, size)
    elif set(init) == {"kind", "frequencies"} and init["kind"] == (
        "multinomial_from_frequencies" if finite else "frequencies"
    ):
        distribution(init["frequencies"], genome.count)
    else:
        raise ValueError(
            "Initialization must declare frequencies, multinomial_from_frequencies, or explicit_counts for its method"
        )
    resolved = {
        "genome": genome.to_dict(),
        "expression": expression.to_dict(),
        "transmission": transmission.to_dict(),
        "fitness": fitness,
        **({"population_size": size} if finite else {}),
    }
    return resolved, binding.to_dict(), init


def validate_counts(values, count, size):
    raw = np.asarray(values)
    if (
        raw.shape != (count,)
        or raw.dtype.kind not in {"i", "u"}
        or np.any(raw < 0)
        or sum(map(int, raw)) != size
    ):
        raise ValueError(
            "Census counts must be nonnegative integers summing to population_size"
        )
    return raw.astype(np.int64, copy=True)


class MultilocusRealisation:
    id = "population.multilocus"

    def __init__(self, game, organisation, method, parameters, seed=0):
        self.organisation, self.method = organisation, method
        self.parameters = copy.deepcopy(parameters)
        limits = {**RESOURCE_DEFAULTS, **parameters["resources"]}
        self.genome = GenomeSpec.from_dict(
            parameters["genome"], max_haplotypes=limits["max_haplotypes"]
        )
        self.expression = ExpressionMap(self.genome, game, parameters["expression"])
        self.transmission = TransmissionSpec(
            self.genome,
            parameters["transmission"],
            max_recombination_patterns=limits["max_recombination_patterns"],
            max_transmission_work=limits["max_transmission_work"],
        )
        self.finite = method.id == "wright_fisher.haploid"
        self.population_size = parameters.get("population_size")
        self.rng = np.random.Generator(np.random.PCG64(seed))
        init = parameters["initialization"]
        self.counts = None
        if self.finite:
            self.counts = (
                validate_counts(init["counts"], self.genome.count, self.population_size)
                if init["kind"] == "explicit_counts"
                else self.rng.multinomial(self.population_size, init["frequencies"])
            )
            self._frequencies = None
        else:
            self._frequencies = distribution(init["frequencies"], self.genome.count)
        self.initial_counts = None if self.counts is None else self.counts.copy()
        self.initial_frequencies = self.frequencies.copy()
        self.last_generation = None
        game.compile()
        self.compatibility = {}
        for name, slots in self.population.individuals.items():
            mask = np.ones((len(game.chance_mass), self.genome.count), dtype=bool)
            for slot in slots:
                mask &= game.masks[slot][:, self.expression.plan_indices[slot]]
            self.compatibility[name] = mask

    @property
    def game(self):
        return self.organisation._game

    @property
    def population(self):
        return self.organisation.population

    @property
    def frequencies(self):
        return self.counts / self.population_size if self.finite else self._frequencies

    @property
    def lineage(self):
        return getattr(self.game, "construction_state", {})

    def expectations(self, frequencies=None):
        x = distribution(
            self.frequencies if frequencies is None else frequencies, self.genome.count
        )
        game, binding = self.game, self.population
        z = {n: mask @ x for n, mask in self.compatibility.items()}
        mass = game.chance_mass.copy()
        for marginal in z.values():
            mass *= marginal
        conditional_returns = {}
        for name, slots in binding.individuals.items():
            other_mass = game.chance_mass.copy()
            for other, marginal in z.items():
                if other != name:
                    other_mass *= marginal
            for slot in slots:
                column = game.roles.index(slot)
                conditional_returns[slot] = (
                    self.compatibility[name].T @ (other_mass * game.returns[:, column])
                    - game.strategy_costs[slot][self.expression.plan_indices[slot]]
                )
        if game.payoff_matrices is not None:
            first, second = game.roles
            a, b = (self.expression.plan_indices[s] for s in game.roles)
            matrices = game.payoff_matrices
            if binding.slot_individual[first] == binding.slot_individual[second]:
                conditional_returns[first] = matrices[0][a, b]
                conditional_returns[second] = matrices[1][a, b]
            else:
                fa = np.bincount(a, weights=x, minlength=len(game.plans[first]))
                fb = np.bincount(b, weights=x, minlength=len(game.plans[second]))
                conditional_returns[first] = (matrices[0] @ fb)[a]
                conditional_returns[second] = (matrices[1].T @ fa)[b]
        returns = sum(
            binding.return_weights[s] * conditional_returns[s] for s in game.roles
        )
        fitness = self.parameters["fitness"]
        reproductive = fitness["background"] + fitness["scale"] * returns
        if not np.isfinite(reproductive).all() or np.any(reproductive < 0):
            raise ValueError(
                "Transformed reproductive weights must be finite and nonnegative"
            )
        mean = float(x @ reproductive)
        if not math.isfinite(mean) or mean < 0:
            raise ValueError("Mean reproduction must be finite and nonnegative")
        return {
            "success_probability": float(mass @ game.success),
            "expected_returns": {
                s: float(x @ conditional_returns[s]) for s in game.roles
            },
            "expected_terms": float(mass @ game.terms),
            "expected_exchanges": float(mass @ game.exchanges),
            "completed_mass": float(mass.sum()),
            "reproductive_weights": reproductive.tolist(),
            "mean_reproduction_fitness": mean,
        }

    def advance(self):
        """Commit one complete generation; a failed proposal preserves RNG/state."""
        x = distribution(self.frequencies, self.genome.count)
        if self.finite:
            validate_counts(self.counts, self.genome.count, self.population_size)
        result = self.expectations(x)
        if result["mean_reproduction_fitness"] == 0:
            # This census has a valid assay but supplies no parents for the
            # next generation. Keep it available for reporting/intervention.
            return None, result
        y = (
            x
            * np.asarray(result["reproductive_weights"])
            / result["mean_reproduction_fitness"]
        )
        v = self.transmission.recombine(y)
        p = self.transmission.mutate(v)
        rng_state = copy.deepcopy(self.rng.bit_generator.state)
        try:
            counts = (
                self.rng.multinomial(self.population_size, p) if self.finite else None
            )
            next_x = counts / self.population_size if self.finite else p
            distribution(next_x, self.genome.count)
            record = {
                "before": x.tolist(),
                "after_selection": y.tolist(),
                "after_recombination": v.tolist(),
                "expected_offspring": p.tolist(),
                "after_census": next_x.tolist(),
                "assay_before_reproduction": result,
            }
            # Validate next-census measurements before committing the proposal.
            next_result = self.expectations(next_x)
        except Exception:
            self.rng.bit_generator.state = rng_state
            raise
        if self.finite:
            self.counts = counts
        else:
            self._frequencies = next_x
        self.last_generation = record
        return record, next_result

    def inspect_dispositions(self):
        return {
            "implementation": self.id,
            "adaptive_operation": self.method.id,
            "haplotypes": self.genome.haplotypes,
            "expression": self.expression.to_dict(),
            "plan_coordinates": {s: p.tolist() for s, p in self.game.plans.items()},
            "frequencies": self.frequencies.tolist(),
            "counts": None if self.counts is None else self.counts.tolist(),
            "allele_frequencies": self.genome.allele_frequencies(self.frequencies),
            "pairwise_linkage_disequilibrium": self.genome.linkage_disequilibria(
                self.frequencies
            ),
            "transmission": self.transmission.to_dict(),
            "life_cycle": list(LIFE_CYCLE),
            "last_generation": copy.deepcopy(self.last_generation),
        }


class MultilocusSimulation:
    """Discrete generation coordinator with exact assays and resumable censuses."""

    def __init__(self, profile, seed=0, *, game=None, protocol=None):
        from .specification import ModelSpecification
        from .simulation import digest, provenance

        specification = (
            profile
            if isinstance(profile, ModelSpecification)
            else ModelSpecification.from_profile(profile, game=game)
        )
        if isinstance(profile, ModelSpecification) and game is not None:
            specification = specification.with_game(game)
        self.profile = specification.to_profile(protocol)
        self.protocol = specification.execution_parameters(protocol)
        self.method = adaptive_method(self.profile.method)
        if self.method.method_class != "population_generations":
            raise ValueError("MultilocusSimulation requires a generation method")
        game = specification.construct_finite_game()
        organisation = Organisation.for_finite(game)
        organisation.population = MultilocusPopulationSpec(
            game, **specification.organisation_parameters
        )
        self._realisation = MultilocusRealisation(
            game, organisation, self.method, specification.realisation_parameters, seed
        )
        self.model_instance = ModelInstance(
            specification, game, organisation, self._realisation
        )
        self.seed, self.generation = seed, 0
        self.success_sum = 0.0
        self.history = [
            {
                "phase": "initialization",
                "generation": 0,
                "initialization": copy.deepcopy(
                    self._realisation.parameters["initialization"]
                ),
            }
        ]
        self.metadata = provenance()
        self.configuration_hash = digest(self.profile.to_dict())
        self.game_hash = digest(self.game.to_dict())
        self.fidelity = {
            **copy.deepcopy(game.fidelity),
            "adaptive_law": {
                "method": self.method.id,
                "source": self.method.source,
                "life_cycle": list(LIFE_CYCLE),
                "census_size_unit": "haploid_genomes",
            },
            "population": organisation.population.to_dict(),
        }
        self.status = "ready"
        self.measurements = [self.evaluate()]
        if self.measurements[-1]["mean_reproduction_fitness"] == 0:
            self.status = "no_reproduction"

    @property
    def game(self):
        return self.model_instance.game

    @property
    def frequencies(self):
        return self._realisation.frequencies

    @property
    def counts(self):
        return self._realisation.counts

    @property
    def population(self):
        return self.model_instance.organisation.population

    def evaluate(self):
        return {
            "protocol": "exact_genotype_expectations",
            "generation": self.generation,
            **self._realisation.expectations(),
        }

    def step(self):
        if self.status == "no_reproduction":
            return {"status": "no_reproduction", "generation": self.generation}
        retain = (self.generation + 1) % self.protocol["sample_interval"] == 0
        if retain and len(self.measurements) >= self._realisation.parameters[
            "resources"
        ].get("max_measurements", RESOURCE_DEFAULTS["max_measurements"]):
            raise ValueError("Generation measurements exceed max_measurements")
        record, result = self._realisation.advance()
        if record is None:
            self.status = "no_reproduction"
            return {"status": "no_reproduction", "generation": self.generation}
        self.generation += 1
        self.success_sum += record["assay_before_reproduction"]["success_probability"]
        self.status = (
            "no_reproduction" if result["mean_reproduction_fitness"] == 0 else "running"
        )
        if retain:
            self.measurements.append(
                {
                    "protocol": "exact_genotype_expectations",
                    "generation": self.generation,
                    **result,
                }
            )
        return copy.deepcopy(record)

    def run(self, generations=None, *, output=None):
        generations = (
            self.protocol["generations"] if generations is None else generations
        )
        if (
            not isinstance(generations, int)
            or isinstance(generations, bool)
            or generations < 0
        ):
            raise ValueError("generations must be a nonnegative integer")
        if self.status == "no_reproduction":
            generations = 0
        if generations and len(self.history) >= self._realisation.parameters[
            "resources"
        ].get("max_history_entries", RESOURCE_DEFAULTS["max_history_entries"]):
            raise ValueError("Phase history exceeds max_history_entries")
        limit = self._realisation.parameters["resources"].get(
            "max_measurements", RESOURCE_DEFAULTS["max_measurements"]
        )
        target = self.generation + generations
        needed = (
            target // self.protocol["sample_interval"]
            - self.generation // self.protocol["sample_interval"]
        )
        if len(self.measurements) + needed > limit:
            raise ValueError("Generation measurements exceed max_measurements")
        start = self.generation
        for _ in range(generations):
            self.step()
            if self.status == "no_reproduction":
                break
        if self.generation > start:
            self.history.append(
                {
                    "phase": "run",
                    "from_generation": start,
                    "to_generation": self.generation,
                }
            )
            if self.status != "no_reproduction":
                self.status = "complete"
        if output is not None:
            from pathlib import Path
            from .simulation import atomic_json

            root = Path(output)
            self.checkpoint(root)
            atomic_json(root / "report.json", self.summary())
        return self.summary()

    def summary(self):
        result = self.evaluate()
        return {
            **result,
            "clock": "generations",
            "status": self.status,
            "final_policy_success_probability": result["success_probability"],
            "generation_average_expected_success": self.success_sum / self.generation
            if self.generation
            else None,
            "average_assay_timing": "before reproduction in every advanced generation",
        }

    def set_population(self, values):
        if len(self.history) >= self._realisation.parameters["resources"].get(
            "max_history_entries", RESOURCE_DEFAULTS["max_history_entries"]
        ):
            raise ValueError("Phase history exceeds max_history_entries")
        if self._realisation.finite:
            counts = validate_counts(
                values,
                self._realisation.genome.count,
                self._realisation.population_size,
            )
            result = self._realisation.expectations(
                counts / self._realisation.population_size
            )
            self._realisation.counts = counts
        else:
            frequencies = distribution(values, self._realisation.genome.count)
            result = self._realisation.expectations(frequencies)
            self._realisation._frequencies = frequencies
        if result["mean_reproduction_fitness"] == 0:
            self.status = "no_reproduction"
        elif self.status == "no_reproduction":
            self.status = "ready"
        self.history.append(
            {
                "phase": "population_intervention",
                "generation": self.generation,
                "frequencies": self.frequencies.tolist(),
            }
        )

    def snapshot(self):
        from .simulation import digest

        r = self._realisation
        return copy.deepcopy(
            {
                "schema_version": 1,
                "engine": "adaptive",
                "method_class": "population_generations",
                "profile": self.profile.to_dict(),
                "configuration_hash": self.configuration_hash,
                "model_specification": self.model_instance.specification.to_dict(),
                "model_specification_hash": digest(
                    self.model_instance.specification.to_dict()
                ),
                "model_protocol": copy.deepcopy(self.protocol),
                "protocol": self.protocol,
                "game": self.game.to_dict(),
                "game_hash": self.game_hash,
                "construction_state": getattr(self.game, "construction_state", {}),
                "seed": self.seed,
                "metadata": self.metadata,
                "fidelity": self.fidelity,
                "population": self.population.to_dict(),
                "generation": self.generation,
                "frequencies": self.frequencies.tolist(),
                "counts": None if self.counts is None else self.counts.tolist(),
                "initial_frequencies": r.initial_frequencies.tolist(),
                "initial_counts": None
                if r.initial_counts is None
                else r.initial_counts.tolist(),
                "rng_state": r.rng.bit_generator.state,
                "last_generation": r.last_generation,
                "success_sum": self.success_sum,
                "history": self.history,
                "measurements": self.measurements,
                "status": self.status,
            }
        )

    def restore_state(self, snapshot):
        r = self._realisation
        if snapshot["population"] != self.population.to_dict():
            raise ValueError("Checkpoint individual bindings differ")
        if (
            not isinstance(snapshot["generation"], int)
            or isinstance(snapshot["generation"], bool)
            or snapshot["generation"] < 0
        ):
            raise ValueError("Invalid checkpoint generation")
        if r.finite:
            counts = validate_counts(
                snapshot["counts"], r.genome.count, r.population_size
            )
            initial = validate_counts(
                snapshot["initial_counts"], r.genome.count, r.population_size
            )
            if not np.array_equal(
                counts / r.population_size, snapshot["frequencies"]
            ) or not np.array_equal(
                initial / r.population_size, snapshot["initial_frequencies"]
            ):
                raise ValueError("Checkpoint frequencies differ from census counts")
            r.counts, r.initial_counts = counts, initial
        else:
            if snapshot["counts"] is not None or snapshot["initial_counts"] is not None:
                raise ValueError("Deterministic checkpoint contains census counts")
            r._frequencies = distribution(snapshot["frequencies"], r.genome.count)
        r.initial_frequencies = distribution(
            snapshot["initial_frequencies"], r.genome.count
        )
        r.expectations()
        total = snapshot["success_sum"]
        if (
            not isinstance(total, (int, float))
            or not math.isfinite(total)
            or not 0 <= total <= snapshot["generation"] + 1e-12
        ):
            raise ValueError("Invalid cumulative generation success")
        r.rng.bit_generator.state = copy.deepcopy(snapshot["rng_state"])
        r.last_generation = copy.deepcopy(snapshot["last_generation"])
        self.generation, self.success_sum = snapshot["generation"], total
        self.measurements = copy.deepcopy(snapshot["measurements"])

    def checkpoint(self, path):
        from .adaptive_runtime import checkpoint

        return checkpoint(self.snapshot(), path)

    def inspect(self):
        return {
            **self.snapshot(),
            "model_instance": self.model_instance.inspect(),
            "assessment": self.summary(),
        }
