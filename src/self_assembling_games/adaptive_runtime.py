"""Shared build, restore, evaluation and checkpoint lifecycle for finite methods."""

from __future__ import annotations

import copy
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from .adaptive import adaptive_method
from .finite_profiles import AdaptiveProfile


def build_adaptive(
    profile: str | dict[str, Any] | AdaptiveProfile,
    seed: int = 0,
    *,
    game=None,
    protocol=None,
):
    from .specification import ModelSpecification

    specification = (
        profile
        if isinstance(profile, ModelSpecification)
        else ModelSpecification.from_profile(profile, game=game)
    )
    if isinstance(profile, ModelSpecification) and game is not None:
        specification = specification.with_game(game)
    resolved = specification.to_profile(protocol)
    method = adaptive_method(resolved.method)
    if method.method_class == "population_generations":
        from .multilocus import MultilocusSimulation

        return MultilocusSimulation(specification, seed, protocol=protocol)
    if method.method_class == "population_evolution":
        from .evolution import EvolutionSimulation

        return EvolutionSimulation(specification, seed, protocol=protocol)
    from .finite_learning import ReinforcementSimulation

    return ReinforcementSimulation(specification, seed, protocol=protocol)


def checkpoint(snapshot: dict[str, Any], path: str | Path) -> Path:
    from .simulation import atomic_json

    path = Path(path)
    if path.suffix != ".json":
        path = path / "checkpoint.json"
    atomic_json(path, snapshot)
    return path


def restore_adaptive(snapshot: dict[str, Any], *, game=None):
    from .simulation import digest, provenance

    if snapshot.get("schema_version") != 1 or snapshot.get("engine") != "adaptive":
        raise ValueError("Unsupported adaptive checkpoint schema")
    if digest(snapshot["profile"]) != snapshot["configuration_hash"]:
        raise ValueError("Adaptive checkpoint configuration hash mismatch")
    current = provenance()
    for field in ["package_version", "numpy", "source_hash"]:
        if snapshot["metadata"].get(field) != current[field]:
            raise ValueError(
                f"Checkpoint {field} differs; use the pinned implementation"
            )
    if "scipy" in snapshot["metadata"]:
        from importlib.metadata import version

        if version("scipy") != snapshot["metadata"]["scipy"]:
            raise ValueError(
                "Checkpoint scipy differs; use the pinned numerical implementation"
            )
    if game is None and snapshot.get("construction_state", {}).get("barrett_source"):
        from .finite_adapters import barrett_game
        from .finite_profiles import load_adaptive_profile, validate_finite_game

        profile = load_adaptive_profile(snapshot["profile"])
        budgets = {
            k: v
            for k, v in profile.resources.items()
            if k
            not in {
                "max_population_strategies",
                "max_measurements",
                "max_field_evaluations",
                "max_haplotypes",
                "max_recombination_patterns",
                "max_transmission_work",
                "max_genotype_mask_entries",
                "max_history_entries",
                "max_generation_work",
            }
        }
        source_profile = snapshot["game"]["parameters"].get("barrett_profile")
        if source_profile is None:
            raise ValueError("Embedded Barrett source has no game binding")
        game = barrett_game(
            source_profile,
            source_snapshot=snapshot["construction_state"]["barrett_source"],
            **budgets,
        )
        if profile.fixed_decisions:
            game.fixed_decisions.update(copy.deepcopy(profile.fixed_decisions))
            game.__post_init__()
        validate_finite_game(profile, game)
    if snapshot.get("sampled_construction"):
        from .finite_learning import ReinforcementSimulation

        simulation = ReinforcementSimulation(
            snapshot["profile"], snapshot["seed"], game=game, sampled=True
        )
    else:
        simulation = build_adaptive(snapshot["profile"], snapshot["seed"], game=game)
    declaration = snapshot.get("model_specification")
    if declaration is not None and digest(declaration) != snapshot.get(
        "model_specification_hash"
    ):
        raise ValueError("Checkpoint model specification hash mismatch")
    simulation.model_instance.specification.restore_declaration(declaration)
    if (
        simulation.game_hash != snapshot["game_hash"]
        or digest(snapshot["game"]) != snapshot["game_hash"]
    ):
        raise ValueError("Checkpoint finite-game specification differs")
    if snapshot["method_class"] != simulation.method.method_class:
        raise ValueError("Checkpoint method class mismatch")
    if simulation.protocol != snapshot["protocol"]:
        raise ValueError("Checkpoint protocol differs")
    for field in ["metadata", "fidelity", "history", "status"]:
        setattr(simulation, field, copy.deepcopy(snapshot[field]))
    if simulation.method.method_class == "population_generations":
        simulation.restore_state(snapshot)
        return simulation
    if simulation.method.method_class == "population_evolution":
        from .finite import distribution

        spec = copy.deepcopy(snapshot["population"])
        spec.pop("strategy_counts")
        spec.pop("inheritance")
        from .evolution import PopulationSpec

        population = PopulationSpec(
            simulation.game,
            **spec,
            max_population_strategies=simulation.profile.resources.get(
                "max_population_strategies", 100_000
            ),
        )
        expected = simulation.population.to_dict()
        expected["frozen"] = spec["frozen"]
        if population.to_dict() != expected:
            raise ValueError("Checkpoint population layout differs")
        simulation.population = population
        simulation.time = snapshot["time"]
        simulation.frequencies = {
            n: distribution(snapshot["frequencies"][n], size)
            for n, size in population.counts.items()
        }
        simulation.initial_frequencies = {
            n: distribution(snapshot["initial_frequencies"][n], size)
            for n, size in population.counts.items()
        }
        simulation.integrals = np.asarray(snapshot["integrals"], dtype=float)
        if (
            simulation.integrals.shape != (len(simulation.game.roles) + 3,)
            or not np.isfinite(simulation.integrals).all()
        ):
            raise ValueError("Invalid checkpoint integrals")
        simulation.measurements = copy.deepcopy(snapshot["measurements"])
    else:
        for name in ["weights", "initial_weights"]:
            vectors = getattr(simulation, name)
            if vectors.keys() != snapshot[name].keys():
                raise ValueError("Checkpoint policy addresses differ")
            for address, weights in vectors.items():
                values = np.asarray(snapshot[name][address], dtype=float)
                if (
                    values.shape != weights.shape
                    or not np.isfinite(values).all()
                    or np.any(values < 0)
                    or values.sum() <= 0
                ):
                    raise ValueError("Invalid checkpoint reinforcement weights")
                vectors[address] = values.copy()
        simulation.rng.bit_generator.state = copy.deepcopy(snapshot["rng_state"])
        simulation.frozen = set(snapshot["frozen"])
        if not simulation.frozen <= simulation.weights.keys():
            raise ValueError("Unknown frozen checkpoint policy")
        for name in ["plays", "successes", "return_totals", "terms", "exchanges"]:
            setattr(simulation, name, copy.deepcopy(snapshot[name]))
        simulation.recent = deque(
            snapshot["recent"], maxlen=simulation.protocol["window"]
        )
    return simulation


def is_adaptive_profile(profile: Any) -> bool:
    from .finite_profiles import adaptive_profiles

    if isinstance(profile, AdaptiveProfile):
        return True
    if isinstance(profile, dict):
        return "method" in profile
    if isinstance(profile, (str, Path)):
        import json

        if Path(profile).is_file():
            return "method" in json.loads(Path(profile).read_text())
        return any(p["id"] == str(profile) for p in adaptive_profiles())
    return False
