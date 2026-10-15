"""Built-in scientific declarations and their numerical execution binding.

Resolution reads declarations and validates choices. Seeded construction and
exact finite compilation occur when a System creates its backend.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping

from .adaptive import adaptive_method
from .definitions import game_definition
from .finite_profiles import AdaptiveProfile, finite_game
from .registry import Profile, load_profile

_FINITE_DEFAULTS = {
    "max_role_strategies": 10_000,
    "max_leaves": 100_000,
    "max_mask_entries": 10_000_000,
    "max_measurements": 100_000,
}


def _merge(base, changes):
    result = copy.deepcopy(base)
    for key, value in changes.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _profile(model, protocol=None):
    """Reconstruct the numerical kernel's internal configuration."""
    interaction, realisation, adaptation = (
        model[k] for k in ("interaction", "realisation", "adaptation")
    )
    rp, ap = (
        copy.deepcopy(realisation["parameters"]),
        copy.deepcopy(adaptation["parameters"]),
    )
    controls = copy.deepcopy(model["execution"]["parameters"])
    budget_key = {
        "encounters": "plays",
        "generations": "generations",
        "time": "duration",
    }[model["execution"]["clock"]]
    controls[budget_key] = model["execution"]["budget"]
    controls.update(copy.deepcopy(dict(protocol or {})))
    budget = model["execution"]["budget"]
    if (
        isinstance(budget, bool)
        or not isinstance(budget, (int, float))
        or not math.isfinite(budget)
        or budget < 0
    ):
        raise ValueError("Execution budget must be finite and nonnegative")
    if model["execution"]["clock"] != "time" and not isinstance(budget, int):
        raise ValueError("Discrete execution budgets require integers")
    # Numerical compatibility configurations keep their positive default run
    # count; System owns zero-budget experiments and explicit advance amounts.
    if budget == 0:
        controls[budget_key] = 1
    source = model["source"]
    gp = copy.deepcopy(interaction["parameters"])
    if realisation["provider"] == "barrett.conditional":
        if (
            adaptation["provider"] != "barrett.reinforcement"
            or interaction["provider"] != "barrett"
        ):
            raise ValueError(
                "Barrett conditional realisation requires its Barrett interaction and reinforcement law"
            )
        expected_initial = {
            "initial_weight": 1.0,
            "initial_reference": 0.0,
            "initial_assignment": "uniform_option"
            if ap.get("rule") in {"R6", "WSLS", "WSLR", "STRATEGY_WSLR"}
            else "per_decision_sampling",
        }
        if rp and rp != expected_initial:
            raise ValueError(
                "Barrett realisation uses declared initial table weight/reference and rule-dependent assignment"
            )
        if controls.keys() - {"plays", "window"}:
            raise ValueError("Barrett execution accepts plays and window")
        if model["resources"]:
            raise ValueError(
                "Unknown Barrett resource budgets: "
                + ", ".join(sorted(model["resources"]))
            )
        unknown = model["organisation"].keys() - {"owners", "functional_roles"}
        if unknown:
            raise ValueError(
                "Unknown Barrett organisation fields: " + ", ".join(sorted(unknown))
            )
        gp["owners"] = copy.deepcopy(model["organisation"]["owners"])
        expected_org = game_definition(interaction["game"]).initial_organisation(gp)
        if set(gp["owners"]) - set(expected_org["owners"]):
            raise ValueError("Organisation declares an unknown module owner")
        if (
            model["organisation"].get(
                "functional_roles", expected_org["functional_roles"]
            )
            != expected_org["functional_roles"]
        ):
            raise ValueError("Functional roles must agree with the source game")
        gp["window"] = controls["window"]
        return Profile(
            model["id"],
            interaction["game"],
            gp,
            ap,
            kind=source["kind"],
            plays=controls["plays"],
            source=list(source["identifiers"]),
            assumptions=list(source["assumptions"]),
        )
    if interaction["provider"] != "finite":
        raise ValueError(
            "Built-in finite realisation requires the finite interaction provider"
        )
    method = adaptive_method(adaptation["provider"])
    expected_realisation = {
        "population_generations": "population.multilocus",
        "population_evolution": "population.complete_plan",
    }.get(
        method.method_class,
        "finite.complete_plan"
        if method.representation == "complete_role_plans"
        else "finite.conditional",
    )
    if realisation["provider"] != expected_realisation:
        raise ValueError("Realisation representation and adaptation law disagree")
    allowed_adaptation = (
        {"transmission", "fitness"}
        if method.method_class == "population_generations"
        else {"mutation_rate"}
        if method.method_class == "population_evolution"
        else set()
    )
    if ap.keys() - allowed_adaptation:
        raise ValueError(
            "Unknown adaptation parameters: "
            + ", ".join(sorted(ap.keys() - allowed_adaptation))
        )
    fixed = rp.pop("fixed_decisions", {})
    lineage = rp.pop("lineage", {})
    genetics = {}
    if method.method_class == "population_generations":
        genetics = {
            k: rp.pop(k)
            for k in list(rp)
            if k in {"genome", "expression", "population_size"}
        }
        genetics.update({k: ap[k] for k in ("transmission", "fitness")})
    elif method.method_class == "population_evolution":
        controls["mutation_rate"] = ap["mutation_rate"]
    controls.update(rp)
    return AdaptiveProfile(
        model["id"],
        interaction["game"],
        gp,
        method=method.id,
        population=copy.deepcopy(model["organisation"]),
        protocol=controls,
        fixed_decisions=fixed,
        resources=copy.deepcopy(model["resources"]),
        kind=source["kind"],
        assumptions=list(source["assumptions"]),
        genetics=genetics,
        lineage=lineage,
    )


def resolve_model(identifier: str | dict, overrides: dict | None = None) -> dict:
    """Resolve one preset or an explicit built-in declaration without sampling."""
    declared = None
    if isinstance(identifier, Mapping) and "interaction" in identifier:
        declared = _merge(identifier, overrides or {})
        allowed = {
            "schema_version",
            "kind",
            "id",
            "interaction",
            "organisation",
            "realisation",
            "adaptation",
            "source",
            "execution",
            "resources",
            "capabilities",
            "runtime",
        }
        if declared.keys() - allowed:
            raise ValueError(
                "Unknown built-in model fields: "
                + ", ".join(sorted(declared.keys() - allowed))
            )
        if declared.get("runtime", "builtin") != "builtin":
            raise ValueError("Built-in declarations require runtime='builtin'")
        for layer, allowed_fields in {
            "interaction": {"provider", "game", "parameters"},
            "realisation": {"provider", "parameters"},
            "adaptation": {"provider", "parameters"},
            "source": {"kind", "identifiers", "assumptions", "questions"},
            "execution": {"clock", "budget", "parameters"},
        }.items():
            unknown = declared[layer].keys() - allowed_fields
            if unknown:
                raise ValueError(
                    "Unknown " + layer + " fields: " + ", ".join(sorted(unknown))
                )
        profile = _profile(declared)
        profile.validate()
    else:
        profile = load_profile(identifier)
    if isinstance(profile, Profile):
        definition = game_definition(profile.model)
        from .learners import resolved_parameters
        from .registry import catalogue

        gp = definition.resolve_parameters(
            {
                k: v
                for k, v in profile.parameters.items()
                if k not in {"window", "owners"}
            }
        )
        org = definition.initial_organisation(profile.parameters)
        entry = next(e for e in catalogue() if e["id"] == profile.model)
        questions, identifiers = entry["source_questions"], profile.source
        ap = resolved_parameters(profile.learner)
        rp = {
            "initial_weight": 1.0,
            "initial_reference": 0.0,
            "initial_assignment": "uniform_option"
            if ap["rule"] in {"R6", "WSLS", "WSLR", "STRATEGY_WSLR"}
            else "per_decision_sampling",
        }
        rid, aid, provider = "barrett.conditional", "barrett.reinforcement", "barrett"
        clock, budget = "encounters", profile.plays
        controls = {"window": profile.parameters.get("window", 1000)}
        resources = {}
        capabilities = ["advance", "snapshot", "restore", "inspect", "evaluate.sampled"]
        if profile.model not in {
            "INQ",
            "VIS",
            "STAG",
            "FFSR",
            "META-RULE",
            "INV",
            "ORD",
            "ADD",
        }:
            capabilities.append("evaluate.exact")
    else:
        method = adaptive_method(profile.method)
        rid = {
            "population_generations": "population.multilocus",
            "population_evolution": "population.complete_plan",
        }.get(
            method.method_class,
            "finite.complete_plan"
            if method.representation == "complete_role_plans"
            else "finite.conditional",
        )
        aid, provider = method.id, "finite"
        from .finite_profiles import finite_catalogue

        entry = next(e for e in finite_catalogue() if e["id"] == profile.model)
        identifiers = list(
            dict.fromkeys([entry.get("source", method.source), method.source])
        )
        org = copy.deepcopy(profile.population)
        gp = copy.deepcopy(profile.parameters)
        game = None
        if profile.model.startswith(("SK-", "ML-")):
            game = finite_game(profile)
            gp = copy.deepcopy(game.parameters)
            # These fields are derived by the named cost case. Keeping its
            # defining inputs preserves source fidelity when reconstructing it.
            if gp.get("cost_case"):
                for k in (
                    "state_probabilities",
                    "background",
                    "signal_costs",
                    "listening_cost",
                ):
                    gp.pop(k, None)
                gp.setdefault("c", 0.1)
            questions = game.fidelity.get("game", {}).get("questions", [])
            if method.representation == "conditional_decisions":
                from .realisations import validate_finite_realisation

                validate_finite_realisation(game, method, sampled=True)
            if method.nonnegative_returns and profile.model == "SK-SIG":
                import numpy as np

                parameters = game.parameters
                rewards = np.asarray(
                    parameters.get(
                        "payoffs", np.eye(parameters["states"], parameters["actions"])
                    ),
                    dtype=float,
                )
                costs = np.asarray(parameters.get("signal_costs", 0), dtype=float)
                cost_by_state = (
                    costs.max(axis=1)
                    if costs.ndim == 2
                    else np.full(parameters["states"], float(costs))
                )
                lower_bound = float(
                    np.min(
                        float(parameters.get("background", 0))
                        + rewards
                        - cost_by_state[:, None]
                    )
                )
                if lower_bound < 0 and not profile.fixed_decisions:
                    raise ValueError(
                        "R1 and birth mutation require nonnegative numerical returns; declare background fitness"
                    )
        else:
            # Barrett adapters initialise source templates only at build time.
            if set(gp) != {"profile"}:
                raise ValueError(
                    "Barrett finite adapters require parameters={'profile': source_profile}"
                )
            source_profile = load_profile(gp["profile"])
            if source_profile.model != profile.model:
                raise ValueError("Barrett source profile and model disagree")
            questions = entry.get("source_questions", [])
        resources = {**_FINITE_DEFAULTS, **profile.resources}
        rp = {"fixed_decisions": copy.deepcopy(profile.fixed_decisions)}
        if profile.lineage:
            rp["lineage"] = copy.deepcopy(profile.lineage)
        ap = {}
        if method.method_class == "population_generations":
            from .multilocus import RESOURCE_DEFAULTS, resolve_genetics

            resources = {
                k: v
                for k, v in resources.items()
                if k not in {"max_population_strategies", "max_field_evaluations"}
            }
            resources = {**resources, **RESOURCE_DEFAULTS, **profile.resources}
            genetics, org, initialization = resolve_genetics(
                game,
                method,
                profile.genetics,
                org,
                profile.protocol.get("initialization"),
                resources,
                declaration_only=True,
            )
            rp.update(
                {
                    k: v
                    for k, v in genetics.items()
                    if k not in {"transmission", "fitness"}
                }
            )
            rp["initialization"] = initialization
            ap = {k: genetics[k] for k in ("transmission", "fitness")}
            clock, budget = "generations", profile.protocol.get("generations", 100)
            controls = {"sample_interval": profile.protocol.get("sample_interval", 1)}
        elif method.method_class == "population_evolution":
            resources = {
                "max_population_strategies": 100_000,
                "max_field_evaluations": 200_000,
                **resources,
                "max_measurements": profile.resources.get("max_measurements", 10_000),
            }
            if game is not None:
                from .organisation import PopulationSpec

                population = PopulationSpec(game, **org, enumerate_components=False)
                org = {
                    k: v
                    for k, v in population.to_dict().items()
                    if k not in {"strategy_counts", "inheritance"}
                }
            rp["initialization"] = copy.deepcopy(
                profile.protocol.get(
                    "initialization", {"kind": "dirichlet", "concentration": 1.0}
                )
            )
            ap["mutation_rate"] = profile.protocol.get("mutation_rate", 0.0)
            clock, budget = "time", profile.protocol.get("duration", 50.0)
            controls = {
                "solver": "DOP853",
                "rtol": 1e-9,
                "atol": 1e-12,
                "max_step": 1.0,
                "sample_interval": 1.0,
            }
            controls.update(
                {k: v for k, v in profile.protocol.items() if k in controls}
            )
        else:
            rp["initial_weight"] = profile.protocol.get("initial_weight", 1.0)
            clock, budget = "encounters", profile.protocol.get("plays", 10_000)
            controls = {"window": profile.protocol.get("window", 1000)}
        capabilities = ["advance", "snapshot", "restore", "inspect", "evaluate.exact"]
        if method.method_class == "reinforcement":
            capabilities.append("evaluate.sampled")
        if game is not None and any(
            count > resources["max_role_strategies"]
            for count in game.plan_counts.values()
        ):
            capabilities.remove("evaluate.exact")
    result = {
        "id": profile.id,
        "interaction": {"provider": provider, "game": profile.model, "parameters": gp},
        "organisation": org,
        "realisation": {"provider": rid, "parameters": rp},
        "adaptation": {"provider": aid, "parameters": ap},
        "source": {
            "kind": profile.kind,
            "identifiers": identifiers,
            "assumptions": list(profile.assumptions),
            "questions": list(questions),
        },
        "execution": {"clock": clock, "budget": budget, "parameters": controls},
        "resources": resources,
        "capabilities": capabilities,
        "runtime": "builtin",
    }
    if declared is not None:
        if declared["realisation"]["provider"] != result["realisation"]["provider"]:
            raise ValueError("Realisation representation and adaptation law disagree")
        result["execution"]["budget"] = declared["execution"]["budget"]
        declared_source = declared.get("source", {})
        required_questions = result["source"]["questions"]
        result["source"] = _merge(result["source"], declared_source)
        result["source"]["questions"] = list(
            dict.fromkeys(required_questions + result["source"]["questions"])
        )
    elif overrides:
        return resolve_model(_merge(result, overrides))
    # Reconstruct to apply validators to the complete defaulted declaration.
    _profile(result).validate()
    if (
        result["source"]["kind"] in {"book_literal", "source_resolved", "paper_game"}
        and result["source"]["questions"]
    ):
        raise ValueError(
            "Source declaration needs resolution: "
            + ", ".join(result["source"]["questions"])
        )
    return copy.deepcopy(result)


class BuiltinBackend:
    """The canonical lifecycle binding for the existing scientific kernels."""

    def __init__(self, model, simulation):
        self.model = copy.deepcopy(model)
        self.simulation = simulation

    @property
    def clock(self):
        return self.model["execution"]["clock"]

    @property
    def position(self):
        if self.clock == "time":
            return self.simulation.time
        if self.clock == "generations":
            return self.simulation.generation
        if hasattr(self.simulation, "plays"):
            return self.simulation.plays
        return self.simulation.runtime.clocks["play"]

    @property
    def status(self):
        return self.simulation.status

    def advance(self, amount):
        if (
            isinstance(amount, bool)
            or not isinstance(amount, (int, float))
            or not math.isfinite(amount)
            or amount < 0
        ):
            raise ValueError("Advance amount must be finite and nonnegative")
        if self.clock != "time" and not isinstance(amount, int):
            raise ValueError("Discrete clocks require an integer advance amount")
        if amount == 0:
            return None
        key = {"encounters": "plays", "time": "duration", "generations": "generations"}[
            self.clock
        ]
        return self.simulation.run(**{key: amount})

    def snapshot(self):
        return self.simulation.snapshot()

    def evaluate(self, **parameters):
        from .simulation import Simulation

        if isinstance(self.simulation, Simulation):
            from .metrics import evaluate

            return evaluate(self.simulation, **parameters)
        if self.model["realisation"]["provider"] == "finite.conditional":
            parameters.setdefault(
                "protocol",
                "exact"
                if "evaluate.exact" in self.model["capabilities"]
                else "sampled",
            )
        if self.clock in {"time", "generations"}:
            requested = parameters.pop("protocol", "exact")
            if requested != "exact":
                raise ValueError(
                    "Population expectation evaluation requires the exact capability"
                )
        return self.simulation.evaluate(**parameters)

    def inspect(self):
        result = self.simulation.inspect()
        result["mechanisms"] = {
            "interaction": copy.deepcopy(self.model["interaction"]),
            "representation": copy.deepcopy(self.model["realisation"]),
            "adaptation": copy.deepcopy(self.model["adaptation"]),
            "organisation": copy.deepcopy(self.model["organisation"]),
            "checkpoint_granularity": {
                "encounters": "completed_encounter",
                "generations": "completed_generation",
                "time": "completed_solver_segment",
            }[self.clock],
            "frozen_components": copy.deepcopy(
                self.model["realisation"]["parameters"].get("fixed_decisions", {})
            ),
        }
        if self.model["interaction"]["game"] == "META-RULE":
            p = self.model["interaction"]["parameters"]
            result["mechanisms"]["inner_task"] = {
                "initialization": "fresh learner and game state for each outer encounter",
                "seed": "fresh inner seed drawn from the outer runtime RNG",
                "game": p["inner_model"],
                "parameters": copy.deepcopy(p["inner_parameters"]),
                "learning_rules": copy.deepcopy(p["rules"]),
                "encounters_per_task": p["inner_plays"],
                "retained_result": "last completed task seed, selected rule, and cumulative accuracy",
                "checkpoint_granularity": "completed_outer_encounter",
            }
        return result


def create_backend(model_dict: dict, seed: int, protocol: dict | None = None):
    from .simulation import build

    profile = _profile(model_dict, protocol)
    if model_dict["realisation"]["provider"] == "finite.conditional":
        from .finite_learning import ReinforcementSimulation

        simulation = ReinforcementSimulation(profile, seed, sampled=True)
    else:
        simulation = build(profile, seed)
    return BuiltinBackend(model_dict, simulation)


def restore_backend(model_dict: dict, snapshot: dict):
    saved_model = resolve_model(snapshot["profile"])
    for field in (
        "interaction",
        "organisation",
        "realisation",
        "adaptation",
        "resources",
    ):
        if saved_model[field] != model_dict[field]:
            raise ValueError("Checkpoint scientific model differs: " + field)
    if snapshot.get("engine") == "adaptive":
        from .adaptive_runtime import restore_adaptive

        simulation = restore_adaptive(snapshot)
    else:
        from .simulation import Simulation

        simulation = Simulation.restore(snapshot)
    return BuiltinBackend(model_dict, simulation)
