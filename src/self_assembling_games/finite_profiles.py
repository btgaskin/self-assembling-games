"""Generic adaptive profiles, finite game factories, and author-family listings."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Callable

from .adaptive import adaptive_method
from .finite import FiniteGame


@dataclass
class AdaptiveProfile:
    id: str
    model: str
    parameters: dict[str, Any] = field(default_factory=dict)
    method: str = "replicator.continuous"
    population: dict[str, Any] = field(default_factory=dict)
    protocol: dict[str, Any] = field(default_factory=dict)
    fixed_decisions: dict[str, list[float]] = field(default_factory=dict)
    resources: dict[str, int] = field(default_factory=dict)
    kind: str = "demonstration"
    assumptions: list[str] = field(default_factory=list)
    lineage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(vars(self))

    def validate(self) -> None:
        method = adaptive_method(self.method)
        if self.kind == "historical_reproduction":
            raise ValueError(
                "Historical adaptive reproduction protocols remain gated; SKQ1"
            )
        if self.kind not in {"demonstration", "deliberate_variant", "paper_game"}:
            raise ValueError("Unknown adaptive profile kind")
        if method.method_class == "reinforcement" and self.population:
            raise ValueError(
                "Reinforcement profiles use role participants, not population inheritance"
            )
        allowed = {
            "duration",
            "plays",
            "initialization",
            "initial_weight",
            "solver",
            "rtol",
            "atol",
            "max_step",
            "sample_interval",
            "mutation_rate",
            "window",
        }
        if self.protocol.keys() - allowed:
            raise ValueError(
                "Unknown protocol fields: "
                + ", ".join(sorted(self.protocol.keys() - allowed))
            )
        class_allowed = (
            {
                "duration",
                "initialization",
                "solver",
                "rtol",
                "atol",
                "max_step",
                "sample_interval",
                "mutation_rate",
            }
            if method.method_class == "population_evolution"
            else {"plays", "initial_weight", "window"}
        )
        if self.protocol.keys() - class_allowed:
            raise ValueError(f"Protocol fields do not belong to {method.method_class}")
        for name in {
            "duration",
            "initial_weight",
            "rtol",
            "atol",
            "max_step",
            "sample_interval",
        }:
            if name in self.protocol and (
                not isinstance(self.protocol[name], (int, float))
                or not math.isfinite(self.protocol[name])
                or self.protocol[name] <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        for name in {"plays", "window"}:
            if name in self.protocol and (
                not isinstance(self.protocol[name], int)
                or isinstance(self.protocol[name], bool)
                or self.protocol[name] <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        mutation = self.protocol.get("mutation_rate", 0)
        if (
            not isinstance(mutation, (int, float))
            or not math.isfinite(mutation)
            or not 0 <= mutation <= 1
        ):
            raise ValueError("mutation_rate must be in [0,1]")
        if self.method != "replicator_mutator.birth_uniform" and mutation:
            raise ValueError("mutation_rate belongs to the birth-mutation method")
        allowed_budgets = {
            "max_role_strategies",
            "max_population_strategies",
            "max_leaves",
            "max_mask_entries",
            "max_measurements",
            "max_field_evaluations",
        }
        if self.resources.keys() - allowed_budgets:
            raise ValueError("Unknown resource budgets")
        if any(
            not isinstance(v, int) or isinstance(v, bool) or v <= 0
            for v in self.resources.values()
        ):
            raise ValueError("Resource budgets must be positive integers")


_CUSTOM_FACTORIES: dict[str, tuple[Callable[..., FiniteGame], dict[str, Any]]] = {}


def register_finite_game(
    identifier: str,
    factory: Callable[..., FiniteGame],
    *,
    family: str,
    description: str,
    source: str,
) -> None:
    """Register an additional declared game without changing adaptive backends.

    Factories accept parameters, fixed_decisions, and finite resource budgets.
    They must be registered again before restoring their checkpoints.
    """
    from .registry import catalogue

    if (
        identifier.startswith("SK-")
        or identifier in {g["id"] for g in catalogue()}
        or identifier in _CUSTOM_FACTORIES
    ):
        raise ValueError(f"Game already registered/reserved: {identifier}")
    if not identifier or not callable(factory) or not family or not source:
        raise ValueError(
            "A game registration needs an identifier, factory, family and source"
        )
    _CUSTOM_FACTORIES[identifier] = (
        factory,
        {"id": identifier, "family": family, "name": description, "source": source},
    )


def finite_catalogue(family: str | None = None) -> list[dict[str, Any]]:
    from .finite_adapters import _LEGACY_FINITE
    from .registry import catalogue
    from .skyrms import SKYRMS_SOURCE

    descriptions = {
        "SK-SIG": "Dyadic signalling and labelled costs",
        "SK-COMP": "Complementary partial observations",
        "SK-VOTE": "Independent observation errors and signal-count aggregation",
        "SK-TEAM": "Addressed signals and coordinated receiver actions",
        "SK-CHAIN": "Relay and frozen-endpoint translation",
        "SK-DIALOGUE": "Queries select the relevant observation partition",
    }
    entries = [
        {
            "id": identifier,
            "family": "skyrms",
            "name": name,
            "source": SKYRMS_SOURCE,
            "source_questions": ["SKQ1", "SKQ4"]
            if identifier == "SK-DIALOGUE"
            else ["SKQ1", "SKQ2", "SKQ3"]
            if identifier == "SK-SIG"
            else ["SKQ1"],
            "support": "finite_game",
            "historical_reproduction": "gated",
        }
        for identifier, name in descriptions.items()
    ]
    entries += [
        {
            "id": g["id"],
            "family": "barrett",
            "name": g["name"],
            "source_questions": g["source_questions"],
            "support": "finite_adapter_with_resource_checks",
        }
        for g in catalogue()
        if g["id"] in _LEGACY_FINITE
    ]
    entries += [copy.deepcopy(value[1]) for value in _CUSTOM_FACTORIES.values()]
    return [e for e in entries if family is None or e["family"] == family]


def finite_game(profile: AdaptiveProfile) -> FiniteGame:
    profile.validate()
    budgets = {
        k: v
        for k, v in profile.resources.items()
        if k
        not in {
            "max_population_strategies",
            "max_measurements",
            "max_field_evaluations",
        }
    }
    if profile.model.startswith("SK-"):
        from .skyrms import skyrms_game

        game = skyrms_game(
            profile.model,
            profile.parameters,
            fixed_decisions=profile.fixed_decisions,
            **budgets,
        )
    elif profile.model in _CUSTOM_FACTORIES:
        game = _CUSTOM_FACTORIES[profile.model][0](
            copy.deepcopy(profile.parameters),
            fixed_decisions=copy.deepcopy(profile.fixed_decisions),
            **budgets,
        )
        if game.id != profile.model:
            raise ValueError("Registered factory returned a different game identifier")
    else:
        from .finite_adapters import barrett_game

        if set(profile.parameters) != {"profile"}:
            raise ValueError(
                "Barrett finite adapters require parameters={'profile': source_profile}"
            )
        game = barrett_game(profile.parameters["profile"], **budgets)
        if game.id != profile.model:
            raise ValueError("Barrett source profile and model disagree")
        if profile.fixed_decisions:
            game.fixed_decisions.update(copy.deepcopy(profile.fixed_decisions))
            game.__post_init__()
    if profile.kind == "paper_game" and game.fidelity.get("game", {}).get("questions"):
        raise ValueError(
            "Paper-game profile needs source resolution: "
            + ", ".join(game.fidelity["game"]["questions"])
        )
    if profile.kind == "paper_game" and game.fidelity.get("game", {}).get(
        "status", ""
    ).startswith("explicit"):
        raise ValueError(
            "Paper-game profiles require a source-defined game; label this a deliberate_variant"
        )
    if profile.fixed_decisions:
        game.fidelity["fixed_components"] = {
            "status": "explicit_phase_choice",
            "decisions": copy.deepcopy(profile.fixed_decisions),
        }
    return game


def adaptive_profiles(family: str | None = None) -> list[dict[str, Any]]:
    profiles = []
    for game in finite_catalogue("skyrms"):
        identifier = game["id"]
        slug = identifier.lower()
        parameters = (
            {"reply_binding": "partition_cell", "state_probabilities": [0.25] * 4}
            if identifier == "SK-DIALOGUE"
            else {}
        )
        for suffix, method in [
            ("evolution", "replicator.continuous"),
            ("r1", "reinforcement.conditional.R1"),
            ("strategy-r1", "reinforcement.strategy.R1"),
        ]:
            profiles.append(
                AdaptiveProfile(
                    f"{slug}.{suffix}.demo",
                    identifier,
                    parameters,
                    method=method,
                    assumptions=[
                        "Library demonstration protocol; historical reproduction remains gated."
                    ],
                ).to_dict()
            )
    profiles += [
        AdaptiveProfile(
            "sk-sig.shared.demo",
            "SK-SIG",
            population={"groups": {"population": ["sender", "receiver"]}},
        ).to_dict(),
        AdaptiveProfile(
            "sk-sig.biased.demo", "SK-SIG", {"state_probabilities": [1 / 3, 2 / 3]}
        ).to_dict(),
        AdaptiveProfile(
            "sk-sig.mutation.demo",
            "SK-SIG",
            {"state_probabilities": [1 / 3, 2 / 3]},
            method="replicator_mutator.birth_uniform",
            protocol={"mutation_rate": 0.01},
        ).to_dict(),
        AdaptiveProfile("sk-sig.n3.demo", "SK-SIG", {"states": 3}).to_dict(),
        AdaptiveProfile(
            "sk-comp.grouped.demo",
            "SK-COMP",
            population={"groups": {"senders": ["A", "B"], "receivers": ["receiver"]}},
            assumptions=[
                "Senders inherit distinct plans for both observational roles; SKQ5."
            ],
        ).to_dict(),
        AdaptiveProfile(
            "sk-chain.translator.demo",
            "SK-CHAIN",
            fixed_decisions={
                "sender/state:0": [1, 0],
                "sender/state:1": [0, 1],
                "receiver/signal:0": [0, 1],
                "receiver/signal:1": [1, 0],
            },
            assumptions=[
                "Constructed source translator fixture; endpoints remain fixed."
            ],
        ).to_dict(),
        AdaptiveProfile(
            "sk-dialogue.query-cell.demo",
            "SK-DIALOGUE",
            {"reply_binding": "query_cell", "state_probabilities": [0.25] * 4},
            kind="deliberate_variant",
        ).to_dict(),
    ]
    for case in ["sending", "listening", "state_specific"]:
        for interpretation in ["prose_derived", "table_literal"]:
            for suffix, method in [
                ("evolution", "replicator.continuous"),
                ("strategy-r1", "reinforcement.strategy.R1"),
            ]:
                profiles.append(
                    AdaptiveProfile(
                        f"sk-sig.cost-{case}.{interpretation}.{suffix}.demo",
                        "SK-SIG",
                        {"cost_case": case, "cost_interpretation": interpretation},
                        method=method,
                        kind="deliberate_variant",
                    ).to_dict()
                )
    for slug, model in [("sig", "SIG"), ("cod", "COD"), ("nand", "NAND")]:
        profiles.append(
            AdaptiveProfile(
                f"barrett.{slug}.evolution.demo", model, {"profile": f"{slug}.demo"}
            ).to_dict()
        )
    models = {e["id"] for e in finite_catalogue(family)}
    return [copy.deepcopy(p) for p in profiles if p["model"] in models]


def load_adaptive_profile(
    value: str | dict[str, Any] | AdaptiveProfile,
) -> AdaptiveProfile:
    from pathlib import Path
    import json

    if isinstance(value, AdaptiveProfile):
        result = copy.deepcopy(value)
    elif isinstance(value, dict):
        result = AdaptiveProfile(**copy.deepcopy(value))
    elif Path(value).is_file():
        result = AdaptiveProfile(**json.loads(Path(value).read_text()))
    else:
        matches = [p for p in adaptive_profiles() if p["id"] == value]
        if not matches:
            raise ValueError(
                f"Unknown adaptive profile {value}; use sg list --family all"
            )
        result = AdaptiveProfile(**matches[0])
    result.validate()
    return result
