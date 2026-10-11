"""Seed-independent construction choices and their source declarations."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from .definitions import GameDefinition, game_definition


@dataclass(frozen=True)
class SourceDeclaration:
    kind: str = "demonstration"
    identifiers: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    questions: tuple[str, ...] = ()

    def to_dict(self):
        return {
            k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()
        }


@dataclass(frozen=True)
class RealisationDefinition:
    id: str
    method_class: str
    representation: str


_REALISATIONS = {
    d.id: d
    for d in (
        RealisationDefinition(
            "barrett.conditional", "reinforcement", "conditional_tables"
        ),
        RealisationDefinition(
            "finite.conditional", "reinforcement", "conditional_decisions"
        ),
        RealisationDefinition(
            "finite.complete_plan", "reinforcement", "complete_role_plans"
        ),
        RealisationDefinition(
            "population.complete_plan",
            "population_evolution",
            "complete_inherited_plans",
        ),
    )
}


def realisation_definition(value):
    identifier = value.id if isinstance(value, RealisationDefinition) else value
    try:
        definition = _REALISATIONS[identifier]
    except KeyError:
        raise ValueError(f"Unknown realisation {identifier}") from None
    if isinstance(value, RealisationDefinition) and value != definition:
        raise ValueError("Realisation differs from its registered implementation")
    return definition


class ModelSpecification:
    """Defensively copied model choices, separate from outer run controls.

    ``resolve`` and ``from_profile`` use the same scientific validators. JSON
    backing gives each accessor its own nested values; caller edits cannot alter
    a resolved specification. Seeded assignments and imported state live in the
    constructed realisation.
    """

    __slots__ = ("_science", "_profile", "_protocol", "_finite_game")

    def __init__(self, *args, **kwargs):
        raise TypeError("Use ModelSpecification.resolve or from_profile")

    @classmethod
    def resolve(
        cls,
        *,
        game: str | GameDefinition,
        game_parameters: Mapping,
        realisation: str | RealisationDefinition,
        realisation_parameters: Mapping,
        organisation: Mapping | None = None,
        source: SourceDeclaration | None = None,
    ):
        from .registry import Profile
        from .finite_profiles import AdaptiveProfile

        definition = realisation_definition(realisation)
        game = game_definition(game, definition.id)
        source = source or SourceDeclaration()
        gp, rp, org = map(
            lambda x: copy.deepcopy(dict(x)),
            (game_parameters, realisation_parameters, organisation or {}),
        )
        if definition.id == "barrett.conditional":
            if org.keys() - {"owners", "functional_roles"}:
                raise ValueError("Barrett organisation accepts module owners")
            if "owners" in gp:
                raise ValueError("Native ownership belongs in organisation")
            if "owners" in org:
                gp["owners"] = org["owners"]
            if (
                "functional_roles" in org
                and org["functional_roles"]
                != game.initial_organisation(gp)["functional_roles"]
            ):
                raise ValueError(
                    "Functional role declarations must agree with the game definition"
                )
            profile = Profile(
                game.id + ".native",
                game.id,
                gp,
                rp,
                kind=source.kind,
                source=list(source.identifiers),
                assumptions=list(source.assumptions),
            )
        else:
            allowed = {
                "method",
                "fixed_decisions",
                "resources",
                "initialization",
                "initial_weight",
                "mutation_rate",
            }
            if rp.keys() - allowed:
                raise ValueError(
                    "Outer run controls belong in build(..., protocol=...)"
                )
            expected = {
                "finite.conditional": "reinforcement.conditional.R1",
                "finite.complete_plan": "reinforcement.strategy.R1",
                "population.complete_plan": "replicator.continuous",
            }[definition.id]
            method = rp.pop("method", expected)
            fixed = rp.pop("fixed_decisions", {})
            resources = rp.pop("resources", {})
            profile = AdaptiveProfile(
                game.id + ".native",
                game.id,
                gp,
                method=method,
                population=org,
                protocol=rp,
                fixed_decisions=fixed,
                resources=resources,
                kind=source.kind,
                assumptions=list(source.assumptions),
            )
        result = cls.from_profile(profile)
        if result.realisation_id != definition.id:
            raise ValueError("Adaptive method and realisation representation disagree")
        # Named open questions can supplement the source register. Claiming a
        # source reproduction still passes the registry's canonical gates.
        science = result.to_dict()
        science["source"]["identifiers"] = (
            list(source.identifiers) or science["source"]["identifiers"]
        )
        science["source"]["questions"] = list(
            dict.fromkeys(science["source"]["questions"] + list(source.questions))
        )
        if (
            source.kind in {"book_literal", "source_resolved", "paper_game"}
            and science["source"]["questions"]
        ):
            raise ValueError(
                "Source declaration needs resolution: "
                + ", ".join(science["source"]["questions"])
            )
        result._science = json.dumps(science, allow_nan=False)
        return result

    @classmethod
    def from_profile(cls, value, *, game=None):
        from .registry import Profile, catalogue, load_profile
        from .adaptive_runtime import is_adaptive_profile
        from .finite_profiles import load_adaptive_profile
        from .adaptive import adaptive_method

        obj = object.__new__(cls)
        obj._finite_game = None
        if is_adaptive_profile(value):
            profile = load_adaptive_profile(value)
            method = adaptive_method(profile.method)
            rid = (
                "population.complete_plan"
                if method.method_class == "population_evolution"
                else "finite.complete_plan"
                if method.representation == "complete_role_plans"
                else "finite.conditional"
            )
            definition = game_definition(profile.model, rid)
            obj._finite_game = (
                copy.deepcopy(game)
                if game is not None
                else definition.create_finite(profile)
            )
            # Overrides go through the same source/fixed-component checks.
            if game is not None:
                from .finite_profiles import validate_finite_game

                validate_finite_game(profile, obj._finite_game)
            rp = {
                "method": profile.method,
                **(
                    {
                        "initialization": {"kind": "dirichlet", "concentration": 1.0},
                        "mutation_rate": 0.0,
                    }
                    if method.method_class == "population_evolution"
                    else {"initial_weight": 1.0}
                ),
                **{
                    k: copy.deepcopy(v)
                    for k, v in profile.protocol.items()
                    if k in {"initialization", "initial_weight", "mutation_rate"}
                },
                "fixed_decisions": copy.deepcopy(profile.fixed_decisions),
                "resources": copy.deepcopy(profile.resources),
            }
            org = copy.deepcopy(profile.population)
            from .realisations import validate_finite_realisation

            validate_finite_realisation(obj._finite_game, method)
            if method.method_class == "population_evolution":
                from .organisation import PopulationSpec

                population = PopulationSpec(
                    obj._finite_game,
                    **org,
                    max_population_strategies=profile.resources.get(
                        "max_population_strategies", 100_000
                    ),
                )
                # Record the operative defaults without changing legacy JSON.
                org = {
                    k: v
                    for k, v in population.to_dict().items()
                    if k not in {"strategy_counts", "inheritance"}
                }
            questions = obj._finite_game.fidelity.get("game", {}).get("questions", [])
            identifiers = [method.source]
            defaults = {
                **(
                    {
                        "duration": 50.0,
                        "solver": "DOP853",
                        "rtol": 1e-9,
                        "atol": 1e-12,
                        "max_step": 1.0,
                        "sample_interval": 1.0,
                    }
                    if method.method_class == "population_evolution"
                    else {"plays": 10_000, "window": 1000}
                ),
                **{
                    k: v
                    for k, v in profile.protocol.items()
                    if k not in {"initialization", "initial_weight", "mutation_rate"}
                },
            }
        else:
            profile = load_profile(value)
            assert isinstance(profile, Profile)
            rid = "barrett.conditional"
            definition = game_definition(profile.model, rid)
            rp = copy.deepcopy(profile.learner)
            org = definition.initial_organisation(profile.parameters)
            entry = next(e for e in catalogue() if e["id"] == profile.model)
            questions, identifiers = entry["source_questions"], profile.source
            defaults = {
                "plays": profile.plays,
                "window": profile.parameters.get("window", 1000),
            }
        gp = (
            {
                k: copy.deepcopy(v)
                for k, v in profile.parameters.items()
                if k not in {"window", "owners"}
            }
            if rid == "barrett.conditional"
            else copy.deepcopy(profile.parameters)
        )
        if rid == "barrett.conditional":
            gp = definition.resolve_parameters(gp)
            from .learners import resolved_parameters

            effective = resolved_parameters(rp)
            initial_conditions = {
                "table_weight": 1.0,
                "reference": 0.0,
                "assignment": "uniform_option"
                if rp["rule"] in {"R6", "WSLS", "WSLR", "STRATEGY_WSLR"}
                else None,
            }
        else:
            effective, initial_conditions = copy.deepcopy(rp), None
        # Ownership is consumed by game initialization from the organisation
        # declaration; legacy JSON can continue spelling it inside parameters.
        obj._science = json.dumps(
            {
                "game": {
                    "id": definition.id,
                    "implementation": definition.implementation,
                    "parameters": gp,
                    **(
                        {"resolved_game": obj._finite_game.to_dict()}
                        if obj._finite_game is not None
                        else {}
                    ),
                },
                "organisation": org,
                "realisation": {
                    "id": rid,
                    "parameters": rp,
                    "effective_parameters": effective,
                    "initial_conditions": initial_conditions,
                },
                "source": SourceDeclaration(
                    profile.kind,
                    tuple(identifiers),
                    tuple(profile.assumptions),
                    tuple(questions),
                ).to_dict(),
            },
            allow_nan=False,
        )
        obj._profile = json.dumps(profile.to_dict(), allow_nan=False)
        obj._protocol = json.dumps(defaults, allow_nan=False)
        return obj

    def to_dict(self):
        return json.loads(self._science)

    def with_game(self, game):
        """Resolve an explicit finite override, including its source gates."""
        result = type(self).from_profile(self.to_profile(), game=game)
        science = result.to_dict()
        declared_source = self.to_dict()["source"]
        resolved_source = science["source"]
        science["source"] = {
            **declared_source,
            **{
                field: list(
                    dict.fromkeys(resolved_source[field] + declared_source[field])
                )
                for field in ("identifiers", "questions")
            },
        }
        result._science = json.dumps(science, allow_nan=False)
        return result

    def construct_finite_game(self):
        game = copy.deepcopy(self._finite_game)
        if game.to_dict() != self.to_dict()["game"]["resolved_game"]:
            raise ValueError("Resolved finite game has changed")
        return game

    def execution_parameters(self, protocol=None):
        """Validated compatibility view of scientific settings and run controls."""
        self.to_profile(protocol)
        controls = {**self.protocol, **copy.deepcopy(dict(protocol or {}))}
        return {
            **controls,
            **{
                k: v
                for k, v in self.realisation_parameters.items()
                if k in {"initialization", "initial_weight", "mutation_rate"}
            },
        }

    def restore_declaration(self, value):
        """Validate a checkpoint declaration against its compatibility config.

        Source identifiers and additional named questions can come from native
        construction. Operational choices must agree with the rebuilt model.
        """
        if value is None:
            return self
        saved, current = copy.deepcopy(value), self.to_dict()
        for key in ("game", "organisation", "realisation"):
            if saved.get(key) != current[key]:
                raise ValueError("Checkpoint model specification differs: " + key)
        source = saved.get("source", {})
        if (
            source.get("kind") != current["source"]["kind"]
            or source.get("assumptions") != current["source"]["assumptions"]
        ):
            raise ValueError("Checkpoint source declaration differs")
        if not set(current["source"]["questions"]) <= set(source.get("questions", [])):
            raise ValueError("Checkpoint omits registered source questions")
        if source["kind"] in {
            "book_literal",
            "source_resolved",
            "paper_game",
        } and source.get("questions"):
            raise ValueError("Checkpoint source declaration has open questions")
        self._science = json.dumps(saved, allow_nan=False)
        return self

    @property
    def game_id(self):
        return self.to_dict()["game"]["id"]

    @property
    def realisation_id(self):
        return self.to_dict()["realisation"]["id"]

    @property
    def game_definition(self):
        return game_definition(self.game_id, self.realisation_id)

    @property
    def game_parameters(self):
        return self.to_dict()["game"]["parameters"]

    @property
    def realisation_parameters(self):
        return self.to_dict()["realisation"]["parameters"]

    @property
    def organisation_parameters(self):
        return self.to_dict()["organisation"]

    @property
    def protocol(self):
        return json.loads(self._protocol)

    def to_profile(self, protocol=None):
        """Compatibility serialization of the resolved choices and run defaults."""
        from .registry import load_profile

        p = json.loads(self._profile)
        controls = copy.deepcopy(dict(protocol or {}))
        if self.realisation_id == "barrett.conditional":
            if controls.keys() - {"plays", "window"}:
                raise ValueError("Barrett outer protocol accepts plays and window")
            if "plays" in controls:
                p["plays"] = controls["plays"]
            if "window" in controls:
                p["parameters"]["window"] = controls["window"]
        else:
            if controls.keys() & {"initialization", "initial_weight", "mutation_rate"}:
                raise ValueError(
                    "Initialisation and adaptive law belong in realisation_parameters"
                )
            p["protocol"].update(controls)
        return load_profile(p)
