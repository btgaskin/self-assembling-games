"""Explicit profiles and a complete, source-linked catalogue."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

RULES = {
    "R1",
    "R2",
    "SIGNED",
    "R3",
    "R4",
    "R6",
    "WSLS",
    "WSLR",
    "STRATEGY_WSLR",
    "BOUNDED",
    "DISCOUNT",
}


def catalogue() -> list[dict[str, Any]]:
    return json.loads(files(__package__).joinpath("data/catalogue.json").read_text())


def profiles() -> list[dict[str, Any]]:
    return json.loads(files(__package__).joinpath("data/profiles.json").read_text())


def reproduction_protocols() -> list[dict[str, Any]]:
    return json.loads(
        files(__package__).joinpath("data/reproductions.json").read_text()
    )


@dataclass
class Profile:
    id: str
    model: str
    parameters: dict[str, Any] = field(default_factory=dict)
    learner: dict[str, Any] = field(default_factory=lambda: {"rule": "R1"})
    kind: str = "demonstration"
    plays: int = 10_000
    source: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(vars(self))

    def validate(self) -> None:
        import math

        entries = {m["id"]: m for m in catalogue()}
        if self.model not in entries:
            raise ValueError(f"Unknown model {self.model}")
        entry = entries[self.model]
        if not entry["implemented"]:
            raise ValueError(
                f"{self.model} is a catalogue entry awaiting an operational protocol: "
                + ", ".join(entry["required_fields"])
            )
        if self.kind not in {
            "book_literal",
            "source_resolved",
            "demonstration",
            "deliberate_variant",
        }:
            raise ValueError("Unknown profile kind")
        if (
            self.kind in {"book_literal", "source_resolved"}
            and entry["source_questions"]
        ):
            raise ValueError(
                f"Source profile {self.model} needs resolution: "
                + ", ".join(entry["source_questions"])
            )
        missing = [f for f in entry["required_fields"] if f not in self.parameters]
        if missing:
            raise ValueError("Missing profile fields: " + ", ".join(missing))
        rule = self.learner.get("rule")
        if rule not in RULES or rule not in entry["learners"]:
            raise ValueError(f"Learner {rule} is unsupported for {self.model}")
        for name in ["reward", "punishment", "floor", "maximum"]:
            if name in self.learner and (
                not math.isfinite(self.learner[name]) or self.learner[name] < 0
            ):
                raise ValueError(f"Invalid learner parameter {name}")
        for name in [
            "epsilon",
            "forgetting",
            "w_plus",
            "w_minus",
            "error",
            "retention",
        ]:
            if name in self.learner and not 0 <= self.learner[name] <= 1:
                raise ValueError(f"Learner {name} must be in [0,1]")
        if not isinstance(self.plays, int) or self.plays <= 0:
            raise ValueError("plays must be a positive integer")
        p = self.parameters
        for field_name in [
            "n",
            "states",
            "signals",
            "actions",
            "phase_length",
            "window",
            "agents",
            "rounds_per_problem",
            "inner_plays",
            "max_exchanges",
            "answers",
            "receiver_features",
            "sender_features",
            "max_tables",
        ]:
            if field_name in p and (
                not isinstance(p[field_name], int) or p[field_name] <= 0
            ):
                raise ValueError(f"{field_name} must be a positive integer")
        if rule == "R3" and (
            not isinstance(self.learner.get("phase_length"), int)
            or self.learner["phase_length"] <= 0
        ):
            raise ValueError("R3 requires positive phase_length")
        if rule == "R4":
            missing = {
                "epsilon",
                "floor",
                "forgetting",
                "w_plus",
                "w_minus",
            } - self.learner.keys()
            if missing:
                raise ValueError("R4 requires " + ", ".join(sorted(missing)))
            if self.model not in {"BAND", "META-ORACLE"}:
                raise ValueError(
                    "R4's source scope is pinned here to single-vector tasks"
                )
        if rule == "BOUNDED" and self.learner.get("maximum", 0) < 1:
            raise ValueError("BOUNDED requires maximum >= 1")
        if rule == "DISCOUNT" and not 0 < self.learner.get("retention", 0) <= 1:
            raise ValueError("DISCOUNT requires 0 < retention <= 1")
        if (
            self.model in {"ATT-R", "ATT-B"}
            and p.get("distractor_law") != "independent_fair"
        ):
            raise ValueError("Implemented distractor_law is independent_fair")
        if self.model == "ATT-B" and p.get("binding") not in {"source_value", "value"}:
            raise ValueError("ATT-B binding must be source_value or value")
        if self.model in {"VIS", "STAG", "FFSR"} and p.get("schedule") not in {
            "ordered",
            "random_permutation",
        }:
            raise ValueError("Visit schedule must be explicit")
        if self.model in {"ORD", "ADD"} and p.get("failed_row") not in {
            "discard",
            "retain",
        }:
            raise ValueError("Pair invention requires failed_row=discard or retain")
        if self.model == "ADD-COMP" and p.get("ties") not in {
            "left",
            "right",
            "exclude",
        }:
            raise ValueError("Addition/comparison requires an explicit tie policy")
        if self.model in {"VIS", "STAG", "FFSR"} and p["agents"] < 2:
            raise ValueError("Visiting networks need at least two agents")
        if self.model == "STAG" and (
            len(p["types"]) != p["agents"]
            or any(t not in {"stag", "hare"} for t in p["types"])
        ):
            raise ValueError("STAG requires a stag/hare type for each agent")
        if self.model == "BOX" and p["placement"] not in {"fixed", "random"}:
            raise ValueError("BOX placement must be fixed or random")
        if (
            self.model == "TR-PART"
            and p["pair_sampler"] != "uniform_adjacent_both_directions"
        ):
            raise ValueError(
                "Implemented partial sampler is uniform_adjacent_both_directions"
            )
        if self.model == "COMP-NAND" and p["selectors"] not in {1, 2}:
            raise ValueError("COMP-NAND has one or two selectors")
        if self.model == "ADD" and p["n"] != 6:
            raise ValueError("ADD's implemented domain is zero through five")
        if self.model == "INV" and p.get("forgetting"):
            f = p["forgetting"]
            needed = {
                "every",
                "mode",
                "protect_mutator",
                "cleanup",
                "protect_empty_urn",
            }
            if needed - f.keys():
                raise ValueError("Forgetting requires " + ", ".join(sorted(needed)))
            if (
                f["every"] <= 0
                or f["mode"] not in {"type", "ball"}
                or f["cleanup"] not in {"retain_receiver", "remove_orphan_receiver"}
                or f["protect_empty_urn"] is not True
            ):
                raise ValueError(
                    "Invalid forgetting policy; empty urns must be protected"
                )
            if self.kind in {"book_literal", "source_resolved"}:
                raise ValueError(
                    "Invention forgetting remains an explicit variant pending Q3"
                )
        if self.model == "DIAL-costly" and p["history_binding"] != "last_two_answers":
            raise ValueError(
                "Costly variant requires explicit last_two_answers history binding"
            )
        if self.model in {"DIAL", "DIAL-restricted"} and p["max_exchanges"] != 2:
            raise ValueError("Finite dialogue terminates on the third question")
        if self.model in {"META-ORACLE", "META-RULE"} and p["failure_return"] <= 0:
            raise ValueError("Meta profiles require a positive failure return")
        if self.model == "META-ORACLE":
            if (
                p["answer_process"] != "independent_bernoulli"
                or not p["reliabilities"]
                or any(not 0 <= r <= 1 for r in p["reliabilities"])
            ):
                raise ValueError(
                    "Oracle variant requires independent Bernoulli reliabilities"
                )
        if self.model == "META-RULE" and (
            p["return"] not in {"accuracy", "threshold"} or not 0 <= p["threshold"] <= 1
        ):
            raise ValueError(
                "Meta return is accuracy or threshold with a cutoff in [0,1]"
            )
        if self.model == "MH" and p.get("host") != "uniform_eligible_goat":
            raise ValueError("Implemented host protocol is uniform_eligible_goat")
        if self.model in {"ASK", "DIAL-costly"}:
            if p.get("negative_returns") != "floor_one":
                raise ValueError(
                    "Costly demonstration requires negative_returns=floor_one"
                )
        if self.model == "FFSR":
            if not 0 < p["retention"] <= 1 or not 0 <= p["error"] <= 1:
                raise ValueError("Invalid FFSR retention/error")
        if self.model in {"TR-ORD", "TR-PART", "TR-LOG", "COMP-NAND", "ADD-COMP"}:
            mode = p.get("template_mode")
            if mode not in {"constructed", "artifact"}:
                raise ValueError("template_mode must be constructed or artifact")
            if mode == "constructed" and self.kind in {
                "book_literal",
                "source_resolved",
            }:
                raise ValueError(
                    "Constructed fixtures belong to demonstration or deliberate_variant profiles"
                )
            if mode == "artifact" and not p.get("templates"):
                raise ValueError("Artifact transfer requires templates")
        if self.model == "INQ":
            for name in ["epistemic", "social"]:
                if name in p and (
                    len(p[name]) != p["agents"] or any(not 0 <= x <= 1 for x in p[name])
                ):
                    raise ValueError(f"{name} needs one probability per inquirer")
        if self.model in {"SIG", "CUE", "MAN", "HYB"}:
            n = p["states"]
            if "state_probabilities" in p:
                validate_distribution(p["state_probabilities"], n)
            if "payoffs" in p and (
                len(p["payoffs"]) != n
                or any(len(row) != p["actions"] for row in p["payoffs"])
            ):
                raise ValueError("payoffs must have shape states × actions")
            if "payoffs" in p:
                if "success_matrix" not in p:
                    raise ValueError("Custom payoffs require a separate success_matrix")
                if any(
                    not math.isfinite(value) or (rule == "R1" and value < 0)
                    for row in p["payoffs"]
                    for value in row
                ):
                    raise ValueError("Invalid payoff for the selected learner")
            if "success_matrix" in p and (
                len(p["success_matrix"]) != n
                or any(
                    len(row) != p["actions"]
                    or any(value not in {False, True} for value in row)
                    for row in p["success_matrix"]
                )
            ):
                raise ValueError(
                    "success_matrix must be a Boolean states × actions matrix"
                )
            if "payoffs" not in p and n != p["actions"]:
                raise ValueError(
                    "Unequal states/actions require an explicit payoff matrix"
                )
        if self.model == "BAND":
            for arm in p["arms"]:
                validate_distribution(arm["probabilities"], len(arm["rewards"]))
                if any(not 1 <= r <= 10 for r in arm["rewards"]):
                    raise ValueError("BAND rewards must be between one and ten")


def validate_distribution(values: list[float], count: int) -> None:
    import math

    if (
        len(values) != count
        or any(not math.isfinite(x) or x < 0 for x in values)
        or not math.isclose(sum(values), 1, rel_tol=1e-12, abs_tol=1e-12)
    ):
        raise ValueError("Invalid probability distribution")


def load_profile(identifier: str | Path | dict[str, Any] | Profile) -> Profile:
    from .adaptive_runtime import is_adaptive_profile

    if is_adaptive_profile(identifier):
        from .finite_profiles import load_adaptive_profile

        return load_adaptive_profile(identifier)
    if isinstance(identifier, Profile):
        profile = copy.deepcopy(identifier)
    elif isinstance(identifier, dict):
        profile = Profile(**copy.deepcopy(identifier))
    else:
        path = Path(identifier)
        if path.is_file():
            profile = Profile(**json.loads(path.read_text()))
        else:
            matches = [p for p in profiles() if p["id"] == str(identifier)]
            if not matches:
                raise ValueError(f"Unknown profile {identifier}; use sg list")
            profile = Profile(**matches[0])
    profile.validate()
    return profile
