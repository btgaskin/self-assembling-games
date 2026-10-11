"""Network interaction: observations, choices and credit in execution order."""

from __future__ import annotations

from typing import TYPE_CHECKING
from ..core import Address, Feedback, Outcome

if TYPE_CHECKING:
    from ..model import ModelInstance

from .common import rows, expected_signaling


def initialize(model_instance: ModelInstance) -> None:
    rt, p, model, rule = (
        model_instance.runtime,
        model_instance.parameters,
        model_instance.model,
        model_instance.learner,
    )
    n = p["agents"]
    for i in range(n):
        options = (
            ["nature"] + list(range(n))
            if model == "INQ"
            else [j for j in range(n) if j != i]
        )
        rows(rt, f"visit:{i}", ["start"], options, rule, owner=f"agent:{i}")
        if model == "FFSR":
            rows(
                rt,
                f"strategy:{i}",
                ["start"],
                list(range(16)),
                rule,
                owner=f"agent:{i}",
            )
    if model == "INQ":
        model_instance.world["beliefs"] = [False] * n
        for field in ["epistemic", "social"]:
            if field in p:
                model_instance.world[field] = list(p[field])
            elif field == "social" and p.get("perfect_communication"):
                model_instance.world[field] = [1.0] * n
            else:
                # Continuous initialization is saved in the checkpoint; evaluations condition on it.
                model_instance.world[field] = rt.rng.random(n).tolist()
                rt.emit("chance", label=field, values=model_instance.world[field])


def encounter(model_instance: ModelInstance) -> Outcome:
    if model_instance.model == "INQ":
        return inquiry_encounter(model_instance)
    rt, p, model = (
        model_instance.runtime,
        model_instance.parameters,
        model_instance.model,
    )
    n = p["agents"]
    if not model_instance.interaction.get("visit_order"):
        model_instance.interaction["visit_order"] = (
            rt.permutation("visit_order", n)
            if p["schedule"] == "random_permutation"
            else list(range(n))
        )
    visitor = model_instance.interaction["visit_order"].pop(0)
    visit = rt.decide(f"visit:{visitor}", "start", "start")
    host = visit.option
    if model == "STAG":
        types = p["types"]
        v, h = types[visitor], types[host]
        visitor_payoff = 3.0 if v == "hare" else 4.0 if h == "stag" else 0.0
        host_payoff = 3.0 if h == "hare" else 4.0 if v == "stag" else 0.0
    elif model == "FFSR":
        v = rt.decide(f"strategy:{visitor}", "start", "start")
        h = rt.decide(f"strategy:{host}", "start", "start")
        visitor_payoff = host_payoff = expected_signaling(v.option, h.option)
        rt.credit([v, h], Outcome(visitor_payoff > 0, visitor_payoff))
    else:
        visitor_payoff = host_payoff = p["payoff"]
    outcome = Outcome(
        visitor_payoff > 0,
        visitor_payoff,
        details={"visitor": visitor, "host": host, "host_payoff": host_payoff},
    )
    rt.credit([visit], outcome)
    if model != "FFSR" and p["reciprocal"]:
        rt.feedback(
            Feedback(
                Address(rt.table_id(f"visit:{host}", "start"), visitor),
                host_payoff > 0,
                host_payoff,
                (visit.occurrence,),
            )
        )
    if model == "FFSR" and rt.training:
        rt.retain_all_tables(p["retention"], p["floor"])
    if not model_instance.interaction["visit_order"]:
        rt.clocks["round"] += 1
    return outcome


def inquiry_encounter(model_instance: ModelInstance) -> Outcome:
    rt, p = model_instance.runtime, model_instance.parameters
    n = p["agents"]
    if not model_instance.interaction.get("inquiry_order"):
        model_instance.interaction["inquiry_order"] = rt.permutation("inquiry_order", n)
    i = model_instance.interaction["inquiry_order"].pop(0)
    source = rt.decide(f"visit:{i}", "start", "start")
    beliefs = model_instance.world["beliefs"]
    if source.option == "nature":
        correct = rt.bernoulli(
            "empirical_observation", model_instance.world["epistemic"][i]
        )
    elif source.option == i:
        correct = beliefs[i]
    else:
        correct = beliefs[source.option] and rt.bernoulli(
            "social_observation", model_instance.world["social"][i]
        )
    beliefs[i] = bool(correct)
    rt.emit("belief", agent=i, belief=bool(correct))
    outcome = Outcome(
        bool(correct), float(correct), details={"inquirer": i, "source": source.option}
    )
    rt.credit([source], outcome)
    if not model_instance.interaction["inquiry_order"]:
        rt.clocks["round"] += 1
        if rt.clocks["round"] % p["rounds_per_problem"] == 0:
            rt.clocks["problem"] += 1
            rt.reset_beliefs(n)
            rt.emit("reset", scope="beliefs", problem=rt.clocks["problem"])
    return outcome


def validate(profile):

    p = profile.parameters
    if profile.model in {"VIS", "STAG", "FFSR"} and p.get("schedule") not in {
        "ordered",
        "random_permutation",
    }:
        raise ValueError("Visit schedule must be explicit")

    if profile.model in {"VIS", "STAG", "FFSR"} and p["agents"] < 2:
        raise ValueError("Visiting networks need at least two agents")

    if profile.model == "STAG" and (
        len(p["types"]) != p["agents"]
        or any(t not in {"stag", "hare"} for t in p["types"])
    ):
        raise ValueError("STAG requires a stag/hare type for each agent")

    if profile.model == "FFSR":
        if not 0 < p["retention"] <= 1 or not 0 <= p["error"] <= 1:
            raise ValueError("Invalid FFSR retention/error")

    if profile.model == "INQ":
        for name in ["epistemic", "social"]:
            if name in p and (
                len(p[name]) != p["agents"] or any(not 0 <= x <= 1 for x in p[name])
            ):
                raise ValueError(f"{name} needs one probability per inquirer")
