"""Task interaction: observations, choices and credit in execution order."""

from __future__ import annotations

import copy

from typing import TYPE_CHECKING
from ..core import Outcome

if TYPE_CHECKING:
    from ..model import ModelInstance

from .common import rows


def initialize(model_instance: ModelInstance) -> None:
    rt, p, model, rule = (
        model_instance.runtime,
        model_instance.parameters,
        model_instance.model,
        model_instance.learner,
    )
    count = (
        len(p["arms"])
        if model == "BAND"
        else len(p["reliabilities"])
        if model == "META-ORACLE"
        else len(p["rules"])
        if model == "META-RULE"
        else 2
    )
    rows(rt, "choice", ["start"], list(range(count)), rule)


def encounter(model_instance: ModelInstance) -> Outcome:
    rt, p, model = (
        model_instance.runtime,
        model_instance.parameters,
        model_instance.model,
    )
    choice = rt.decide("choice", "start", "start")
    if model == "BOX":
        rewarded = model_instance.world["special"]
        if p["placement"] == "random":
            rewarded = (
                rewarded if rt.bernoulli("food_placement", 2 / 3) else 1 - rewarded
            )
        success = choice.option == rewarded
        reward = float(success)
    elif model == "BAND":
        arm = p["arms"][choice.option]
        reward = arm["rewards"][rt.chance("machine_payoff", arm["probabilities"])]
        success = True
    elif model == "META-ORACLE":
        success = rt.bernoulli("oracle_answer", p["reliabilities"][choice.option])
        reward = 1.0 if success else p["failure_return"]
    elif model == "META-RULE":
        from ..registry import Profile
        from ..simulation import build

        seed = rt.integer("fresh_inner_seed", 2**32)
        selected = p["rules"][choice.option]
        inner = build(
            Profile(
                id="inner",
                model=p["inner_model"],
                parameters=copy.deepcopy(p["inner_parameters"]),
                learner=selected,
                plays=p["inner_plays"],
            ),
            seed=seed,
        )
        inner.run(plays=p["inner_plays"])
        accuracy = inner.metrics.summary()["cumulative_accuracy"]
        success = accuracy > p["threshold"]
        reward = (
            accuracy
            if p["return"] == "accuracy"
            else 1.0
            if success
            else p["failure_return"]
        )
        model_instance.interaction["last_inner"] = {
            "seed": seed,
            "rule": selected,
            "accuracy": accuracy,
        }
        rt.clocks["problem"] += 1
        rt.emit("inner_task", **model_instance.interaction["last_inner"])
    else:
        prize = rt.uniform("prize", 3)
        initial = rt.uniform("initial_door", 3)
        eligible = [d for d in range(3) if d != prize and d != initial]
        opened = eligible[rt.uniform("host_goat_door", len(eligible))]
        selected = (
            initial
            if choice.option == 0
            else next(d for d in range(3) if d not in {initial, opened})
        )
        success = selected == prize
        reward = float(success)
    outcome = Outcome(success, reward)
    rt.credit([choice], outcome)
    return outcome


def validate(profile):
    from ..registry import validate_distribution

    p = profile.parameters
    if profile.model in {"META-ORACLE", "META-RULE"} and p["failure_return"] <= 0:
        raise ValueError("Meta profiles require a positive failure return")

    if profile.model == "META-ORACLE":
        if (
            p["answer_process"] != "independent_bernoulli"
            or not p["reliabilities"]
            or any(not 0 <= r <= 1 for r in p["reliabilities"])
        ):
            raise ValueError(
                "Oracle variant requires independent Bernoulli reliabilities"
            )

    if profile.model == "META-RULE" and (
        p["return"] not in {"accuracy", "threshold"} or not 0 <= p["threshold"] <= 1
    ):
        raise ValueError("Meta return is accuracy or threshold with a cutoff in [0,1]")

    if profile.model == "MH" and p.get("host") != "uniform_eligible_goat":
        raise ValueError("Implemented host protocol is uniform_eligible_goat")

    if profile.model == "BAND":
        for arm in p["arms"]:
            validate_distribution(arm["probabilities"], len(arm["rewards"]))
            if any(not 1 <= r <= 10 for r in arm["rewards"]):
                raise ValueError("BAND rewards must be between one and ten")
