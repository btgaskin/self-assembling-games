"""Signalling interaction: observations, choices and credit in execution order."""

from __future__ import annotations

from typing import TYPE_CHECKING
from ..core import Outcome

if TYPE_CHECKING:
    from ..model import ModelInstance

from .common import rows, fixed
from .participant_payoffs import (
    explicit_outcome,
    has_separate_payoffs,
    validate_separate_payoffs,
)


def resolve_parameters(model, parameters):
    parameters.setdefault(
        "state_probabilities", [1 / parameters["states"]] * parameters["states"]
    )
    return parameters


def initialize(model_instance: ModelInstance) -> None:
    rt, p, model, rule = (
        model_instance.runtime,
        model_instance.parameters,
        model_instance.model,
        model_instance.learner,
    )
    rows(rt, "sender", list(range(p["states"])), list(range(p["signals"])), rule)
    rows(rt, "receiver", list(range(p["signals"])), list(range(p["actions"])), rule)
    if model == "CUE":
        for x in range(p["states"]):
            fixed(rt, "sender", x, p["fixed_map"][x])
    if model == "MAN":
        for s in range(p["signals"]):
            fixed(rt, "receiver", s, p["fixed_map"][s])


def encounter(model_instance: ModelInstance) -> Outcome:
    rt, p = model_instance.runtime, model_instance.parameters
    x = rt.chance("nature", p["state_probabilities"])
    sender = rt.decide("sender", x, x)
    receiver = rt.decide("receiver", sender.option, sender.option)
    reward = (
        p["payoffs"][x][receiver.option]
        if "payoffs" in p
        else int(x == receiver.option)
    )
    success = (
        bool(p["success_matrix"][x][receiver.option])
        if "success_matrix" in p
        else x == receiver.option
    )
    if has_separate_payoffs(p):
        return explicit_outcome(
            rt,
            p,
            [sender, receiver],
            x,
            receiver.option,
            success,
            reward,
            sender_signal=sender.option,
        )
    outcome = Outcome(success, reward)
    if model_instance.learner["rule"] == "STRATEGY_WSLR":
        if not outcome.success:
            rt.resample_complete_assignment()
    else:
        rt.credit([sender, receiver], outcome)
    return outcome


def validate(profile):
    import math
    from ..registry import validate_distribution

    p, rule = profile.parameters, profile.learner["rule"]
    validate_separate_payoffs(profile, p["states"], p["actions"])
    if profile.model in {"SIG", "CUE", "MAN", "HYB"}:
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
                not math.isfinite(value)
                or (rule == "R1" and value < 0 and not has_separate_payoffs(p))
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
            raise ValueError("success_matrix must be a Boolean states × actions matrix")
        if "payoffs" not in p and n != p["actions"]:
            raise ValueError("Unequal states/actions require an explicit payoff matrix")
