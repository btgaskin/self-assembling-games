"""Coding interaction: observations, choices and credit in execution order."""

from __future__ import annotations

from typing import TYPE_CHECKING
from ..core import Outcome

if TYPE_CHECKING:
    from ..model import ModelInstance

from .common import rows
from .participant_payoffs import (
    explicit_outcome,
    has_separate_payoffs,
    validate_separate_payoffs,
)


def initialize(model_instance: ModelInstance) -> None:
    rt, model, rule = (
        model_instance.runtime,
        model_instance.model,
        model_instance.learner,
    )
    for module in ["A", "B"]:
        rows(rt, module, list(range(4)), [0, 1], rule)
    pairs = [[a, b] for a in range(2) for b in range(2)]
    rows(rt, "R", pairs, list(range(4)), rule)
    if model == "SCOD":
        rows(rt, "salience", ["start"], ["A", "B", "AB"], rule)
        for module in ["RA", "RB"]:
            rows(rt, module, [0, 1], list(range(4)), rule)


def encounter(model_instance: ModelInstance) -> Outcome:
    rt, p, model = (
        model_instance.runtime,
        model_instance.parameters,
        model_instance.model,
    )
    x = rt.chance("nature", p.get("state_probabilities", [0.25] * 4))
    a = rt.decide("A", x, x)
    b = rt.decide("B", x, x)
    choices = [a, b]
    if model == "SCOD":
        selected_attention = rt.decide("salience", "start", "start")
        choices.append(selected_attention)
        module, context = (
            ("R", [a.option, b.option])
            if selected_attention.option == "AB"
            else (
                "R" + selected_attention.option,
                a.option if selected_attention.option == "A" else b.option,
            )
        )
    else:
        module, context = "R", [a.option, b.option]
    action = rt.decide(module, context, context)
    choices.append(action)
    target = (
        p["new_mapping"][x]
        if model == "CHANGE" and rt.clocks["play"] >= p["change_at"]
        else x
    )
    success = (
        bool(p["success_matrix"][x][action.option])
        if "success_matrix" in p
        else action.option == target
    )
    if has_separate_payoffs(p):
        return explicit_outcome(
            rt, p, choices, x, action.option, success, float(success)
        )
    outcome = Outcome(success, float(success))
    rt.credit(choices, outcome)
    return outcome


def validate(profile):
    from ..registry import validate_distribution

    if "state_probabilities" in profile.parameters:
        validate_distribution(profile.parameters["state_probabilities"], 4)
    validate_separate_payoffs(profile, 4, 4)
