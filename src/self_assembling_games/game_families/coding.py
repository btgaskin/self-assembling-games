"""Coding interaction: observations, choices and credit in execution order."""

from __future__ import annotations

from typing import TYPE_CHECKING
from ..core import Outcome

if TYPE_CHECKING:
    from ..model import ModelInstance

from .common import rows


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
    x = rt.uniform("nature", 4)
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
    outcome = Outcome(action.option == target, float(action.option == target))
    rt.credit(choices, outcome)
    return outcome


def validate(profile):
    pass
