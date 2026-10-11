"""Hierarchy interaction: observations, choices and credit in execution order."""

from __future__ import annotations

from typing import TYPE_CHECKING
from ..core import Outcome

if TYPE_CHECKING:
    from ..model import ModelInstance

from .common import rows


def initialize(model_instance: ModelInstance) -> None:
    rt, rule = model_instance.runtime, model_instance.learner
    for module in ["A", "B"]:
        rows(rt, module, list(range(4)), [0, 1], rule)
    rows(rt, "ES", [0, 1, 2], ["A", "B", "AB"], rule)
    rows(rt, "ER", ["A", "B", "AB"], [0, 1, 2], rule)
    rows(rt, "R", [[a, b] for a in range(2) for b in range(2)], list(range(4)), rule)


def encounter(model_instance: ModelInstance) -> Outcome:
    rt, p = model_instance.runtime, model_instance.parameters
    x = rt.uniform("properties", 4)
    context = rt.uniform("context", 3)
    executive = rt.decide("ES", context, context)
    active = executive.option
    choices = [executive]
    signals = {}
    for module in ["A", "B"]:
        if module in active:
            d = rt.decide(module, x, x)
            choices.append(d)
            signals[module] = d.option
    pair = [
        signals["A"] if "A" in signals else rt.uniform("missing_A", 2),
        signals["B"] if "B" in signals else rt.uniform("missing_B", 2),
    ]
    action = rt.decide("R", pair, {"expression": signals})
    choices.append(action)
    interpretation = rt.decide("ER", active, active)
    choices.append(interpretation)
    projection = [lambda y: y // 2, lambda y: y % 2, lambda y: y][context]
    success = interpretation.option == context and projection(
        action.option
    ) == projection(x)
    outcome = Outcome(
        success,
        (p["complex_reward"] if context == 2 else p["simple_reward"])
        if success
        else 0.0,
        p["term_cost"] * len(signals),
        terms=len(signals),
    )
    rt.credit(choices, outcome)
    return outcome


def validate(profile):
    pass
