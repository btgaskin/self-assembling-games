"""Nand interaction: observations, choices and credit in execution order."""

from __future__ import annotations

from typing import TYPE_CHECKING
from ..core import Outcome

if TYPE_CHECKING:
    from ..model import ModelInstance

from .common import nand_rows, nand_call


def resolve_parameters(model, parameters):
    parameters.setdefault("truth_table", [1, 1, 1, 0])
    return parameters


def initialize(model_instance: ModelInstance) -> None:
    rt, rule = model_instance.runtime, model_instance.learner
    nand_rows(rt, "", rule)


def encounter(model_instance: ModelInstance) -> Outcome:
    rt, p = model_instance.runtime, model_instance.parameters
    a = rt.uniform("input_A", 2)
    b = rt.uniform("input_B", 2)
    output, choices = nand_call(rt, "", a, b)
    target = p["truth_table"][2 * a + b]
    outcome = Outcome(output == target, float(output == target))
    rt.credit(choices, outcome)
    return outcome


def validate(profile):
    pass
