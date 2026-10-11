"""Composition interaction: observations, choices and credit in execution order."""

from __future__ import annotations

from typing import TYPE_CHECKING
from ..core import Outcome

if TYPE_CHECKING:
    from ..model import ModelInstance

from .common import rows, install_nand, nand_call, ordering_call, relation


def includes_module(model, module, parameters):
    return not (model == "COMP-NAND" and module == "Q" and parameters["selectors"] == 1)


def initialize(model_instance: ModelInstance) -> None:
    rt, p, model, rule = (
        model_instance.runtime,
        model_instance.parameters,
        model_instance.model,
        model_instance.learner,
    )
    if model == "TR-LOG":
        install_nand(model_instance, "template:", "template")
        for module in ["TA", "TB"]:
            rows(rt, module, [0, 1], [0, 1], rule)
    else:
        for name in ["first", "second"]:
            install_nand(model_instance, name + ":", name)
        rows(rt, "P", ["start"], ["Z", "first"], rule)
        if p["selectors"] == 2:
            rows(rt, "Q", ["start"], ["Q", "first"], rule)


def encounter(model_instance: ModelInstance) -> Outcome:
    rt, p, model = (
        model_instance.runtime,
        model_instance.parameters,
        model_instance.model,
    )
    if model == "TR-LOG":
        a, b = rt.uniform("input_A", 2), rt.uniform("input_B", 2)
        da, db = rt.decide("TA", a, a), rt.decide("TB", b, b)
        output, _ = nand_call(rt, "template:", da.option, db.option)
        target = p["truth_table"][a * 2 + b]
        outcome = Outcome(output == target, float(output == target))
        rt.credit([da, db], outcome)
        return outcome
    if model == "COMP-NAND":
        x, y, q, z = [rt.uniform(name, 2) for name in ["X", "Y", "Q", "Z"]]
        first = rt.invoke("first", lambda: nand_call(rt, "first:", x, y)[0])
        selector = rt.decide("P", "start", "start")
        choices = [selector]
        a = z if selector.option == "Z" else first
        b = q
        if p["selectors"] == 2:
            second_selector = rt.decide("Q", "start", "start")
            choices.append(second_selector)
            b = q if second_selector.option == "Q" else first
        output = rt.invoke("second", lambda: nand_call(rt, "second:", a, b)[0])
        target = 1 - (1 - x * y) * q
        outcome = Outcome(output == target, float(output == target))
        rt.credit(choices, outcome)
        return outcome
    if model in {"TR-ORD", "TR-PART"}:
        if model == "TR-PART":
            a = rt.uniform("adjacent_pair", p["n"] - 1)
            b = a + 1
            if rt.uniform("pair_direction", 2):
                a, b = b, a
        else:
            a, b = rt.uniform("new_A", p["n"]), rt.uniform("new_B", p["n"])
        da, db = rt.decide("TA", a, a), rt.decide("TB", b, b)
        output = (
            relation(da.option, db.option)
            if p["template_mode"] == "constructed"
            else ordering_call(rt, da.option, db.option)
        )
        outcome = Outcome(output == relation(a, b), float(output == relation(a, b)))
        rt.credit([da, db], outcome)
        return outcome
    a, b = rt.uniform("addend_A", 6), rt.uniform("addend_B", 6)
    if p["template_mode"] == "constructed":
        summed = a + b
    else:
        da = rt.decide("addition:A", a, a)
        db = rt.decide("addition:B", b, b)
        summed = rt.decide(
            "addition:R", [da.option, db.option], [da.option, db.option]
        ).option
    eligible = [r for r in range(11) if p["ties"] != "exclude" or r != a + b]
    r = eligible[rt.uniform("right_cardinality", len(eligible))]
    translation = rt.decide("translation", summed, summed)
    interpreted = (
        relation(translation.option, r)
        if p["template_mode"] == "constructed"
        else ordering_call(rt, translation.option, r)
    )
    expected = relation(a + b, r)
    tie_answer = 1 if p["ties"] == "left" else 0
    output = tie_answer if interpreted == 2 else interpreted
    target = tie_answer if expected == 2 else expected
    outcome = Outcome(output == target, float(output == target))
    rt.credit([translation], outcome)
    return outcome


def validate(profile):

    p = profile.parameters
    if profile.model == "COMP-NAND" and p["selectors"] not in {1, 2}:
        raise ValueError("COMP-NAND has one or two selectors")

    if profile.model in {"TR-ORD", "TR-PART", "TR-LOG", "COMP-NAND", "ADD-COMP"}:
        mode = p.get("template_mode")
        if mode not in {"constructed", "artifact"}:
            raise ValueError("template_mode must be constructed or artifact")
        if mode == "constructed" and profile.kind in {
            "book_literal",
            "source_resolved",
        }:
            raise ValueError(
                "Constructed fixtures belong to demonstration or deliberate_variant profiles"
            )
        if mode == "artifact" and not p.get("templates"):
            raise ValueError("Artifact transfer requires templates")
