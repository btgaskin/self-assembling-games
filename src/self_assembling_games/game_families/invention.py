"""Invention interaction: observations, choices and credit in execution order."""

from __future__ import annotations

from typing import TYPE_CHECKING
from ..core import Address, Feedback, Outcome

if TYPE_CHECKING:
    from ..model import ModelInstance

from .common import rows, relation, forget_signal


def resolve_parameters(model, parameters):
    parameters.setdefault(
        "state_probabilities", [1 / parameters["n"]] * parameters["n"]
    )
    return parameters


def initialize(model_instance: ModelInstance) -> None:
    rt, p, model = (
        model_instance.runtime,
        model_instance.parameters,
        model_instance.model,
    )
    modules = ["sender"] if model == "INV" else ["A", "B"]
    for module in modules:
        rows(rt, module, list(range(p["n"])), ["mutator"], {"rule": "R1"})


def encounter(model_instance: ModelInstance) -> Outcome:
    rt, p, model = (
        model_instance.runtime,
        model_instance.parameters,
        model_instance.model,
    )
    modules = ["sender"] if model == "INV" else ["A", "B"]
    inputs = [rt.chance("input:" + m, p["state_probabilities"]) for m in modules]
    draws = [rt.decide(m, x, x) for m, x in zip(modules, inputs)]
    proposed = [d.option == "mutator" for d in draws]
    signals = [
        rt.fresh(d.module) if new else d.option for d, new in zip(draws, proposed)
    ]
    context = signals[0] if model == "INV" else signals
    identity = rt.table_id("R", context)
    created = identity not in rt.tables
    count = p["n"] if model == "INV" else 3 if model == "ORD" else 2 * p["n"] - 1
    if created:
        if len(rt.tables) >= p["max_tables"]:
            return Outcome(
                False, status="resource_limit", details={"limit": "max_tables"}
            )
        rt.add_table("R", context, list(range(count)), {"rule": "R1"})
    receiver = rt.decide("R", context, context)
    target = (
        inputs[0]
        if model == "INV"
        else relation(*inputs)
        if model == "ORD"
        else sum(inputs)
    )
    success = receiver.option == target
    outcome = Outcome(success, float(success))
    if rt.training:
        if success:
            for d, new, signal in zip(draws, proposed, signals):
                if new:
                    for address, table in list(rt.tables.items()):
                        if table.module == d.module:
                            rt.insert(address, signal)
                    rt.feedback(
                        Feedback(Address(d.table, signal), True, 1.0, (d.occurrence,))
                    )
                else:
                    rt.credit([d], outcome)
            rt.credit([receiver], outcome)
            rt.emit("commit_invention", signals=signals)
        elif (
            created
            and any(proposed)
            and (model == "INV" or p["failed_row"] == "discard")
        ):
            rt.remove(identity)
            rt.emit("rollback_invention", signals=signals)
        if (
            model == "INV"
            and p.get("forgetting")
            and (rt.clocks["play"] + 1) % p["forgetting"]["every"] == 0
        ):
            forget_signal(rt, p["forgetting"])
    return outcome


def validate(profile):

    p = profile.parameters
    if profile.model in {"ORD", "ADD"} and p.get("failed_row") not in {
        "discard",
        "retain",
    }:
        raise ValueError("Pair invention requires failed_row=discard or retain")

    if profile.model == "ADD" and p["n"] != 6:
        raise ValueError("ADD's implemented domain is zero through five")

    if profile.model == "INV" and p.get("forgetting"):
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
            raise ValueError("Invalid forgetting policy; empty urns must be protected")
        if profile.kind in {"book_literal", "source_resolved"}:
            raise ValueError(
                "Invention forgetting remains an explicit variant pending Q3"
            )
