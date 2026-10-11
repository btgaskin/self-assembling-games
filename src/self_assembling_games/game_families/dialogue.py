"""Dialogue interaction: observations, choices and credit in execution order."""

from __future__ import annotations

from typing import TYPE_CHECKING
from ..core import Outcome

if TYPE_CHECKING:
    from ..model import ModelInstance

from .common import rows


def includes_module(model, module, parameters):
    return not (model == "ASK" and module == "reply2")


def resolve_parameters(model, parameters):
    parameters.setdefault("exchange_cost", 0.0)
    parameters.setdefault("failure_cost", True)
    return parameters


def initialize(model_instance: ModelInstance) -> None:
    rt, p, model, rule = (
        model_instance.runtime,
        model_instance.parameters,
        model_instance.model,
        model_instance.learner,
    )
    answers = p["answers"]
    materials = 2 if model == "ASK" else 4
    questions = ["Q"] if model == "ASK" else ["Q0", "Q1"]
    choices = list(range(materials)) + questions
    rows(rt, "assistant", ["start"], choices, rule)
    rows(rt, "reply1", list(range(answers)), choices, rule)
    if model != "ASK":
        rows(
            rt,
            "reply2",
            [[a, b] for a in range(answers) for b in range(answers)],
            choices,
            rule,
        )
    rows(
        rt,
        "builder",
        [[q, m] for q in questions for m in range(materials)],
        list(range(answers)),
        rule,
    )


def encounter(model_instance: ModelInstance) -> Outcome:
    rt, p, model = (
        model_instance.runtime,
        model_instance.parameters,
        model_instance.model,
    )
    material = rt.uniform("needed_material", 2 if model == "ASK" else 4)
    choices = []
    answers = []
    exchanges = 0
    module, context = "assistant", "start"
    while True:
        action = rt.decide(module, context, context)
        choices.append(action)
        rt.clocks["turn"] += 1
        if isinstance(action.option, int):
            success = action.option == material
            reward = p["base_reward"] if success else 0.0
            cost = p["exchange_cost"] * exchanges
            if not success and p["failure_cost"] is False:
                cost = 0.0
            outcome = Outcome(success, reward, cost, exchanges=exchanges)
            rt.credit(choices, outcome)
            return outcome
        if exchanges >= p["max_exchanges"]:
            if model in {"DIAL", "DIAL-restricted"}:
                outcome = Outcome(False, exchanges=exchanges)
                rt.credit(choices, outcome)
            else:
                # Engineering censoring supplies no learning update.
                outcome = Outcome(False, exchanges=exchanges, status="truncated")
            return outcome
        reply = rt.decide(
            "builder", [action.option, material], [action.option, material]
        )
        choices.append(reply)
        answers.append(reply.option)
        exchanges += 1
        rt.clocks["turn"] += 1
        if model == "ASK" or exchanges == 1:
            module, context = "reply1", reply.option
        else:
            module, context = "reply2", answers[-2:]
            identity = rt.table_id(module, context)
            if identity not in rt.tables:
                rt.add_table(
                    module,
                    context,
                    list(range(4)) + ["Q0", "Q1"],
                    model_instance.learner,
                )


def validate(profile):

    p = profile.parameters
    if profile.model == "DIAL-costly" and p["history_binding"] != "last_two_answers":
        raise ValueError(
            "Costly variant requires explicit last_two_answers history binding"
        )

    if profile.model in {"DIAL", "DIAL-restricted"} and p["max_exchanges"] != 2:
        raise ValueError("Finite dialogue terminates on the third question")

    if profile.model in {"ASK", "DIAL-costly"}:
        if p.get("negative_returns") != "floor_one":
            raise ValueError("Costly demonstration requires negative_returns=floor_one")
