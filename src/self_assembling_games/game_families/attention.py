"""Attention interaction: observations, choices and credit in execution order."""

from __future__ import annotations

from typing import TYPE_CHECKING
from ..core import Outcome

if TYPE_CHECKING:
    from ..model import ModelInstance

from .common import rows


def includes_module(model, module, parameters):
    return not (model == "ATT-R" and module == "sender_attention")


def initialize(model_instance: ModelInstance) -> None:
    rt, p, model, rule = (
        model_instance.runtime,
        model_instance.parameters,
        model_instance.model,
        model_instance.learner,
    )
    receiver_features = p["receiver_features"]
    contexts = (
        [0, 1]
        if model == "ATT-R" or p["binding"] == "value"
        else [
            [source, value]
            for source in range(p["sender_features"])
            for value in [0, 1]
        ]
    )
    rows(rt, "sender", contexts, [0, 1], rule)
    if model == "ATT-B":
        rows(
            rt,
            "sender_attention",
            ["start"],
            list(range(p["sender_features"])),
            rule,
        )
    rows(rt, "receiver_attention", ["start"], list(range(receiver_features)), rule)
    rows(
        rt,
        "receiver",
        [[f, v] for f in range(receiver_features) for v in [0, 1]],
        [0, 1],
        rule,
    )


def encounter(model_instance: ModelInstance) -> Outcome:
    rt, p, model = (
        model_instance.runtime,
        model_instance.parameters,
        model_instance.model,
    )
    state = rt.uniform("true_state", 2)
    choices = []
    if model == "ATT-B":
        source_values = [state] + [
            rt.uniform("source_distractor", 2) for _ in range(p["sender_features"] - 1)
        ]
        source = rt.decide("sender_attention", "start", "start")
        choices.append(source)
        observed = source_values[source.option]
        observation = {"source": source.option, "value": observed}
        context = (
            [source.option, observed] if p["binding"] == "source_value" else observed
        )
    else:
        context = state
        observation = state
    signal = rt.decide("sender", context, observation)
    choices.append(signal)
    feature_values = [signal.option] + [
        rt.uniform("receiver_distractor", 2) for _ in range(p["receiver_features"] - 1)
    ]
    salience = rt.decide("receiver_attention", "start", "start")
    choices.append(salience)
    value = feature_values[salience.option]
    receiver = rt.decide(
        "receiver",
        [salience.option, value],
        {"feature": salience.option, "value": value},
    )
    choices.append(receiver)
    outcome = Outcome(receiver.option == state, float(receiver.option == state))
    rt.credit(choices, outcome)
    return outcome


def validate(profile):

    p = profile.parameters
    if (
        profile.model in {"ATT-R", "ATT-B"}
        and p.get("distractor_law") != "independent_fair"
    ):
        raise ValueError("Implemented distractor_law is independent_fair")

    if profile.model == "ATT-B" and p.get("binding") not in {"source_value", "value"}:
        raise ValueError("ATT-B binding must be source_value or value")
