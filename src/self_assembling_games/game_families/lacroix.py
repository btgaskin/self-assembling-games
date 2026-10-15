"""LaCroix's contextual logical composition, BJPS axz049, sections 3.2/3.5.

State and receiver actions use 2*game + input, with game 0=taut and 1=neg.
The source task checks the context-relevant symbolic coordinates. Equal evaluated
Boolean outputs alone do not establish success (Figure 8). Unary meanings remain
supplied in these models; the senders' representational roles can be learned.
"""

from __future__ import annotations

from ..core import Outcome
from .common import rows

CASES = {
    "stipulated_roles": ("assigned_property", "assigned_roles"),
    "role_free": ("full_world", "minimal_signals"),
    "full_observation_fixed_roles": ("full_world", "assigned_roles"),
}


def resolve_parameters(model, parameters):
    case = parameters.get("case")
    if case in CASES:
        observation, relevance = CASES[case]
        parameters.setdefault("sender_observation", observation)
        parameters.setdefault("relevance", relevance)
    return parameters


def initialize(instance):
    rt, p, rule = instance.runtime, instance.parameters, instance.learner
    contexts = list(range(2 if p["sender_observation"] == "assigned_property" else 4))
    for sender in ["A", "B"]:
        rows(rt, sender, contexts, [0, 1], rule)
    rows(rt, "ES", [0, 1, 2], ["A", "B", "AB"], rule)
    rows(rt, "ER", ["A", "B", "AB"], [0, 1, 2], rule)
    rows(rt, "R", [[a, b] for a in [0, 1] for b in [0, 1]], list(range(4)), rule)


def _project(state, context):
    return state // 2 if context == 0 else state % 2 if context == 1 else state


def encounter(instance):
    rt, p = instance.runtime, instance.parameters
    game, input_value = rt.uniform("game", 2), rt.uniform("input", 2)
    context = rt.uniform("context", 3)
    state = 2 * game + input_value
    executive = rt.decide("ES", context, context)
    active = executive.option
    choices, signals = [executive], {}
    for sender, property_value in [("A", game), ("B", input_value)]:
        if sender in active:
            observation = property_value if p["sender_observation"] == "assigned_property" else state
            decision = rt.decide(sender, observation, observation)
            choices.append(decision)
            signals[sender] = decision.option
    pair = [
        signals["A"] if "A" in signals else rt.uniform("missing_A", 2),
        signals["B"] if "B" in signals else rt.uniform("missing_B", 2),
    ]
    receiver = rt.decide("R", pair, {"signals": signals})
    interpretation = rt.decide("ER", active, active)
    choices.extend([receiver, interpretation])
    context_correct = (
        interpretation.option == context
        and _project(receiver.option, context) == _project(state, context)
    )
    relevance_correct = (
        active == ["A", "B", "AB"][context]
        if p["relevance"] == "assigned_roles"
        else (active == "AB") == (context == 2)
    )
    success = context_correct and relevance_correct
    decoded_game, decoded_input = divmod(receiver.option, 2)
    output = 1 if decoded_game == 0 else 1 - decoded_input
    target = 1 if game == 0 else 1 - input_value
    outcome = Outcome(
        success,
        float(success),
        terms=len(signals),
        details={
            "state": ["taut" if game == 0 else "neg", input_value],
            "context": ["game", "input", "both"][context],
            "active_senders": active,
            "decoded_state": ["taut" if decoded_game == 0 else "neg", decoded_input],
            "interpretation": ["game", "input", "both"][interpretation.option],
            "context_correct": context_correct,
            "relevance_correct": relevance_correct,
            "boolean_output": output,
            "boolean_correct": output == target,
        },
    )
    # A missing sender never drew a ball. All other involved agents reinforce
    # their chosen option on success, including both executive decisions.
    rt.credit(choices, outcome)
    return outcome


def validate(profile):
    p = resolve_parameters(profile.model, dict(profile.parameters))
    case = p.get("case")
    if case not in CASES:
        raise ValueError("LaCroix composition requires an explicit supported case")
    if set(p) - {"case", "sender_observation", "relevance", "window", "owners"}:
        raise ValueError("Unknown LaCroix composition parameters")
    if (p["sender_observation"], p["relevance"]) != CASES[case]:
        raise ValueError("Observation and relevance must match the declared LaCroix case")
    if profile.learner.get("rule") != "R1" or profile.learner.get("error", 0) != 0:
        raise ValueError("LaCroix composition uses proportional R1 without exploration error")
    if set(profile.learner) - {"rule", "error"}:
        raise ValueError("LaCroix composition has no additional learning parameters")
    if case == "full_observation_fixed_roles" and profile.kind not in {
        "demonstration", "deliberate_variant"
    }:
        raise ValueError("The full-observation fixed-role control is a deliberate variant")
