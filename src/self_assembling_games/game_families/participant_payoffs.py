"""Explicit participant utilities and addressed feedback for fixed SIG/SCOD games."""

from __future__ import annotations

import math

import numpy as np

from ..core import Feedback, Outcome


FIELDS = {"owner_payoffs", "feedback_payoffs", "sender_signal_costs"}


def has_separate_payoffs(parameters):
    return bool(FIELDS & parameters.keys())


def validate_separate_payoffs(profile, states, actions):
    """Validate both return contracts before a game can issue any feedback."""
    from ..definitions import game_definition

    p = profile.parameters
    if not has_separate_payoffs(p):
        return
    if profile.model not in {"SIG", "SCOD"}:
        raise ValueError("Separate participant payoffs are supported for SIG and SCOD")
    if not {"owner_payoffs", "feedback_payoffs", "success_matrix"} <= p.keys():
        raise ValueError(
            "Separate returns require owner_payoffs, feedback_payoffs and success_matrix"
        )
    if profile.learner["rule"] not in {"R1", "SIGNED"}:
        raise ValueError(
            "Recipient-local payoff profiles support R1 or SIGNED learning"
        )
    declaration = game_definition(profile.model).initial_organisation(p)
    modules = declaration["owners"]
    for name, expected in (
        ("owner_payoffs", set(modules.values())),
        ("feedback_payoffs", set(modules)),
    ):
        values = p[name]
        if not isinstance(values, dict) or set(values) != expected:
            raise ValueError(f"{name} must cover exactly its declared recipients")
        for matrix in values.values():
            array = np.asarray(matrix, dtype=float)
            if array.shape != (states, actions) or not np.isfinite(array).all():
                raise ValueError(f"{name} requires finite states × actions matrices")
    success = np.asarray(p["success_matrix"])
    if success.shape != (states, actions) or not np.isin(success, [False, True]).all():
        raise ValueError("success_matrix must be Boolean states × actions")
    costs = None
    if "sender_signal_costs" in p:
        if profile.model != "SIG":
            raise ValueError("sender_signal_costs belongs to the SIG cost comparator")
        costs = np.asarray(p["sender_signal_costs"], dtype=float)
        if (
            costs.shape != (states, p["signals"])
            or not np.isfinite(costs).all()
            or np.any(costs < 0)
        ):
            raise ValueError(
                "sender_signal_costs requires nonnegative states × signals"
            )
    if profile.learner["rule"] == "R1":
        for module, matrix in p["feedback_payoffs"].items():
            values = np.asarray(matrix, dtype=float)
            if module == "sender" and costs is not None:
                values = values - costs.max(axis=1)[:, None]
            if np.any(values < 0):
                raise ValueError(
                    "R1 requires nonnegative recipient-local feedback after signal costs"
                )


def explicit_outcome(
    runtime,
    parameters,
    choices,
    state,
    action,
    success,
    task_reward,
    *,
    sender_signal=None,
):
    """Compute utilities once and credit each participating occurrence explicitly.

    Owner utilities and module feedback have independent tables. The SIG cost
    comparator explicitly charges its sender's chosen signal to both contracts.
    """
    owner_returns = {
        owner: float(matrix[state][action])
        for owner, matrix in parameters["owner_payoffs"].items()
    }
    signal_cost = (
        float(parameters["sender_signal_costs"][state][sender_signal])
        if "sender_signal_costs" in parameters
        else 0.0
    )
    if "sender_signal_costs" in parameters:
        owner_returns[runtime.owners["sender"]] -= signal_cost
    outcome = Outcome(success, task_reward, owner_returns=owner_returns)
    outcome.returns_for(runtime.owners.values())
    feedback = [
        Feedback(
            decision.address,
            success,
            float(parameters["feedback_payoffs"][decision.module][state][action])
            - (signal_cost if decision.module == "sender" else 0.0),
            (decision.occurrence,),
        )
        for decision in choices
    ]
    if any(not math.isfinite(value.payoff) for value in feedback):
        raise ValueError("Recipient-local feedback must be finite after signal costs")
    for value in feedback:
        runtime.feedback(value)
    return outcome
