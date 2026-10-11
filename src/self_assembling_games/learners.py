"""Concrete adaptive operations supporting conditional dispositions."""

from __future__ import annotations

import numpy as np
from .core import Feedback, Runtime, Table


DEFAULTS = {
    "R1": {"error": 0.0},
    "R2": {"error": 0.0, "floor": 1.0, "reward": 1, "punishment": 1},
    "SIGNED": {"error": 0.0, "floor": 1.0},
    "R3": {"error": 0.0, "floor": 1.0, "reward": 2, "punishment": 4},
    "R4": {"error": 0.0},
    "R6": {"error": 0.0, "reward": 1},
    "WSLS": {"error": 0.0},
    "WSLR": {"error": 0.0},
    "STRATEGY_WSLR": {"error": 0.0},
    "BOUNDED": {"error": 0.0},
    "DISCOUNT": {"error": 0.0, "floor": 1.0},
}


def resolved_parameters(configuration):
    return {
        "error": 0.0,
        **DEFAULTS.get(configuration.get("rule"), {}),
        **configuration,
    }


def _apply_r1(runtime: Runtime, table: Table, feedback: Feedback) -> None:
    index = table.index(feedback.address.option)
    reward = feedback.payoff
    if reward < 0:
        raise ValueError("R1 requires nonnegative reinforcement")
    table.weights[index] += reward


def _apply_r2(runtime: Runtime, table: Table, feedback: Feedback) -> None:
    config = resolved_parameters(table.learner)
    index = table.index(feedback.address.option)
    floor = config.get("floor", 1.0)
    delta = (
        config.get("reward", 1) if feedback.success else -config.get("punishment", 1)
    )
    table.weights[index] = max(floor, table.weights[index] + delta)


def _apply_signed(runtime: Runtime, table: Table, feedback: Feedback) -> None:
    config = resolved_parameters(table.learner)
    index = table.index(feedback.address.option)
    reward = feedback.payoff
    floor = config.get("floor", 1.0)
    table.weights[index] = max(floor, table.weights[index] + reward)


def _apply_r3(runtime: Runtime, table: Table, feedback: Feedback) -> None:
    config = resolved_parameters(table.learner)
    index = table.index(feedback.address.option)
    floor = config.get("floor", 1.0)
    length = config["phase_length"]
    if (runtime.clocks["play"] // length) % 2 == 0:
        if feedback.success:
            table.weights[index] += config.get("reward", 2)
    elif not feedback.success:
        table.weights[index] = max(
            floor, table.weights[index] - config.get("punishment", 4)
        )


def _apply_r4(runtime: Runtime, table: Table, feedback: Feedback) -> None:
    config = resolved_parameters(table.learner)
    index = table.index(feedback.address.option)
    reward = feedback.payoff
    reference = table.reference
    experience = reward - reference
    epsilon = config["epsilon"]
    increments = np.full(len(table.options), experience * epsilon)
    increments[index] = experience * (1 - epsilon)
    table.weights[:] = np.maximum(
        config["floor"], (1 - config["forgetting"]) * table.weights + increments
    )
    learning_rate = config["w_plus"] if reward >= reference else config["w_minus"]
    table.reference = (1 - learning_rate) * reference + learning_rate * reward


def _apply_r6(runtime: Runtime, table: Table, feedback: Feedback) -> None:
    config = resolved_parameters(table.learner)
    rule = config["rule"]
    index = table.index(feedback.address.option)
    if feedback.success:
        if rule == "R6":
            table.weights[index] += config.get("reward", 1)
    elif rule == "WSLS":
        if len(table.options) != 2:
            raise ValueError("WSLS is defined here for two options")
        table.current = table.options[1 - index]
    else:
        probabilities = (
            table.probabilities(response=False)
            if rule == "R6"
            else np.full(len(table.options), 1 / len(table.options))
        )
        table.current = table.options[
            runtime.chance(
                "resample:" + runtime.table_id(table.module, table.context),
                probabilities,
            )
        ]
    if not feedback.success:
        runtime.emit("assignment", table=feedback.address.table, option=table.current)


def _apply_bounded(runtime: Runtime, table: Table, feedback: Feedback) -> None:
    config = resolved_parameters(table.learner)
    index = table.index(feedback.address.option)
    reward = feedback.payoff
    if reward < 0:
        raise ValueError("Bounded positive reinforcement needs nonnegative payoff")
    table.weights[index] = min(config["maximum"], table.weights[index] + reward)


def _apply_discount(runtime: Runtime, table: Table, feedback: Feedback) -> None:
    config = resolved_parameters(table.learner)
    index = table.index(feedback.address.option)
    reward = feedback.payoff
    floor = config.get("floor", 1.0)
    table.weights[:] = np.maximum(floor, table.weights * config["retention"])
    table.weights[index] += reward


UPDATES = {
    "R1": _apply_r1,
    "R2": _apply_r2,
    "SIGNED": _apply_signed,
    "R3": _apply_r3,
    "R4": _apply_r4,
    "R6": _apply_r6,
    "WSLS": _apply_r6,
    "WSLR": _apply_r6,
    "BOUNDED": _apply_bounded,
    "DISCOUNT": _apply_discount,
}


def update(runtime: Runtime, table: Table, feedback: Feedback) -> None:
    try:
        operation = UPDATES[table.learner["rule"]]
    except KeyError:
        raise ValueError(f"Unsupported learner rule {table.learner['rule']}") from None
    operation(runtime, table, feedback)
