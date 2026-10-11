"""Source labels compose updates, persistence and phase-controlled behavior."""

from __future__ import annotations

import numpy as np

from .core import Feedback, Runtime, Table


def update(runtime: Runtime, table: Table, feedback: Feedback) -> None:
    config = table.learner
    rule = config["rule"]
    index = table.index(feedback.address.option)
    reward = feedback.payoff
    floor = config.get("floor", 1.0)
    if rule == "R1":
        if reward < 0:
            raise ValueError("R1 requires nonnegative reinforcement")
        table.weights[index] += reward
    elif rule == "R2":
        delta = (
            config.get("reward", 1)
            if feedback.success
            else -config.get("punishment", 1)
        )
        table.weights[index] = max(floor, table.weights[index] + delta)
    elif rule == "SIGNED":
        table.weights[index] = max(floor, table.weights[index] + reward)
    elif rule == "R3":
        length = config["phase_length"]
        if (runtime.clocks["play"] // length) % 2 == 0:
            if feedback.success:
                table.weights[index] += config.get("reward", 2)
        elif not feedback.success:
            table.weights[index] = max(
                floor, table.weights[index] - config.get("punishment", 4)
            )
    elif rule == "R4":
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
    elif rule in {"R6", "WSLS", "WSLR"}:
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
            runtime.emit(
                "assignment", table=feedback.address.table, option=table.current
            )
    elif rule == "BOUNDED":
        if reward < 0:
            raise ValueError("Bounded positive reinforcement needs nonnegative payoff")
        table.weights[index] = min(config["maximum"], table.weights[index] + reward)
    elif rule == "DISCOUNT":
        table.weights[:] = np.maximum(floor, table.weights * config["retention"])
        table.weights[index] += reward
    else:
        raise ValueError(f"Unsupported learner rule {rule}")
