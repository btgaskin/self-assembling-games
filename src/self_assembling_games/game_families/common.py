"""Table construction, template installation and shared concrete operations."""

from __future__ import annotations

import copy

from typing import TYPE_CHECKING, Any
from ..core import Decision, Runtime, Table

if TYPE_CHECKING:
    from ..model import ModelInstance


def rows(
    rt: Runtime,
    module: str,
    contexts: list[Any],
    options: list[Any],
    learner: dict[str, Any],
    *,
    owner: str | None = None,
) -> None:
    for context in contexts:
        rt.add_table(module, context, options, learner, owner=owner)


def fixed(rt: Runtime, module: str, context: Any, option: Any) -> None:
    t = rt.tables[rt.table_id(module, context)]
    t.current = option
    t.frozen = True


def nand_rows(rt: Runtime, prefix: str, learner: dict[str, Any]) -> None:
    for module in ["A", "B"]:
        rows(rt, prefix + module, [0, 1], [0, 1], learner)
    rows(rt, prefix + "R", [[a, b] for a in [0, 1] for b in [0, 1]], [0, 1], learner)


def install_artifact(
    model_instance: ModelInstance, name: str, prefix: str, expected: str
) -> None:
    from ..component_artifacts import load_component_snapshot, ordered_component_tables

    snapshot = load_component_snapshot(model_instance.parameters["templates"][name])
    if snapshot["profile"]["model"] != expected:
        raise ValueError(f"Template {name} must be a {expected} artifact")
    if expected == "NAND" and snapshot["profile"]["parameters"].get(
        "truth_table", [1, 1, 1, 0]
    ) != [1, 1, 1, 0]:
        raise ValueError(
            "A NAND template must have been trained for the NAND truth function"
        )
    if expected in {"ORD", "ADD"}:
        tables = snapshot["runtime"]["tables"]
        for record in tables.values():
            if record["module"] in {"A", "B"} and "mutator" in record["options"]:
                raise ValueError(
                    "Frozen vocabulary needs an explicit export_template mutator policy"
                )
        if snapshot["profile"]["parameters"]["n"] != (
            model_instance.parameters.get("n", 11) if expected == "ORD" else 6
        ):
            raise ValueError("Template stimulus domain differs from the adapter domain")
        import itertools

        vocab = {
            m: {
                option
                for record in tables.values()
                if record["module"] == m
                for option in record["options"]
            }
            for m in ["A", "B"]
        }
        if any(
            model_instance.runtime.table_id("R", [a, b]) not in tables
            for a, b in itertools.product(vocab["A"], vocab["B"])
        ):
            raise ValueError(
                "Frozen vocabulary lacks a receiver row for an available signal pair"
            )
    for old in ordered_component_tables(snapshot):
        record = copy.deepcopy(old)
        record["module"] = prefix + record["module"]
        record["frozen"] = True
        table = Table.from_dict(record)
        model_instance.runtime.tables[
            model_instance.runtime.table_id(table.module, table.context)
        ] = table
        model_instance.runtime.owners[table.module] = prefix.rstrip(":")
        model_instance.runtime.frozen_modules.add(table.module)
    model_instance.lineage[name] = {
        "path": str(model_instance.parameters["templates"][name]),
        "profile": snapshot["profile"],
        "metrics": snapshot["metrics"],
        "seed": snapshot["seed"],
        "play": snapshot["runtime"]["clocks"]["play"],
        "configuration_hash": snapshot["configuration_hash"],
    }
    component = snapshot.get("component_artifact")
    model_instance.lineage[name]["artifact_hash"] = (
        component["artifact_hash"] if component else sim_hash(snapshot)
    )
    model_instance.lineage[name]["export_transforms"] = (
        component["transformations"] if component else snapshot["lineage"].get("vocabulary_export")
    )
    if component:
        model_instance.lineage[name]["component"] = copy.deepcopy(component)
    if "checkpoint_origin" in snapshot:
        model_instance.lineage[name]["source_checkpoint"] = copy.deepcopy(snapshot["checkpoint_origin"])


def sim_hash(snapshot: dict[str, Any]) -> str:
    from ..simulation import digest

    return digest(snapshot)


def install_nand(model_instance: ModelInstance, prefix: str, name: str) -> None:
    if model_instance.parameters["template_mode"] == "artifact":
        install_artifact(model_instance, name, prefix, "NAND")
        return
    rt = model_instance.runtime
    nand_rows(rt, prefix, {"rule": "R1"})
    for x in [0, 1]:
        for module in ["A", "B"]:
            fixed(rt, prefix + module, x, x)
    for a in [0, 1]:
        for b in [0, 1]:
            fixed(rt, prefix + "R", [a, b], 1 - a * b)
    model_instance.lineage[name] = {
        "kind": "constructed_perfect_fixture",
        "function": "NAND",
    }


def nand_call(rt: Runtime, prefix: str, a: int, b: int) -> tuple[int, list[Decision]]:
    da = rt.decide(prefix + "A", a, a)
    db = rt.decide(prefix + "B", b, b)
    dr = rt.decide(prefix + "R", [da.option, db.option], [da.option, db.option])
    return dr.option, [da, db, dr]


def ordering_call(rt: Runtime, a: int, b: int) -> int:
    da = rt.decide("ordering:A", a, a)
    db = rt.decide("ordering:B", b, b)
    return rt.decide(
        "ordering:R", [da.option, db.option], [da.option, db.option]
    ).option


def relation(a: int, b: int) -> int:
    return 0 if a < b else 1 if a > b else 2


def forget_signal(rt: Runtime, config: dict[str, Any]) -> None:
    """An explicit type/ball forgetting variant with declared extinction semantics."""
    urns = [(identity, t) for identity, t in rt.tables.items() if t.module == "sender"]
    identity, table = urns[rt.uniform("forget_urn", len(urns))]
    eligible = [
        i
        for i, option in enumerate(table.options)
        if not config["protect_mutator"] or option != "mutator"
    ]
    if not eligible:
        return
    probabilities = (
        [1 / len(eligible)] * len(eligible)
        if config["mode"] == "type"
        else (table.weights[eligible] / table.weights[eligible].sum()).tolist()
    )
    index = eligible[rt.chance("forget_option", probabilities)]
    option = table.options[index]
    if table.weights[index] <= 1 and len(table.options) == 1:
        rt.emit("forget_skipped", reason="protect_empty_urn")
        return
    if table.weights[index] <= 1:
        rt.delete_option(identity, option)
        represented = any(option in t.options for _, t in urns)
        receiver = rt.table_id("R", option)
        if (
            not represented
            and config["cleanup"] == "remove_orphan_receiver"
            and receiver in rt.tables
        ):
            rt.remove(receiver)
    else:
        table.weights[index] -= 1
        table.validate()
    rt.emit("forget_signal", table=identity, option=option, mode=config["mode"])


def plans(plan: int) -> tuple[list[int], list[int]]:
    return [(plan // 4) // 2, (plan // 4) % 2], [(plan % 4) // 2, plan % 2]


def expected_signaling(left: int, right: int) -> float:
    ls, lr = plans(left)
    rs, rr = plans(right)
    return sum(int(rr[ls[x]] == x) + int(lr[rs[x]] == x) for x in [0, 1]) / 4


def np_maximum_floor(weights: Any, floor: float) -> Any:
    import numpy as np

    return np.maximum(weights, floor)
