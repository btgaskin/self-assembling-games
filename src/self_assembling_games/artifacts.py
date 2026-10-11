"""Full event replay and explicitly transformed frozen vocabulary exports."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from .core import Table, key
from .simulation import Simulation, atomic_json, digest


def replay_trace(path: str | Path) -> Simulation:
    events = [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]
    if not events or events[0].get("event") != "trace_start":
        raise ValueError("A full trace starts with a checkpoint")
    simulation = Simulation.restore(events[0]["snapshot"])
    position = 1
    while position < len(events):
        if events[position].get("event") == "trace_start":
            if simulation.snapshot() != events[position]["snapshot"]:
                raise ValueError(
                    "Appended trace checkpoint does not match replay state"
                )
            position += 1
            continue
        end = position
        while end < len(events) and events[end]["event"] != "outcome":
            end += 1
        if end == len(events):
            raise ValueError("Incomplete trace: terminal outcome absent")
        expected = events[position : end + 1]
        simulation.runtime.replay_chances = [
            e for e in expected if e["event"] == "chance"
        ]
        simulation.runtime.record = True
        simulation.step()
        if simulation.runtime.events != expected:
            raise ValueError(f"Event replay diverged at play {expected[0]['play']}")
        if simulation.runtime.replay_chances:
            raise ValueError("Unused chance outcomes during replay")
        position = end + 1
    simulation.runtime.replay_chances = None
    simulation.runtime.record = False
    return simulation


def export_template(
    simulation: Simulation,
    path: str | Path,
    *,
    suppress_mutator: bool,
    complete_pair_rows: bool,
    max_pair_rows: int = 10_000,
) -> Path:
    """Export a fixed vocabulary under explicit, recorded transforms.

    Suppressing invention and filling unseen receiver pairs are deliberate transforms.
    Their source-canonical interpretation remains a source question.
    """
    model = simulation.profile.model
    if model not in {"ORD", "ADD"}:
        raise ValueError("Vocabulary export requires ORD or ADD")
    snapshot = simulation.snapshot()
    data = snapshot["runtime"]["tables"]
    for record in data.values():
        if record["module"] in {"A", "B"} and suppress_mutator:
            keep = [
                i for i, option in enumerate(record["options"]) if option != "mutator"
            ]
            if not keep:
                raise ValueError("Cannot export an empty invented vocabulary")
            record["options"] = [record["options"][i] for i in keep]
            record["weights"] = [record["weights"][i] for i in keep]
            record["current"] = None
        record["frozen"] = True
    signals = {
        m: sorted(
            {
                option
                for record in data.values()
                if record["module"] == m
                for option in record["options"]
            },
            key=key,
        )
        for m in ["A", "B"]
    }
    if max_pair_rows <= 0 or len(signals["A"]) * len(signals["B"]) > max_pair_rows:
        raise ValueError(
            "Vocabulary export exceeds max_pair_rows; set an explicit larger budget"
        )
    count = 3 if model == "ORD" else 2 * simulation.profile.parameters["n"] - 1
    added = 0
    for a, b in itertools.product(signals["A"], signals["B"]):
        identity = simulation.runtime.table_id("R", [a, b])
        if identity not in data:
            if not complete_pair_rows:
                raise ValueError("Unseen pair rows remain in the frozen vocabulary")
            import numpy as np

            data[identity] = Table(
                "R",
                [a, b],
                list(range(count)),
                np.ones(count),
                {"rule": "R1"},
                frozen=True,
            ).to_dict()
            added += 1
    snapshot["lineage"]["vocabulary_export"] = {
        "kind": "deliberate_transform",
        "suppress_mutator": suppress_mutator,
        "complete_pair_rows": complete_pair_rows,
        "added_rows": added,
        "max_pair_rows": max_pair_rows,
        "original_snapshot_hash": digest(simulation.snapshot()),
    }
    snapshot["history"].append(
        {"export": "freeze_vocabulary", "play": snapshot["runtime"]["clocks"]["play"]}
    )
    path = Path(path)
    if path.suffix != ".json":
        path = path / "checkpoint.json"
    atomic_json(path, snapshot)
    return path
