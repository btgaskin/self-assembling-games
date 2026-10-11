"""Explicit binding adapters for pretrained frozen conditional components."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .core import key


def import_frozen_decisions(
    path: str | Path,
    bindings: dict[str, str],
    *,
    option_indices: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    """Copy learned policies using destination-address -> source-address bindings.

    Option adapters list source option indices in destination order. They are
    complete permutations. Joint mixtures over plans require their own matching
    model and are never silently replaced by independent conditional marginals.
    """
    from .simulation import read_snapshot, digest

    path = Path(path)
    if path.is_dir():
        path = path / "checkpoint.json"
    snapshot = read_snapshot(path)
    if not bindings:
        raise ValueError("Frozen imports need explicit decision bindings")
    policies, options = {}, {}
    if snapshot.get("engine") == "adaptive":
        if snapshot["profile"]["method"] != "reinforcement.conditional.R1":
            raise ValueError(
                "Frozen decision import requires conditional policies; complete-plan correlations need an explicit model"
            )
        for row in snapshot["game"]["decisions"]:
            address = row["role"] + "/" + row["context"]
            options[address] = row["options"]
        for address, w in snapshot["weights"].items():
            weights = np.asarray(w, dtype=float)
            policies[address] = weights / weights.sum()
        policies.update(
            {k: np.asarray(v) for k, v in snapshot["game"]["fixed_decisions"].items()}
        )
        clock = snapshot["plays"]
    else:
        for address, table in snapshot["runtime"]["tables"].items():
            weights = np.asarray(table["weights"], dtype=float)
            p = weights / weights.sum()
            if table["current"] is not None:
                p = np.array(
                    [float(key(o) == key(table["current"])) for o in table["options"]]
                )
            error = table["learner"].get("error", 0)
            policies[address] = (1 - error) * p + error / len(p)
            options[address] = table["options"]
        clock = snapshot["runtime"]["clocks"]["play"]
    option_indices = option_indices or {}
    if option_indices.keys() - bindings.keys():
        raise ValueError("Option adapters refer to an unbound destination")
    result = {}
    for destination, source in bindings.items():
        if source not in policies:
            raise ValueError(f"Unknown source policy {source}")
        p = policies[source]
        if not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1):
            raise ValueError("Invalid pretrained conditional policy")
        order = option_indices.get(destination, list(range(len(p))))
        if sorted(order) != list(range(len(p))):
            raise ValueError("Option adapters must be complete permutations")
        result[destination] = p[order].tolist()
    return {
        "fixed_decisions": result,
        "lineage": {
            "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_configuration_hash": snapshot["configuration_hash"],
            "source_profile": copy.deepcopy(snapshot["profile"]),
            "source_metadata": copy.deepcopy(snapshot["metadata"]),
            "seed": snapshot["seed"],
            "training_plays": clock,
            "bindings": copy.deepcopy(bindings),
            "source_options": {s: options[s] for s in bindings.values()},
            "option_indices": copy.deepcopy(option_indices),
            "policy_hash": digest(result),
            "transform": "freeze_conditional_responses",
        },
    }
