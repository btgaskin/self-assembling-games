"""Portable, immutable frozen components and checkpoint-format adapters.

Artifacts retain conditional dispositions, including current assignments and error
rates. They do not replace a correlated distribution over complete plans by its
marginals. Source checkpoints are evidence; importing an artifact needs no source
checkpoint or source executable.
"""

from __future__ import annotations

import copy
import itertools
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .core import Table, key
from .simulation import digest

ARTIFACT_KIND = "self_assembling_games.component"
ARTIFACT_SCHEMA = 1


def _table_id(module, context):
    return module + "/" + key(context)


def _read(value):
    if isinstance(value, (str, Path)):
        path = Path(value)
        if path.is_dir():
            path = path / ("component.json" if (path / "component.json").is_file() else "checkpoint.json")
        return json.loads(path.read_text())
    if isinstance(value, dict):
        return copy.deepcopy(value)
    if hasattr(value, "snapshot"):
        return copy.deepcopy(value.snapshot())
    raise TypeError("Expected a System, simulation, checkpoint mapping or path")


def _unwrap_checkpoint(value):
    """Validate either checkpoint envelope without requiring its source code."""
    if value.get("schema_version") == 2:
        payload = {k: v for k, v in value.items() if k != "checkpoint_hash"}
        if (
            digest(payload) != value.get("checkpoint_hash")
            or digest(value["model"]) != value.get("model_hash")
        ):
            raise ValueError("Checkpoint integrity mismatch")
        snapshot = value["state"]
    elif value.get("schema_version") == 1:
        snapshot = value
    else:
        raise ValueError("Unsupported component checkpoint schema")
    if snapshot.get("schema_version") != 1:
        raise ValueError("Unsupported component state schema")
    if digest(snapshot["profile"]) != snapshot.get("configuration_hash"):
        raise ValueError("Checkpoint configuration hash mismatch")
    declaration = snapshot.get("model_specification")
    if declaration is not None and digest(declaration) != snapshot.get(
        "model_specification_hash"
    ):
        raise ValueError("Checkpoint model specification hash mismatch")
    result = copy.deepcopy(snapshot)
    if value.get("schema_version") == 2:
        result["checkpoint_origin"] = {
            "checkpoint_hash": value["checkpoint_hash"],
            "model_hash": value["model_hash"],
            "source": copy.deepcopy(value["model"].get("source", {})),
        }
    return result


def load_component_snapshot(value) -> dict[str, Any]:
    """Read legacy/System checkpoints or an integrity-checked component artifact.

    The returned mapping is owned by the caller. Explicit table order survives
    JSON's sorted object keys in every supported envelope.
    """
    document = _read(value)
    if document.get("kind") != ARTIFACT_KIND:
        return _unwrap_checkpoint(document)
    if document.get("schema_version") != ARTIFACT_SCHEMA:
        raise ValueError("Unsupported component artifact schema")
    payload = {k: v for k, v in document.items() if k != "artifact_hash"}
    if digest(payload) != document.get("artifact_hash"):
        raise ValueError("Component artifact integrity mismatch")
    policy = document["policy"]
    if policy.get("representation") != "conditional_tables":
        raise ValueError("Component requires conditional table policies")
    _validate_interface(document["interface"], policy["tables"], policy["table_order"])
    if any(not row["frozen"] for row in policy["tables"].values()):
        raise ValueError("Component artifacts must contain frozen tables")
    source = document["source"]
    snapshot = {
        "schema_version": 1,
        "profile": document["profile"],
        "configuration_hash": digest(document["profile"]),
        "seed": source["seed"],
        "metadata": source["metadata"],
        "metrics": document["training_metrics"],
        "lineage": source["lineage"],
        "runtime": {
            "tables": policy["tables"],
            "table_order": policy["table_order"],
            "clocks": {"play": source["position"]},
        },
        "component_artifact": {
            "artifact_hash": document["artifact_hash"],
            "source": source,
            "interface": document["interface"],
            "transformations": document["transformations"],
            "evaluation": document["evaluation"],
        },
    }
    return copy.deepcopy(snapshot)


def ordered_component_tables(snapshot):
    tables = snapshot["runtime"]["tables"]
    order = snapshot["runtime"].get("table_order", list(tables))
    if len(order) != len(set(order)) or set(order) != set(tables):
        raise ValueError("Component table order must contain every table once")
    return [tables[address] for address in order]


def _validate_interface(interface, tables, order):
    if len(order) != len(set(order)) or set(order) != set(tables):
        raise ValueError("Component table order must contain every table once")
    family, n = interface["family"], interface["input_size"]
    if family not in {"NAND", "ORD", "ADD"} or type(n) is not int or n < 1:
        raise ValueError("Unsupported component interface")
    if family == "NAND" and n != 2:
        raise ValueError("NAND components require binary inputs")
    if family == "NAND" and (
        len(interface.get("truth_table", [])) != 4
        or any(type(v) is not int or v not in [0, 1] for v in interface["truth_table"])
    ):
        raise ValueError("NAND components require a four-entry binary truth table")
    if family == "ADD" and n != 6:
        raise ValueError("ADD components require inputs zero through five")
    by_module = {m: {} for m in ["A", "B", "R"]}
    for address in order:
        row = tables[address]
        table = Table.from_dict(row)
        if table.module not in by_module or address != _table_id(
            table.module, table.context
        ):
            raise ValueError("Component table does not match its interface address")
        error = table.learner.get("error", 0)
        if not isinstance(error, (int, float)) or not 0 <= error <= 1:
            raise ValueError("Component error probability must lie in [0, 1]")
        by_module[table.module][key(table.context)] = table
    signals = {}
    for module in ["A", "B"]:
        if set(by_module[module]) != {key(i) for i in range(n)}:
            raise ValueError("Component sender contexts differ from its input domain")
        options = [o for t in by_module[module].values() for o in t.options]
        if any(o == "mutator" for o in options):
            raise ValueError("Frozen vocabulary needs an explicit mutator transform")
        signals[module] = list({key(o): o for o in options}.values())
        if family == "NAND" and any(type(o) is not int or o not in [0, 1] for o in options):
            raise ValueError("NAND components require binary sender messages")
    output_count = 2 if family == "NAND" else 3 if family == "ORD" else 2 * n - 1
    for a, b in itertools.product(signals["A"], signals["B"]):
        table = by_module["R"].get(key([a, b]))
        if table is None:
            raise ValueError("Frozen vocabulary lacks a receiver row for an available signal pair")
        if table.options != list(range(output_count)):
            raise ValueError("Component receiver options differ from its output interface")
    if family == "NAND" and set(by_module["R"]) != {
        key([a, b]) for a, b in itertools.product(range(2), repeat=2)
    }:
        raise ValueError("NAND components need all four receiver rows")


def _evaluation(interface, tables):
    """Exact detached evaluation of the exported policies, including transforms."""
    n, family = interface["input_size"], interface["family"]
    rows = {m: {} for m in ["A", "B", "R"]}
    for record in tables.values():
        table = Table.from_dict(record)
        rows[table.module][key(table.context)] = table
    signals = {
        m: list({key(o): o for t in rows[m].values() for o in t.options}.values())
        for m in ["A", "B"]
    }
    sender = {}
    for m in ["A", "B"]:
        indices = {key(o): i for i, o in enumerate(signals[m])}
        matrix = np.zeros((n, len(indices)))
        for i in range(n):
            table = rows[m][key(i)]
            for option, p in zip(table.options, table.conditional_disposition()):
                matrix[i, indices[key(option)]] = p
        sender[m] = matrix
    count = 2 if family == "NAND" else 3 if family == "ORD" else 2 * n - 1
    conditional = np.empty((len(signals["A"]), len(signals["B"]), count))
    for i, a in enumerate(signals["A"]):
        for j, b in enumerate(signals["B"]):
            conditional[i, j] = rows["R"][key([a, b])].conditional_disposition()
    targets = np.array([
        [
            interface["truth_table"][a * 2 + b] if family == "NAND"
            else (0 if a < b else 1 if a > b else 2) if family == "ORD"
            else a + b
            for b in range(n)
        ]
        for a in range(n)
    ])
    success = 0.0
    for output in range(count):
        probabilities = sender["A"] @ conditional[:, :, output] @ sender["B"].T
        success += float(probabilities[targets == output].sum()) / (n * n)
    return {
        "protocol": "exact_frozen_component",
        "input_distribution": "uniform_cartesian",
        "input_pairs": n * n,
        "success_probability": success,
        "expected_net_payoff": success,
    }


def export_component(system_or_checkpoint, path, specification=None) -> Path:
    """Export a frozen NAND/ORD/ADD policy without changing its source state.

    ``specification`` may declare ``interface`` (NAND/ORD/ADD), ``modules``
    (interface module A/B/R -> source module), and the explicit vocabulary
    transforms ``suppress_mutator`` and ``complete_pair_rows``. A module selection
    must provide all three interface modules. Existing files are never replaced.
    """
    document = _read(system_or_checkpoint)
    snapshot = _unwrap_checkpoint(document)
    if snapshot.get("engine") == "adaptive" or "runtime" not in snapshot:
        raise ValueError("Component export requires conditional tables; correlated complete plans need their own component interface")
    spec = copy.deepcopy(specification or {})
    allowed = {"interface", "modules", "suppress_mutator", "complete_pair_rows", "max_pair_rows"}
    if not isinstance(spec, dict) or set(spec) - allowed:
        raise ValueError("Unknown component export specification fields")
    model = snapshot["profile"]["model"]
    family = spec.get("interface", model)
    if family not in {"NAND", "ORD", "ADD"}:
        raise ValueError("Component export needs an explicit NAND, ORD or ADD interface")
    modules = spec.get("modules", {m: m for m in ["A", "B", "R"]})
    if (
        not isinstance(modules, dict)
        or set(modules) != {"A", "B", "R"}
        or any(not isinstance(m, str) for m in modules.values())
        or len(set(modules.values())) != 3
    ):
        raise ValueError("Component modules must bind distinct source modules to A, B and R")
    if model != family and "modules" not in spec:
        raise ValueError("Exporting a subcomponent requires explicit module bindings")
    for flag in ["suppress_mutator", "complete_pair_rows"]:
        if flag in spec and type(spec[flag]) is not bool:
            raise ValueError("Component transforms must be explicit booleans")
    budget = spec.get("max_pair_rows", 10_000)
    if type(budget) is not int or budget <= 0:
        raise ValueError("max_pair_rows must be a positive integer")
    n = 2 if family == "NAND" else snapshot["profile"]["parameters"].get("n")
    if type(n) is not int:
        raise ValueError("Component source must declare its input domain")
    interface = {"family": family, "input_size": n, "modules": ["A", "B", "R"]}
    if family == "NAND":
        interface["truth_table"] = (
            snapshot["profile"]["parameters"].get("truth_table", [1, 1, 1, 0])
            if model == "NAND" else [1, 1, 1, 0]
        )
    tables, order = {}, []
    rename = {source: destination for destination, source in modules.items()}
    transformations = [{"operation": "freeze_conditional_tables"}]
    if "modules" in spec:
        transformations.append({"operation": "select_modules", "bindings": modules})
    removed = 0
    for old in ordered_component_tables(snapshot):
        if old["module"] not in rename:
            continue
        record = copy.deepcopy(old)
        record["module"] = rename[record["module"]]
        if record["module"] in {"A", "B"} and "mutator" in record["options"]:
            if not spec.get("suppress_mutator", False):
                raise ValueError("Frozen vocabulary needs an explicit mutator transform")
            keep = [i for i, o in enumerate(record["options"]) if o != "mutator"]
            if not keep or record.get("current") == "mutator":
                raise ValueError("Cannot export an empty vocabulary or a current mutator assignment")
            removed += len(record["options"]) - len(keep)
            record["options"] = [record["options"][i] for i in keep]
            record["weights"] = [record["weights"][i] for i in keep]
        record["frozen"] = True
        address = _table_id(record["module"], record["context"])
        tables[address] = record
        order.append(address)
    if spec.get("suppress_mutator", False):
        transformations.append({"operation": "suppress_mutator", "removed_options": removed})
    signals = {
        m: list({key(o): o for row in tables.values() if row["module"] == m for o in row["options"]}.values())
        for m in ["A", "B"]
    }
    if len(signals["A"]) * len(signals["B"]) > budget:
        raise ValueError("Vocabulary export exceeds max_pair_rows")
    added = 0
    count = 2 if family == "NAND" else 3 if family == "ORD" else 2 * n - 1
    for a, b in itertools.product(signals["A"], signals["B"]):
        address = _table_id("R", [a, b])
        if address not in tables:
            if not spec.get("complete_pair_rows", False):
                raise ValueError("Unseen pair rows remain in the frozen vocabulary")
            tables[address] = Table("R", [a, b], list(range(count)), np.ones(count), {"rule": "R1"}, frozen=True).to_dict()
            order.append(address)
            added += 1
    if spec.get("complete_pair_rows", False):
        transformations.append({"operation": "complete_pair_rows", "added_rows": added, "initial_weight": 1.0, "max_pair_rows": budget})
    _validate_interface(interface, tables, order)
    profile = copy.deepcopy(snapshot["profile"])
    profile["model"] = family
    if model != family:
        profile["parameters"] = {"truth_table": interface["truth_table"]} if family == "NAND" else {"n": n}
    source_model = document.get("model", snapshot.get("model_specification"))
    source = {
        "snapshot_hash": digest(document),
        "checkpoint_hash": document.get("checkpoint_hash", digest(document)),
        "model_hash": document.get("model_hash", snapshot.get("model_specification_hash", digest(snapshot["profile"]))),
        "model": copy.deepcopy(source_model),
        "declaration": copy.deepcopy((source_model or {}).get("source", {})),
        "profile": copy.deepcopy(snapshot["profile"]),
        "configuration_hash": snapshot["configuration_hash"],
        "seed": snapshot["seed"],
        "clock": (source_model or {}).get("execution", {}).get("clock", "encounters"),
        "position": snapshot["runtime"]["clocks"]["play"],
        "metadata": copy.deepcopy(snapshot["metadata"]),
        "provenance": copy.deepcopy(document.get("provenance", snapshot["metadata"])),
        "provider": copy.deepcopy(document.get("provider")),
        "lineage": copy.deepcopy(snapshot.get("lineage", {})),
    }
    artifact = {
        "kind": ARTIFACT_KIND,
        "schema_version": ARTIFACT_SCHEMA,
        "interface": interface,
        "profile": profile,
        "policy": {"representation": "conditional_tables", "tables": tables, "table_order": order},
        "source": source,
        "transformations": transformations,
        "training_metrics": copy.deepcopy(snapshot.get("metrics", {})),
        "evaluation": _evaluation(interface, tables),
    }
    artifact["artifact_hash"] = digest(artifact)
    target = Path(path)
    if target.suffix != ".json":
        target = target / "component.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    # Linking a complete neighbouring file publishes atomically and exclusively.
    # A concurrent exporter can never replace a different artifact at this path.
    with tempfile.NamedTemporaryFile(mode="w", dir=target.parent, delete=False) as f:
        temporary = Path(f.name)
        try:
            json.dump(artifact, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            try:
                os.link(temporary, target)
            except FileExistsError:
                if json.loads(target.read_text()) != artifact:
                    raise FileExistsError(f"Component artifact already exists: {target}") from None
        finally:
            temporary.unlink(missing_ok=True)
    return target
