"""Optional molecular realisation; resolving declarations does not import toyLIFE."""

from __future__ import annotations

import copy
import importlib
import math

from ..config import merge
from .apparatus import resolve_apparatus
from .coverage import coverage
from .tasks import resolve_task

CONTRACT = "games-canonical-cell-v1"
RUNTIME = "self_assembling_games.toylife"
BOOLEAN = {
    "nand": [1, 1, 1, 0],
    "and": [0, 0, 0, 1],
    "or": [0, 1, 1, 1],
    "xor": [0, 1, 1, 0],
}


def _keys(value, allowed, label):
    if not isinstance(value, dict) or value.keys() - set(allowed):
        raise ValueError(f"Unknown {label} fields")


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} requires an integer >= {minimum}")


def interaction(data):
    """Construct a pure task; custom factories declare fields and action labels."""
    spec = data["interaction"]
    if spec["provider"] == "toylife.tasks":
        from .tasks import TaskInteraction

        return TaskInteraction(resolve_task(spec["game"], spec["parameters"]))
    module, sep, name = spec["provider"].partition(":")
    if not sep:
        raise ValueError("Custom interactions require module:factory")
    factory = getattr(importlib.import_module(module), name)
    result = factory(copy.deepcopy(spec["parameters"]))
    if tuple(result.actors) != ("cell",):
        raise ValueError("This adapter requires one chemical owner, actor 'cell'")
    if not isinstance(result.fields, dict) or not isinstance(
        result.actions, (list, tuple)
    ):
        raise ValueError("Custom interactions declare fields and actions")
    return result


def resolve_model(identifier, overrides):
    if isinstance(identifier, str):
        if not identifier.startswith("toylife."):
            raise ValueError("Molecular presets start with toylife.")
        task = identifier[len("toylife.") :]
        parameters = {}
        if task.startswith("boolean."):
            suffix = task.split(".", 1)[1]
            if suffix not in BOOLEAN:
                raise ValueError(
                    "Unknown Boolean preset; supply truth_table for other functions"
                )
            parameters = {"truth_table": BOOLEAN[suffix]}
            task = "boolean"
        data = dict(
            id=identifier,
            runtime=RUNTIME,
            interaction={
                "provider": "toylife.tasks",
                "game": task,
                "parameters": parameters,
            },
            organisation={
                "organisms": 1,
                "compartments": 1,
                "owner": "cell",
                "selection_unit": "organism",
            },
            realisation={"provider": "toylife-canonical-v1", "parameters": {}},
            adaptation={"provider": "wright_fisher", "parameters": {}},
            execution={"clock": "generations", "budget": 2, "parameters": {}},
            source={},
            resources={},
            capabilities=[],
        )
    else:
        data = copy.deepcopy(identifier)
    data = merge(data, overrides)
    _keys(
        data,
        (
            "id",
            "runtime",
            "interaction",
            "organisation",
            "realisation",
            "adaptation",
            "execution",
            "source",
            "resources",
            "capabilities",
            "kind",
            "schema_version",
        ),
        "model",
    )
    if data["runtime"] != RUNTIME:
        raise ValueError("Molecular runtime identity mismatch")
    expected_org = {
        "organisms": 1,
        "compartments": 1,
        "owner": "cell",
        "selection_unit": "organism",
    }
    if data["organisation"] != expected_org:
        raise ValueError(
            "Canonical adapter currently supports one owned molecular pool"
        )
    for section in ("interaction", "realisation", "adaptation"):
        _keys(
            data[section],
            ("provider", "parameters", "game")
            if section == "interaction"
            else ("provider", "parameters"),
            section,
        )
        data[section].setdefault("parameters", {})
    if data["realisation"]["provider"] != "toylife-canonical-v1":
        raise ValueError("This driver implements toylife-canonical-v1 only")
    if data["adaptation"]["provider"] != "wright_fisher":
        raise ValueError("This driver uses organism-level Wright-Fisher evolution")
    r = data["realisation"]["parameters"]
    _keys(
        r,
        (
            "genes",
            "apparatus",
            "controller",
            "initial_genotypes",
            "resolved_apparatus",
            "reset_between_phases",
        ),
        "realisation parameters",
    )
    r.setdefault("genes", 2)
    r.setdefault("reset_between_phases", False)
    if type(r["reset_between_phases"]) is not bool:
        raise ValueError("reset_between_phases must be Boolean")
    _integer(r["genes"], "genes")
    r.setdefault("apparatus", {})
    r.setdefault("controller", {})
    _keys(
        r["controller"],
        (
            "initial_expression",
            "initial_proteins",
            "initial_dimers",
            "initial_metabolites",
            "initial_bound",
            "warmup",
            "capacity",
        ),
        "controller",
    )
    r["controller"] = merge(
        {"warmup": 0, "initial_expression": 0, "capacity": 4096}, r["controller"]
    )
    # Warmup is advanced by the evaluator so its work and pending state are counted.
    _integer(r["controller"]["warmup"], "warmup", 0)
    if r["controller"]["warmup"]:
        raise ValueError("Use an explicit initial exposure stage for nonzero warmup")
    a = data["adaptation"]["parameters"]
    _keys(
        a,
        ("population_size", "mutation_probability", "fitness_floor"),
        "adaptation parameters",
    )
    a = data["adaptation"]["parameters"] = merge(
        {"population_size": 4, "mutation_probability": 0.01, "fitness_floor": 1e-8}, a
    )
    _integer(a["population_size"], "population_size")
    if "initial_genotypes" in r:
        initial = r["initial_genotypes"]
        if (
            not isinstance(initial, list)
            or len(initial) not in (1, a["population_size"])
            or any(
                not isinstance(g, str)
                or len(g) != 20 * r["genes"]
                or set(g) - {"0", "1"}
                for g in initial
            )
        ):
            raise ValueError(
                "initial_genotypes requires one founder or a full population of 20 * genes binary strings"
            )
    for name in ("mutation_probability", "fitness_floor"):
        if (
            isinstance(a[name], bool)
            or not isinstance(a[name], (int, float))
            or not math.isfinite(a[name])
            or not 0 <= a[name] <= 1
        ):
            raise ValueError(f"Invalid {name}")
    if a["fitness_floor"] >= 1:
        raise ValueError("fitness_floor must be below one")
    _keys(data["execution"], ("clock", "budget", "parameters"), "execution")
    if data["execution"]["clock"] != "generations":
        raise ValueError("Population execution uses generations")
    _integer(data["execution"]["budget"], "budget", 0)
    e = data["execution"].setdefault("parameters", {})
    _keys(
        e,
        (
            "backend",
            "response_updates",
            "episodes",
            "trace_limit",
            "memory_budget_mib",
            "max_chemical_updates",
            "max_world_ticks",
            "max_episode_phases",
        ),
        "execution parameters",
    )
    e = data["execution"]["parameters"] = merge(
        {
            "backend": "cpu",
            "response_updates": 8,
            "episodes": 1,
            "trace_limit": 128,
            "memory_budget_mib": 1024,
            "max_episode_phases": 10000,
        },
        e,
    )
    if e["backend"] not in ("cpu", "metal", "cuda"):
        raise ValueError("Choose an explicit native backend")
    for name, value in e.items():
        if name != "backend":
            _integer(value, name, 0 if name == "trace_limit" else 1)
    obj = interaction(data)
    r["resolved_apparatus"] = resolve_apparatus(obj.fields, obj.actions, r["apparatus"])
    if data["interaction"]["provider"] == "toylife.tasks":
        data["interaction"]["parameters"] = obj.task["parameters"]
    sources = [
        item["id"]
        for item in coverage()
        if data["interaction"].get("game") in item["tasks"]
    ]
    source = {
        "kind": "deliberate_variant",
        "identifiers": sources,
        "assumptions": [
            "One inherited organism with a shared canonical molecular pool",
            "Fieldwise fixed exposures and structural action readouts",
            "Experience changes chemistry; scores guide organism selection",
            "Functional molecular roles are assessed after evolution",
        ],
        "questions": [],
        "contract": CONTRACT,
    }
    data["source"] = merge(source, data.get("source", {}))
    data["source"]["kind"] = "deliberate_variant"
    data["source"]["contract"] = CONTRACT
    data["capabilities"] = [
        "evaluate.molecular",
        "checkpoint.molecular",
        "evolution.wright_fisher",
    ]
    data.update(kind="model", schema_version=2)
    return data


def create_backend(model_dict, seed, protocol=None):
    from .runtime import MolecularBackend

    return MolecularBackend(model_dict, seed, protocol or {})


def restore_backend(model_dict, snapshot):
    from .runtime import MolecularBackend

    return MolecularBackend(
        model_dict, snapshot["seed"], snapshot["protocol"], snapshot=snapshot
    )


def presets():
    from .tasks import _DEFAULTS

    names = list(_DEFAULTS) + ["boolean." + name for name in BOOLEAN]
    return [
        {
            "id": "toylife." + name,
            "runtime": RUNTIME,
            "family": "toylife",
            "source_status": "deliberate_variant",
        }
        for name in names
    ]


__all__ = ["coverage", "presets", "resolve_model", "create_backend", "restore_backend"]
