"""Immutable scientific declarations shared by Python and TOML clients."""

from __future__ import annotations

import importlib
import json
import math
from dataclasses import dataclass


def provider(runtime):
    """Resolve built-in or importable provider modules in a fresh process."""
    name = {"builtin": ".builtin", "situated": ".situated"}.get(runtime, runtime)
    return importlib.import_module(name, __package__)


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, init=False)
class Model:
    """A resolved declaration; mutable runtime state belongs to its System."""

    _json: str

    def __init__(self, *args, **kwargs):
        raise TypeError("Use the resolving helper or from_dict")

    @classmethod
    def from_dict(cls, value):
        data = json.loads(_encode(value))
        for key in (
            "id",
            "interaction",
            "organisation",
            "realisation",
            "adaptation",
            "source",
            "execution",
            "runtime",
        ):
            if key not in data:
                raise ValueError(f"Model requires {key}")
        execution = data["execution"]
        if execution["clock"] not in ("encounters", "generations", "time"):
            raise ValueError("Unknown execution clock")
        budget = execution["budget"]
        if (
            isinstance(budget, bool)
            or not isinstance(budget, (int, float))
            or not math.isfinite(budget)
            or budget < 0
        ):
            raise ValueError("Budget must be finite and nonnegative")
        if execution["clock"] != "time" and not isinstance(budget, int):
            raise ValueError("Discrete clocks require integer budgets")
        allowed = {
            "schema_version",
            "kind",
            "id",
            "interaction",
            "organisation",
            "realisation",
            "adaptation",
            "source",
            "execution",
            "runtime",
            "resources",
            "capabilities",
        }
        if data.keys() - allowed:
            raise ValueError(
                "Unknown model fields: " + ", ".join(sorted(data.keys() - allowed))
            )
        if data.get("schema_version", 2) != 2 or data.get("kind", "model") != "model":
            raise ValueError("Expected a schema 2 model")
        data = provider(data["runtime"]).resolve_model(data, {})
        data.update(schema_version=2, kind="model")
        result = object.__new__(cls)
        object.__setattr__(result, "_json", _encode(data))
        return result

    def to_dict(self):
        return json.loads(self._json)

    @property
    def id(self):
        return self.to_dict()["id"]

    def instantiate(self, seed=0, protocol=None):
        from .system import System

        return System(self, seed=seed, protocol=protocol)

    def inspect(self):
        return self.to_dict()


@dataclass(frozen=True, init=False)
class Experiment:
    """Resolved conditions and independent replicate seeds."""

    _json: str

    def __init__(self, *args, **kwargs):
        raise TypeError("Use the resolving helper or from_dict")

    @classmethod
    def from_dict(cls, value):
        data = json.loads(_encode(value))
        allowed = {
            "schema_version",
            "kind",
            "conditions",
            "seeds",
            "workers",
            "analysis",
        }
        if data.keys() - allowed:
            raise ValueError(
                "Unknown experiment fields: " + ", ".join(sorted(data.keys() - allowed))
            )
        version = data.get("schema_version", 2)
        if (
            version not in (2, 3)
            or data.get("kind", "experiment") != "experiment"
        ):
            raise ValueError("Expected a schema 2 or 3 experiment")
        conditions = data.get("conditions", [])
        if not conditions or len({c["name"] for c in conditions}) != len(conditions):
            raise ValueError("Experiment requires uniquely named conditions")
        for condition in conditions:
            if not isinstance(condition["name"], str) or not condition["name"]:
                raise ValueError("Condition names must be nonempty strings")
            if version == 3:
                _validate_phases(condition, data.get("analysis", {}))
                continue
            if condition.keys() - {"name", "model", "budget", "protocol"}:
                raise ValueError("Unknown condition fields")
            if not isinstance(condition.get("protocol", {}), dict):
                raise ValueError("Condition protocol must be a table")
            condition["model"] = Model.from_dict(condition["model"]).to_dict()
            condition.setdefault("budget", condition["model"]["execution"]["budget"])
            condition.setdefault("protocol", {})
            check = condition["model"].copy()
            check["execution"] = {**check["execution"], "budget": condition["budget"]}
            Model.from_dict(check)
        seeds = data.setdefault("seeds", [0])
        if (
            not seeds
            or any(type(s) is not int or s < 0 for s in seeds)
            or len(set(seeds)) != len(seeds)
        ):
            raise ValueError("Seeds must be distinct nonnegative integers")
        workers = data.setdefault("workers", 1)
        if type(workers) is not int or workers < 1:
            raise ValueError("Workers must be a positive integer")
        data.setdefault("analysis", {})
        data.update(schema_version=version, kind="experiment")
        result = object.__new__(cls)
        object.__setattr__(result, "_json", _encode(data))
        return result

    def to_dict(self):
        return json.loads(self._json)


def _validate_phases(condition, analysis):
    """Validate ordered, explicit initialisations without constructing state."""
    if condition.keys() - {"name", "phases"}:
        raise ValueError("Schema 3 conditions accept name and phases")
    phases = condition.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError("A phased condition requires at least one phase")
    previous = {}
    for phase in phases:
        if not isinstance(phase, dict) or phase.keys() - {
            "id", "model", "budget", "start", "protocol", "interventions", "analysis"
        }:
            raise ValueError("Unknown phase fields")
        identifier = phase.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in previous:
            raise ValueError("Phase ids must be nonempty and unique within a condition")
        phase["model"] = Model.from_dict(phase["model"]).to_dict()
        phase.setdefault("budget", phase["model"]["execution"]["budget"])
        check = phase["model"].copy()
        check["execution"] = {**check["execution"], "budget": phase["budget"]}
        Model.from_dict(check)
        for key in ("protocol", "analysis"):
            if not isinstance(phase.setdefault(key, {}), dict):
                raise ValueError(f"Phase {key} must be a table")
        from .config import merge

        phase["analysis"] = merge(analysis, phase["analysis"])
        start = phase.setdefault("start", {"kind": "fresh"})
        if not isinstance(start, dict) or start.get("kind") not in {"fresh", "carry", "compose"}:
            raise ValueError("Phase start.kind must be fresh, carry or compose")
        allowed = {
            "fresh": {"kind"},
            "carry": {"kind", "phase", "transition"},
            "compose": {"kind", "bindings"},
        }[start["kind"]]
        if start.keys() - allowed:
            raise ValueError("Unknown phase start fields")
        if start["kind"] == "carry":
            if start.get("phase") not in previous:
                raise ValueError("Carry references must name an earlier phase")
            transition = start.setdefault("transition", {"kind": "identity"})
            if not isinstance(transition, dict) or not isinstance(transition.get("kind"), str):
                raise ValueError("A carry transition requires a named kind")
            source = previous[start["phase"]]
            if source["model"]["execution"]["clock"] != phase["model"]["execution"]["clock"]:
                raise ValueError("State carry requires the same native clock")
        elif start["kind"] == "compose":
            bindings = start.get("bindings")
            if not isinstance(bindings, dict) or not bindings:
                raise ValueError("Compose requires named component bindings")
            for slot, binding in bindings.items():
                if not isinstance(slot, str) or not slot or not isinstance(binding, dict):
                    raise ValueError("Invalid component binding")
                if binding.keys() - {"phase", "export"} or binding.get("phase") not in previous:
                    raise ValueError("Component bindings must reference an earlier phase")
                if not isinstance(binding.setdefault("export", {}), dict):
                    raise ValueError("Component export specification must be a table")
        interventions = phase.setdefault("interventions", [])
        if not isinstance(interventions, list):
            raise ValueError("Phase interventions must be an array")
        for intervention in interventions:
            if (
                not isinstance(intervention, dict)
                or intervention.keys() - {"kind", "modules", "enabled"}
                or intervention.get("kind") != "trainability"
                or type(intervention.get("enabled")) is not bool
            ):
                raise ValueError("Supported phase interventions declare trainability, modules and enabled")
            modules = intervention.get("modules")
            if (
                not isinstance(modules, list) or not modules
                or any(not isinstance(module, str) or not module for module in modules)
                or len(set(modules)) != len(modules)
            ):
                raise ValueError("Trainability modules must be distinct nonempty names")
        previous[identifier] = phase


def model(identifier, **overrides):
    if isinstance(identifier, Model):
        identifier = identifier.to_dict()
    runtime = (
        identifier.get("runtime", "builtin")
        if isinstance(identifier, dict)
        else (
            "self_assembling_games.toylife"
            if identifier.lower().startswith("toylife.")
            else "situated"
            if identifier.lower() in ("ecg", "ecg-inspired", "ecg.demo", "ecg.scripted")
            else "builtin"
        )
    )
    return Model.from_dict(provider(runtime).resolve_model(identifier, overrides))


def experiment(
    model_or_preset=None,
    *,
    seeds=None,
    budget=None,
    conditions=None,
    workers=1,
    analysis=None,
    phases=None,
):
    if phases is not None:
        if conditions is not None:
            raise ValueError("Declare phased conditions through config.from_dict")
        from .config import from_dict

        declaration = {
            "schema_version": 3, "phases": phases,
            "seeds": [0] if seeds is None else seeds,
            "workers": workers, "analysis": analysis or {},
        }
        if model_or_preset is not None:
            declaration["model"] = (
                model_or_preset.to_dict()
                if isinstance(model_or_preset, Model) else model_or_preset
            )
        if budget is not None:
            declaration["budget"] = budget
        return from_dict(declaration)
    base = model(model_or_preset)
    resolved = []
    for name, changes in (conditions or {"default": {}}).items():
        chosen = model(base, **changes)
        resolved.append(
            {
                "name": name,
                "model": chosen.to_dict(),
                "budget": chosen.to_dict()["execution"]["budget"]
                if budget is None
                else budget,
                "protocol": {},
            }
        )
    return Experiment.from_dict(
        {
            "conditions": resolved,
            "seeds": [0] if seeds is None else seeds,
            "workers": workers,
            "analysis": analysis or {},
        }
    )


def resolve(value):
    if isinstance(value, (Model, Experiment)):
        return value.to_dict()
    return model(value).to_dict()
