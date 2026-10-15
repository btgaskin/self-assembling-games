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
        if (
            data.get("schema_version", 2) != 2
            or data.get("kind", "experiment") != "experiment"
        ):
            raise ValueError("Expected a schema 2 experiment")
        conditions = data.get("conditions", [])
        if not conditions or len({c["name"] for c in conditions}) != len(conditions):
            raise ValueError("Experiment requires uniquely named conditions")
        for condition in conditions:
            if not isinstance(condition["name"], str) or not condition["name"]:
                raise ValueError("Condition names must be nonempty strings")
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
        data.update(schema_version=2, kind="experiment")
        result = object.__new__(cls)
        object.__setattr__(result, "_json", _encode(data))
        return result

    def to_dict(self):
        return json.loads(self._json)


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
    model_or_preset,
    *,
    seeds=None,
    budget=None,
    conditions=None,
    workers=1,
    analysis=None,
):
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
