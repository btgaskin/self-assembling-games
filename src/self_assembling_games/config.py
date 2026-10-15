"""Small TOML declarations and resolved, portable scientific configurations."""

from __future__ import annotations

import copy
from pathlib import Path
import tomllib


def _decode_nulls(data):
    data = copy.deepcopy(data)
    for pointer in data.pop("toml_nulls", []):
        parts = [
            part.replace("~1", "/").replace("~0", "~")
            for part in pointer.split("/")[1:]
        ]
        destination = data
        for part in parts[:-1]:
            destination = (
                destination[int(part)]
                if isinstance(destination, list)
                else destination[part]
            )
        key = int(parts[-1]) if isinstance(destination, list) else parts[-1]
        destination[key] = None
    return data


def _encode_nulls(data):
    """TOML has no null; retain their JSON-pointer locations beside the data."""
    nulls = []

    def visit(value, pointer):
        if isinstance(value, dict):
            result = {}
            for key, child in value.items():
                address = pointer + "/" + key.replace("~", "~0").replace("/", "~1")
                if child is None:
                    nulls.append(address)
                else:
                    result[key] = visit(child, address)
            return result
        if isinstance(value, list):
            result = []
            for index, child in enumerate(value):
                address = pointer + f"/{index}"
                if child is None:
                    nulls.append(address)
                    result.append("")
                else:
                    result.append(visit(child, address))
            return result
        return value

    if "toml_nulls" in data:
        raise ValueError("toml_nulls is reserved for TOML null locations")
    result = visit(data, "")
    if nulls:
        result["toml_nulls"] = nulls
    return result


def merge(base: dict, overrides: dict) -> dict:
    """Recursively merge tables; scalar values and arrays replace earlier values."""
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def from_dict(declaration: dict):
    """Resolve a compact declaration or load an already resolved declaration."""
    from .domain import Experiment, Model, model

    data = _decode_nulls(declaration)
    if "schema_version" in data and data["schema_version"] not in (2, 3):
        raise ValueError("Configuration schema_version must be 2 or 3")
    kind = data.get("kind", "model" if "interaction" in data else "experiment")
    if kind not in {"model", "experiment"}:
        raise ValueError("Configuration kind must be model or experiment")
    phased = (
        data.get("schema_version") == 3
        or "phases" in data
        or any("phases" in condition for condition in data.get("conditions", []))
    )
    if phased:
        if kind != "experiment":
            raise ValueError("Schema 3 describes phased experiments; models use schema 2")
        if data.get("schema_version") == 2:
            raise ValueError("Phased experiments require schema_version = 3")
        return _phased(data)
    if kind == "model" and "interaction" in data:
        unknown = data.keys() - {
            "schema_version",
            "kind",
            "id",
            "runtime",
            "interaction",
            "organisation",
            "realisation",
            "adaptation",
            "source",
            "execution",
            "resources",
            "capabilities",
        }
        if unknown:
            raise ValueError("Unknown model fields: " + ", ".join(sorted(unknown)))
        return Model.from_dict(data)
    if (
        kind == "experiment"
        and data.get("conditions")
        and all(
            isinstance(c.get("model"), dict) and "interaction" in c["model"]
            for c in data["conditions"]
        )
    ):
        unknown = data.keys() - {
            "schema_version",
            "kind",
            "conditions",
            "seeds",
            "workers",
            "analysis",
        }
        if unknown:
            raise ValueError("Unknown experiment fields: " + ", ".join(sorted(unknown)))
        for condition in data["conditions"]:
            unknown = condition.keys() - {"name", "model", "budget", "protocol"}
            if unknown:
                raise ValueError(
                    "Unknown condition fields: " + ", ".join(sorted(unknown))
                )
        return Experiment.from_dict(data)

    allowed = {"schema_version", "kind", "preset", "model", "overrides"}
    if kind == "experiment":
        allowed |= {"seeds", "budget", "workers", "analysis", "conditions", "protocol"}
    unknown = data.keys() - allowed
    if unknown:
        raise ValueError("Unknown configuration fields: " + ", ".join(sorted(unknown)))
    if "preset" in data and "model" in data:
        raise ValueError("Choose either preset or model")
    base = data.get("preset", data.get("model"))
    shared = data.get("overrides", {})
    if not isinstance(shared, dict):
        raise ValueError("overrides must be a table")
    if kind == "model":
        if base is None:
            raise ValueError("A model configuration requires preset or model")
        return model(base, **shared)

    declared_conditions = data.get("conditions", [{"name": "default"}])
    if not isinstance(declared_conditions, list) or not declared_conditions:
        raise ValueError("An experiment requires at least one condition")
    conditions = []
    for index, condition in enumerate(declared_conditions):
        extra = condition.keys() - {
            "name",
            "preset",
            "model",
            "overrides",
            "budget",
            "protocol",
        }
        if extra:
            raise ValueError("Unknown condition fields: " + ", ".join(sorted(extra)))
        if "preset" in condition and "model" in condition:
            raise ValueError("Choose either condition preset or model")
        chosen = condition.get("preset", condition.get("model", base))
        if chosen is None:
            raise ValueError("Each condition requires a model or an inherited preset")
        resolved = model(chosen, **merge(shared, condition.get("overrides", {})))
        budget = condition.get(
            "budget", data.get("budget", resolved.to_dict()["execution"]["budget"])
        )
        conditions.append(
            {
                "name": condition.get(
                    "name",
                    "default"
                    if len(declared_conditions) == 1
                    else f"condition-{index + 1}",
                ),
                "model": resolved.to_dict(),
                "budget": budget,
                "protocol": merge(
                    data.get("protocol", {}), condition.get("protocol", {})
                ),
            }
        )
    return Experiment.from_dict(
        {
            "conditions": conditions,
            "seeds": data.get("seeds", [0]),
            "workers": data.get("workers", 1),
            "analysis": data.get("analysis", {}),
        }
    )


def _phased(data):
    from .domain import Experiment, model

    allowed = {
        "schema_version", "kind", "preset", "model", "overrides", "seeds",
        "budget", "workers", "analysis", "conditions", "protocol", "phases",
    }
    if data.keys() - allowed:
        raise ValueError("Unknown configuration fields: " + ", ".join(sorted(data.keys() - allowed)))
    if "preset" in data and "model" in data:
        raise ValueError("Choose either preset or model")
    base = data.get("preset", data.get("model"))
    shared = data.get("overrides", {})
    if not isinstance(shared, dict):
        raise ValueError("overrides must be a table")
    conditions = data.get("conditions", [{"name": "default", "phases": data.get("phases")}])
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("An experiment requires at least one condition")
    resolved = []
    for index, condition in enumerate(conditions):
        if condition.keys() - {"name", "phases", "preset", "model", "overrides", "budget", "protocol", "analysis"}:
            raise ValueError("Unknown phased condition fields")
        if "preset" in condition and "model" in condition:
            raise ValueError("Choose either condition preset or model")
        phases = condition.get("phases", data.get("phases"))
        if not isinstance(phases, list) or not phases:
            raise ValueError("A phased condition requires at least one phase")
        items = []
        for phase in phases:
            if not isinstance(phase, dict) or phase.keys() - {
                "id", "preset", "model", "overrides", "budget", "start", "interventions", "analysis", "protocol"
            }:
                raise ValueError("Unknown phase fields")
            if "preset" in phase and "model" in phase:
                raise ValueError("Choose either phase preset or model")
            chosen = phase.get("preset", phase.get("model", condition.get("preset", condition.get("model", base))))
            if chosen is None:
                raise ValueError("Each phase requires a model or an inherited preset")
            overrides = merge(merge(shared, condition.get("overrides", {})), phase.get("overrides", {}))
            specification = model(chosen, **overrides).to_dict()
            items.append({
                "id": phase.get("id"), "model": specification,
                "budget": phase.get("budget", condition.get("budget", data.get("budget", specification["execution"]["budget"]))),
                "start": phase.get("start", {"kind": "fresh"}),
                "interventions": phase.get("interventions", []),
                "protocol": merge(merge(data.get("protocol", {}), condition.get("protocol", {})), phase.get("protocol", {})),
                "analysis": merge(merge(data.get("analysis", {}), condition.get("analysis", {})), phase.get("analysis", {})),
            })
        resolved.append({"name": condition.get("name", "default" if len(conditions) == 1 else f"condition-{index + 1}"), "phases": items})
    return Experiment.from_dict({
        "schema_version": 3, "kind": "experiment", "conditions": resolved,
        "seeds": data.get("seeds", [0]), "workers": data.get("workers", 1),
        "analysis": data.get("analysis", {}),
    })


def load(path: str | Path):
    """Read TOML and resolve defaults without constructing or advancing a system."""
    with Path(path).open("rb") as stream:
        return from_dict(tomllib.load(stream))


def dumps(value) -> str:
    """Write TOML using tomli-w, including nested tables and condition arrays."""
    import tomli_w

    data = value.to_dict() if hasattr(value, "to_dict") else value
    return tomli_w.dumps(_encode_nulls(data))


def save(value, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(value), encoding="utf-8")
    return path
