"""Declared, fieldwise molecular apparatus and portable array serialization."""

from __future__ import annotations

import base64
import hashlib
import json
import math

import numpy as np


def _integer(value, name, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} requires an integer")
    value = int(value)
    if value < 0 or (maximum is not None and value > maximum):
        raise ValueError(f"{name} outside permitted range")
    return value


def _key(value):
    if isinstance(value, np.generic):
        value = value.item()
    if value is not None and type(value) not in (str, bool, int, float):
        raise ValueError("field values must be JSON scalars")
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def resolve_apparatus(fields, actions, parameters):
    """Resolve channels solely from declared field domains and action labels."""
    p = dict(parameters)
    allowed = {
        "encoding",
        "ligands",
        "molecules",
        "output",
        "dose",
        "max_dose",
        "target_copies",
    }
    if set(p) - allowed:
        raise ValueError(
            "unknown apparatus parameters: " + ", ".join(sorted(set(p) - allowed))
        )
    if "ligands" in p and "molecules" in p:
        raise ValueError("supply ligands or molecules once")
    actions = list(actions)
    if not actions or len({_key(x) for x in actions}) != len(actions):
        raise ValueError("actions must be nonempty and unique")
    maximum = _integer(p.get("max_dose", 8), "max_dose")
    dose = _integer(p.get("dose", 1), "dose", maximum)
    encodings, inputs = {}, {}
    overrides = p.get("encoding", {})
    if set(overrides) - set(fields):
        raise ValueError("encoding contains unknown fields")
    for field in sorted(fields):
        values = list(fields[field])
        keys = [_key(v) for v in values]
        if not values or len(set(keys)) != len(values):
            raise ValueError("field domains must be nonempty and unique")
        boolean = all(type(v) is bool for v in values) or all(
            type(v) is int and v in (0, 1) for v in values
        )
        choice = overrides.get(
            field,
            "presence" if boolean and field not in {"cue", "stimulus"} else "one_hot",
        )
        if isinstance(choice, dict) and (
            set(choice) - {"kind", "mode"} or {"kind", "mode"} <= set(choice)
        ):
            raise ValueError("encoding requires one kind or mode")
        kind = (
            choice.get("kind", choice.get("mode"))
            if isinstance(choice, dict)
            else choice
        )
        if kind not in ("presence", "one_hot", "dose"):
            raise ValueError("unknown encoding")
        if kind == "presence" and not boolean:
            raise ValueError("presence requires Boolean field values")
        if kind == "dose":
            for v in values:
                _integer(v, field, maximum)
        channels = [
            f"input_{len(inputs) + i}"
            for i in range(len(values) if kind == "one_hot" else 1)
        ]
        inputs.update(dict.fromkeys(channels))
        encodings[field] = dict(kind=kind, values=values, channels=channels)
    output = p.get("output", {})
    output = {"representation": output} if isinstance(output, str) else dict(output)
    if set(output) - {"representation", "rule", "permutation"}:
        raise ValueError("unknown output parameters")
    representation = output.get(
        "representation", "bound_free" if len(actions) == 2 else "occupied"
    )
    if representation in ("bound_free", "bound_pair") and len(actions) != 2:
        raise ValueError("binary representations require two actions")
    if representation not in ("bound_free", "bound_pair", "occupied"):
        raise ValueError("unknown output representation")
    count = 1 if representation == "bound_free" else len(actions)
    targets = dict.fromkeys(
        ["act"] if count == 1 else [f"act_{i}" for i in range(count)]
    )
    rule = output.get("rule", "proportional")
    if rule not in ("proportional", "majority") or (
        len(actions) != 2 and rule != "proportional"
    ):
        raise ValueError("invalid output rule")
    permutation = output.get("permutation", list(range(len(actions))))
    if any(type(v) is not int for v in permutation) or sorted(permutation) != list(
        range(len(actions))
    ):
        raise ValueError("invalid output permutation")
    # This fixed panel starts with established experiment endpoints, then enumerates
    # canonical identities. It makes no claim of equivalent binding strengths.
    panel = ["0001", "0110", "1111", "1001"]
    panel += [
        format(i, f"0{n}b")
        for n in range(4, 9)
        for i in range(2**n)
        if i > 0 and format(i, f"0{n}b") <= format(i, f"0{n}b")[::-1]
    ]
    panel = list(dict.fromkeys(min(v, v[::-1]) for v in panel))
    supplied = p.get("ligands", p.get("molecules", {}))
    if ("inputs" in supplied or "targets" in supplied) and set(supplied) - {
        "inputs",
        "targets",
    }:
        raise ValueError("unknown ligand mapping section")
    supplied = (
        {**supplied.get("inputs", {}), **supplied.get("targets", {})}
        if ("inputs" in supplied or "targets" in supplied)
        else dict(supplied)
    )
    names = list(inputs) + list(targets)
    if set(supplied) - set(names):
        raise ValueError("ligands contain unknown channels")
    assigned = {}
    for name, molecule in supplied.items():
        if (
            not isinstance(molecule, str)
            or not 1 <= len(molecule) <= 8
            or set(molecule) - {"0", "1"}
        ):
            raise ValueError("ligands require binary strings of length 1..8")
        assigned[name] = min(molecule, molecule[::-1])
    if len(set(assigned.values())) != len(assigned):
        raise ValueError("ligands collide under reversal")
    available = iter(v for v in panel if v not in assigned.values())
    for name in names:
        if name not in assigned:
            try:
                assigned[name] = next(available)
            except StopIteration as exc:
                raise ValueError("molecular panel exhausted") from exc
    copies = p.get("target_copies", 1)
    if isinstance(copies, dict) and set(copies) - set(targets):
        raise ValueError("unknown target copies")
    copies = {
        name: _integer(
            copies.get(name, 1) if isinstance(copies, dict) else copies,
            "target_copies",
            maximum,
        )
        for name in targets
    }
    return dict(
        inputs={n: assigned[n] for n in inputs},
        targets={n: assigned[n] for n in targets},
        encoding=encodings,
        output=dict(
            representation=representation,
            markers=list(targets),
            rule=rule,
            permutation=list(permutation),
        ),
        diagnostics={
            "panel": "fixed_canonical_4_to_8_v1",
            "contact_profiles": "Distinct canonical identities can share contact profiles; inspect StructuralInterface.endpoint_information with the loaded tables.",
        },
        dose=dose,
        target_copies=copies,
        max_dose=maximum,
        actions=actions,
    )


def encode(apparatus, observation):
    if set(observation) - set(apparatus["encoding"]):
        raise ValueError("unknown observation fields")
    result = dict.fromkeys(apparatus["inputs"], 0)
    for field, value in observation.items():
        spec = apparatus["encoding"][field]
        keys = [_key(v) for v in spec["values"]]
        key = _key(value)
        if key not in keys:
            raise ValueError(f"invalid observation value for {field}")
        if spec["kind"] == "one_hot":
            result[spec["channels"][keys.index(key)]] = apparatus["dose"]
        elif spec["kind"] == "presence":
            result[spec["channels"][0]] = apparatus["dose"] if value else 0
        else:
            result[spec["channels"][0]] = _integer(value, field, apparatus["max_dose"])
    return result


def controller(apparatus, parameters=None):
    from toylife2.protocols import BinaryOutputSpec, ControllerSpec, StructuralInterface

    interface = StructuralInterface(
        **{k: apparatus[k] for k in ("inputs", "targets", "target_copies", "max_dose")}
    )
    kwargs = dict(parameters or {})
    if kwargs.get("mode", "canonical_cell") != "canonical_cell":
        raise ValueError("apparatus requires canonical_cell")
    output = apparatus["output"]
    if output["representation"] in ("bound_free", "bound_pair"):
        kwargs["output"] = BinaryOutputSpec(**output)
    return ControllerSpec(interface=interface, **kwargs)


def probabilities(apparatus, readouts, legal_actions=None):
    output = apparatus["output"]
    markers = output["markers"]
    names = (
        [f"target_free:{markers[0]}", markers[0]]
        if output["representation"] == "bound_free"
        else markers
    )
    arrays = [np.asarray(readouts[n]) for n in names]
    if any(
        a.shape != arrays[0].shape
        or a.dtype.kind not in "iu"
        or np.any(a < 0)
        or np.any(a > np.iinfo(np.int32).max)
        for a in arrays
    ):
        raise ValueError(
            "readouts require matching nonnegative integer counts within int32"
        )
    counts = np.stack(arrays, axis=-1).astype(np.int64)
    if np.any(counts.sum(axis=-1) > np.iinfo(np.int32).max):
        raise OverflowError("output count total exceeds int32")
    if output["rule"] == "majority":
        counts = (counts == counts.max(axis=-1, keepdims=True)).astype(np.int64)
    ordered = np.empty_like(counts)
    ordered[..., output["permutation"]] = counts
    legal = np.ones(len(apparatus["actions"]), dtype=bool)
    if legal_actions is not None:
        supplied = np.asarray(legal_actions)
        if supplied.dtype.kind == "b":
            if supplied.shape[-1:] != (len(legal),):
                raise ValueError("legal mask has incorrect action dimension")
            legal = supplied
        else:
            keys = [_key(a) for a in apparatus["actions"]]
            if len({_key(a) for a in legal_actions}) != len(legal_actions) or any(
                _key(a) not in keys for a in legal_actions
            ):
                raise ValueError("unknown or duplicate legal action")
            legal = np.array([a in {_key(x) for x in legal_actions} for a in keys])
    try:
        legal = np.broadcast_to(legal, ordered.shape)
    except ValueError as exc:
        raise ValueError("legal mask shape mismatch") from exc
    if np.any(legal.sum(axis=-1) == 0):
        raise ValueError("at least one action must be legal")
    weights = np.where(legal, ordered, 0).astype(float)
    weights = np.where(weights.sum(axis=-1, keepdims=True) == 0, legal, weights)
    return weights / weights.sum(axis=-1, keepdims=True)


def pack(value):
    """Represent arrays and scalars by exact typed bytes, without pickle."""
    if isinstance(value, (np.ndarray, np.generic)):
        a = np.asarray(value)
        if a.dtype.hasobject:
            raise ValueError("object arrays are unsupported")
        return {
            "__numpy__": "scalar" if isinstance(value, np.generic) else "array",
            "dtype": np.lib.format.dtype_to_descr(a.dtype),
            "shape": list(a.shape),
            "data": base64.b64encode(a.tobytes(order="C")).decode("ascii"),
        }
    if isinstance(value, dict):
        if "__numpy__" in value:
            raise ValueError("reserved serialization key")
        if any(not isinstance(k, str) for k in value):
            raise ValueError("JSON dictionaries require string keys")
        return {k: pack(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [pack(v) for v in value]
    _key(value)
    return value


def unpack(value):
    if isinstance(value, dict) and "__numpy__" in value:
        if set(value) != {"__numpy__", "dtype", "shape", "data"} or value[
            "__numpy__"
        ] not in ("scalar", "array"):
            raise ValueError("invalid numpy payload")
        descr = value["dtype"]
        if isinstance(descr, list):
            descr = [tuple(x) for x in descr]
        dtype = np.dtype(descr)
        if dtype.hasobject:
            raise ValueError("object arrays are unsupported")
        shape = value["shape"]
        if not isinstance(shape, list) or any(
            type(n) is not int or n < 0 for n in shape
        ):
            raise ValueError("invalid array shape")
        raw = base64.b64decode(value["data"], validate=True)
        if len(raw) != math.prod(shape) * dtype.itemsize:
            raise ValueError("array byte count does not match shape")
        array = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
        if value["__numpy__"] == "scalar":
            if shape:
                raise ValueError("scalar payload requires scalar shape")
            return array[()]
        return array
    if isinstance(value, dict):
        return {k: unpack(v) for k, v in value.items()}
    if isinstance(value, list):
        return [unpack(v) for v in value]
    _key(value)
    return value


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(
            pack(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
