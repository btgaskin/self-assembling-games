"""One lifecycle for bound state, execution, inspection and checkpointing."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path

from .domain import Model, provider
from .simulation import atomic_json, digest, provenance


def _identity(runtime):
    module = provider(runtime)
    path = inspect.getsourcefile(module)
    return {
        "module": module.__name__,
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None,
    }


class System:
    def __init__(self, model, seed=0, protocol=None):
        if type(seed) is not int or seed < 0:
            raise ValueError("Seed must be a nonnegative integer")
        self.model = model if isinstance(model, Model) else Model.from_dict(model)
        self.seed = seed
        self.protocol = copy.deepcopy(protocol or {})
        data = self.model.to_dict()
        self._backend = provider(data["runtime"]).create_backend(
            data, seed, self.protocol
        )

    @property
    def clock(self):
        return self._backend.clock

    @property
    def position(self):
        return self._backend.position

    @property
    def status(self):
        return self._backend.status

    def advance(self, amount):
        self._backend.advance(amount)
        return self

    def step_stage(self):
        self._backend.step_stage()
        return self

    def inspect(self):
        result = copy.deepcopy(self._backend.inspect())
        if hasattr(self, "lineage"):
            result["lifecycle"] = copy.deepcopy(self.lineage)
        return result

    def fork(self):
        snapshot = self.snapshot()
        child = type(self).restore(snapshot)
        child.lineage = {"parent_checkpoint_hash": snapshot["checkpoint_hash"],
                         "operation": "fork"}
        return child

    def set_trainable(self, modules, enabled):
        from .transitions import set_trainable

        set_trainable(self, modules, enabled)
        return self

    def transition(self, target_model, transition):
        from .transitions import transition_system

        return transition_system(self, target_model, transition)

    def snapshot(self):
        data = self.model.to_dict()
        result = {
            "schema_version": 2,
            "model": data,
            "model_hash": digest(data),
            "seed": self.seed,
            "protocol": copy.deepcopy(self.protocol),
            "provenance": provenance(),
            "provider": _identity(data["runtime"]),
            "state": self._backend.snapshot(),
        }
        if hasattr(self, "lineage"):
            result["lineage"] = copy.deepcopy(self.lineage)
        result["checkpoint_hash"] = digest(result)
        return copy.deepcopy(result)

    def checkpoint(self, path):
        atomic_json(Path(path), self.snapshot())
        return Path(path)

    @classmethod
    def restore(cls, snapshot):
        value = (
            json.loads(Path(snapshot).read_text())
            if isinstance(snapshot, (str, Path))
            else copy.deepcopy(snapshot)
        )
        if value.get("schema_version") != 2:
            raise ValueError("Unsupported System checkpoint schema")
        expected = value.pop("checkpoint_hash", None)
        if digest(value) != expected or digest(value["model"]) != value["model_hash"]:
            raise ValueError("Checkpoint integrity mismatch")
        current = provenance()
        for key in ("package_version", "source_hash", "numpy"):
            if value["provenance"][key] != current[key]:
                raise ValueError(f"Checkpoint provenance mismatch: {key}")
        if value["provider"] != _identity(value["model"]["runtime"]):
            raise ValueError("Checkpoint provider mismatch")
        instance = cls.__new__(cls)
        instance.model = Model.from_dict(value["model"])
        instance.seed, instance.protocol = value["seed"], value["protocol"]
        if "lineage" in value:
            instance.lineage = copy.deepcopy(value["lineage"])
        instance._backend = provider(value["model"]["runtime"]).restore_backend(
            value["model"], value["state"]
        )
        return instance

    def evaluate(self, protocol=None, **parameters):
        # Evaluation owns a restored copy, including its RNG and controller state.
        data = self.model.to_dict()
        if protocol is not None:
            parameters["protocol"] = protocol
        elif data["runtime"] == "builtin" and "evaluate.exact" not in data.get(
            "capabilities", []
        ):
            parameters["protocol"] = "sampled"
        detached = type(self).restore(self.snapshot())
        return copy.deepcopy(detached._backend.evaluate(**parameters))


def evaluate(system, **parameters):
    return system.evaluate(**parameters)
