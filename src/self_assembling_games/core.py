"""Conditional policies, explicitly addressed credit, and stochastic events."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def key(value: Any) -> str:
    """Stable typed JSON context binding, independent of the observation."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class Address:
    table: str
    option: Any


@dataclass(frozen=True)
class Decision:
    occurrence: int
    module: str
    table: str
    observation: Any
    option: Any

    @property
    def address(self) -> Address:
        return Address(self.table, self.option)


@dataclass(frozen=True)
class Feedback:
    address: Address
    success: bool
    payoff: float
    origins: tuple[int, ...] = ()


@dataclass(frozen=True)
class Outcome:
    success: bool
    base_reward: float = 0.0
    cost: float = 0.0
    terms: int = 0
    exchanges: int = 0
    status: str = "complete"
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def payoff(self) -> float:
        return self.base_reward - self.cost


@dataclass
class Table:
    module: str
    context: Any
    options: list[Any]
    weights: np.ndarray
    learner: dict[str, Any]
    current: Any = None
    reference: float = 0.0
    frozen: bool = False
    total: float = field(init=False)

    def __post_init__(self) -> None:
        self.weights = np.array(self.weights, dtype=np.float64, copy=True)
        self.validate()

    def validate(self) -> None:
        if not math.isfinite(self.reference):
            raise ValueError("Reference point must be finite")
        if not self.options or len(self.options) != len(self.weights):
            raise ValueError("A table needs one weight for each available option")
        if len({key(x) for x in self.options}) != len(self.options):
            raise ValueError("Duplicate option identity")
        if not np.isfinite(self.weights).all() or np.any(self.weights < 0):
            raise ValueError("Propensities must be finite and nonnegative")
        self.total = float(self.weights.sum())
        if self.total <= 0 or not math.isfinite(self.total):
            raise ValueError("A decision requires positive finite total mass")
        if self.current is not None:
            self.index(self.current)

    def index(self, option: Any) -> int:
        option_key = key(option)
        for i, existing in enumerate(self.options):
            if key(existing) == option_key:
                return i
        raise KeyError(f"Unknown option {option!r}")

    def probabilities(self, *, response: bool = True) -> np.ndarray:
        if response and self.current is not None:
            p = np.zeros(len(self.options))
            p[self.index(self.current)] = 1
        else:
            p = self.weights / self.total
        error = self.learner.get("error", 0.0) if response else 0.0
        return (1 - error) * p + error / len(self.options) if error else p

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(
            {
                "module": self.module,
                "context": self.context,
                "options": self.options,
                "weights": self.weights.tolist(),
                "learner": self.learner,
                "current": self.current,
                "reference": self.reference,
                "frozen": self.frozen,
            }
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Table:
        return cls(**value)


class BranchPoint(Exception):
    def __init__(self, probabilities: list[float]):
        self.probabilities = probabilities


class Runtime:
    """Mutable adaptive state. Games alone specify observations and credit."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.tables: dict[str, Table] = {}
        self.owners: dict[str, str] = {}
        self.module_roles: dict[str, str] = {}
        self.frozen_modules: set[str] = set()
        self.clocks = {"play": 0, "turn": 0, "round": 0, "problem": 0, "phase": 0}
        self.next_identity = 0
        self.next_occurrence = 0
        self.events: list[dict[str, Any]] = []
        self.record = False
        self.sink: Callable[[dict[str, Any]], None] | None = None
        self.training = True
        self.forced: list[int] | None = None
        self.force_position = 0
        self.invocations: dict[str, Any] = {}
        self.replay_chances: list[dict[str, Any]] | None = None

    def emit(self, event: str, **payload: Any) -> None:
        if not self.record and self.sink is None:
            return
        item = {"event": event, "play": self.clocks["play"], **payload}
        if self.record:
            self.events.append(item)
        if self.sink is not None:
            self.sink(item)

    def begin(self) -> None:
        self.events.clear()
        self.invocations.clear()
        self.clocks["turn"] = 0
        self.force_position = 0

    def table_id(self, module: str, context: Any) -> str:
        return module + "/" + key(context)

    def add_table(
        self,
        module: str,
        context: Any,
        options: list[Any],
        learner: dict[str, Any],
        initial: float = 1.0,
        *,
        owner: str | None = None,
        frozen: bool = False,
    ) -> str:
        identity = self.table_id(module, context)
        if module in self.frozen_modules:
            raise ValueError("Cannot create policy state in a frozen module")
        if owner is not None and module in self.owners and self.owners[module] != owner:
            raise ValueError("Module ownership conflicts with its declaration")
        if identity in self.tables:
            raise ValueError(f"Table already exists: {identity}")
        table = Table(
            module,
            context,
            list(options),
            np.full(len(options), initial),
            copy.deepcopy(learner),
            frozen=frozen,
        )
        if learner["rule"] in {"R6", "WSLS", "WSLR", "STRATEGY_WSLR"}:
            table.current = options[
                self.chance(
                    "initial_assignment:" + identity, [1 / len(options)] * len(options)
                )
            ]
        self.tables[identity] = table
        if owner is not None:
            self.owners[module] = owner
        else:
            self.owners.setdefault(module, module)
        self.module_roles.setdefault(module, "decision")
        self.emit("create_table", table=identity, options=options)
        return identity

    def chance(self, label: str, probabilities: list[float] | np.ndarray) -> int:
        p = np.asarray(probabilities, dtype=np.float64)
        if p.ndim != 1 or len(p) == 0 or not np.isfinite(p).all() or np.any(p < 0):
            raise ValueError("Invalid chance distribution")
        if not np.isclose(p.sum(), 1, atol=1e-12, rtol=1e-12):
            raise ValueError("Chance probabilities must sum to one")
        nonzero = np.flatnonzero(p)
        if self.replay_chances is not None:
            if not self.replay_chances:
                raise ValueError("Replay chance tape exhausted")
            event = self.replay_chances.pop(0)
            if event.get("label") != label or event.get("probabilities") != p.tolist():
                raise ValueError("Replay chance distribution changed")
            index = event["index"]
            if not isinstance(index, int) or not 0 <= index < len(p) or p[index] <= 0:
                raise ValueError("Replay outcome is unavailable")
            if len(nonzero) != 1:
                self.rng.random()
        elif len(nonzero) == 1:
            index = int(nonzero[0])
        elif self.forced is not None:
            if self.force_position == len(self.forced):
                raise BranchPoint(p.tolist())
            index = self.forced[self.force_position]
            self.force_position += 1
            if not 0 <= index < len(p) or p[index] <= 0:
                raise ValueError("Forced outcome has zero probability")
        else:
            index = min(
                int(np.searchsorted(np.cumsum(p), self.rng.random(), side="right")),
                len(p) - 1,
            )
        self.emit("chance", label=label, index=index, probabilities=p.tolist())
        return index

    def uniform(self, label: str, n: int) -> int:
        return self.chance(label, [1 / n] * n)

    def integer(self, label: str, upper: int) -> int:
        """Large uniform ranges without allocating a probability vector."""
        if self.forced is not None:
            raise ValueError(
                "Exact enumeration is unavailable for large integer task seeds"
            )
        value = int(self.rng.integers(upper))
        if self.replay_chances is not None:
            event = self.replay_chances.pop(0)
            if (
                event.get("label") != label
                or event.get("upper") != upper
                or event.get("value") != value
            ):
                raise ValueError("Replay integer request changed")
        self.emit("chance", label=label, value=value, upper=upper)
        return value

    def bernoulli(self, label: str, probability: float) -> bool:
        return bool(self.chance(label, [1 - probability, probability]))

    def permutation(self, label: str, n: int) -> list[int]:
        remaining = list(range(n))
        order = []
        while remaining:
            order.append(remaining.pop(self.uniform(label, len(remaining))))
        return order

    def decide(self, module: str, context: Any, observation: Any) -> Decision:
        identity = self.table_id(module, context)
        table = self.tables[identity]
        index = self.chance("decision:" + identity, table.probabilities())
        result = Decision(
            self.next_occurrence,
            module,
            identity,
            copy.deepcopy(observation),
            table.options[index],
        )
        self.next_occurrence += 1
        self.emit(
            "decision",
            occurrence=result.occurrence,
            module=module,
            table=identity,
            observation=observation,
            option=result.option,
        )
        return result

    def credit(self, choices: list[Decision], outcome: Outcome) -> None:
        for decision in choices:
            self.feedback(
                Feedback(
                    decision.address,
                    outcome.success,
                    outcome.payoff,
                    (decision.occurrence,),
                )
            )

    def feedback(self, feedback: Feedback) -> None:
        from .learners import update

        if not math.isfinite(feedback.payoff):
            raise ValueError("Feedback payoff must be finite")
        table = self.tables[feedback.address.table]
        table.index(feedback.address.option)
        if self.training and not table.frozen:
            update(self, table, feedback)
            table.validate()
        self.emit(
            "feedback",
            table=feedback.address.table,
            option=feedback.address.option,
            success=feedback.success,
            payoff=feedback.payoff,
            origins=list(feedback.origins),
            applied=self.training and not table.frozen,
        )

    def fresh(self, namespace: str) -> str:
        value = f"{namespace}:signal:{self.next_identity}"
        self.next_identity += 1
        self.emit("allocate", identity=value)
        return value

    def insert(self, identity: str, option: Any, weight: float = 1.0) -> None:
        table = self.tables[identity]
        if table.frozen:
            raise ValueError("Cannot alter frozen adaptive state")
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("A new option needs positive finite weight")
        if key(option) in {key(x) for x in table.options}:
            raise ValueError("Option already exists")
        table.options.append(option)
        table.weights = np.append(table.weights, weight)
        table.validate()
        self.emit("insert", table=identity, option=option, weight=weight)

    def remove(self, identity: str) -> None:
        if self.tables[identity].frozen:
            raise ValueError("Cannot alter frozen adaptive state")
        del self.tables[identity]
        self.emit("remove_table", table=identity)

    def delete_option(self, identity: str, option: Any) -> None:
        table = self.tables[identity]
        if table.frozen or table.current is not None:
            raise ValueError("Option deletion requires an unfrozen proportional policy")
        if len(table.options) == 1:
            raise ValueError("Cannot empty a decision's option set")
        index = table.index(option)
        if table.total - table.weights[index] <= 0:
            raise ValueError("Option deletion would remove all propensity mass")
        table.options.pop(index)
        table.weights = np.delete(table.weights, index)
        table.validate()
        self.emit("delete_option", table=identity, option=option)

    def freeze(self, module: str) -> None:
        self.frozen_modules.add(module)
        for table in self.tables.values():
            if table.module == module:
                table.frozen = True
        self.emit("freeze", module=module)

    def invoke(self, identity: str, operation: Callable[[], Any]) -> Any:
        if identity not in self.invocations:
            self.invocations[identity] = operation()
            self.emit("invoke", identity=identity)
        return self.invocations[identity]

    def state(self) -> dict[str, Any]:
        return copy.deepcopy(
            {
                "tables": {k: t.to_dict() for k, t in self.tables.items()},
                "table_order": list(self.tables),
                "owners": self.owners,
                "module_roles": self.module_roles,
                "frozen_modules": sorted(self.frozen_modules),
                "clocks": self.clocks,
                "next_identity": self.next_identity,
                "next_occurrence": self.next_occurrence,
                "rng": self.rng.bit_generator.state,
            }
        )

    @classmethod
    def restore(cls, state: dict[str, Any]) -> Runtime:
        runtime = cls()
        order = state["table_order"]
        if len(order) != len(set(order)) or set(order) != set(state["tables"]):
            raise ValueError("Checkpoint table order must contain every table once")
        runtime.tables = {k: Table.from_dict(state["tables"][k]) for k in order}
        runtime.owners = copy.deepcopy(state["owners"])
        runtime.module_roles = copy.deepcopy(state["module_roles"])
        runtime.frozen_modules = set(state["frozen_modules"])
        runtime.clocks = dict(state["clocks"])
        runtime.next_identity = state["next_identity"]
        runtime.next_occurrence = state["next_occurrence"]
        runtime.rng.bit_generator.state = copy.deepcopy(state["rng"])
        return runtime
