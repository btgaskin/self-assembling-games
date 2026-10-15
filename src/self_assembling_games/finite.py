"""Finite interactions and exact complete-plan expectations.

The leaf compiler enumerates chance and decision paths, not encounters between
all composite inherited types. Matching samples a distinct individual for each
role slot. Correlations between decisions within that individual's plan survive
in the leaf compatibility masks.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from dataclasses import dataclass, field
from collections.abc import MutableMapping
from typing import Any, Callable

import numpy as np

from .core import key


@dataclass(frozen=True)
class FiniteDecision:
    """A module's conditional decision bound to an encounter participant slot.

    ``role`` is the established constructor spelling for that slot. Functional
    role and organism identity can be described separately by the source game.
    """

    role: str
    context: str
    options: tuple[Any, ...]
    module: str

    @property
    def slot(self):
        return self.role


@dataclass(frozen=True)
class ParticipantOutcome:
    success: bool
    returns: tuple[float, ...]
    terms: int = 0
    exchanges: int = 0


class _Branch(Exception):
    def __init__(self, probabilities: np.ndarray):
        self.probabilities = probabilities


class DecisionProvider:
    """One encounter's choices; indices are local to each decision's options."""

    def __init__(self, game: FiniteGame):
        self.game = game
        self.draws: list[tuple[str, str, int]] = []
        # Legacy adapters provide credit explicitly; new kernels credit draws.
        self.credits: list[tuple[str, str, int, float]] | None = None
        self.plan_indices: dict[str, int] | None = None

    def chance(self, probabilities: Any) -> int:
        raise NotImplementedError

    def choose(self, role: str, context: str) -> int:
        raise NotImplementedError

    def decide(self, role: str, context: str) -> Any:
        row = self.game.decision(role, context)
        frozen = self.game.fixed_decisions.get(role + "/" + context)
        if frozen is not None:
            index = self.chance(frozen)
        else:
            index = self.choose(role, context)
            self.draws.append((role, context, index))
        return row.options[index]


class _Enumeration(DecisionProvider):
    def __init__(self, game: FiniteGame, prefix: list[int]):
        super().__init__(game)
        self.prefix = prefix
        self.position = 0

    def chance(self, probabilities: Any) -> int:
        raw = np.asarray(probabilities, dtype=float)
        if raw.ndim != 1:
            raise ValueError("Chance probabilities must be a vector")
        p = distribution(raw, len(raw))
        nonzero = np.flatnonzero(p)
        if len(nonzero) == 1:
            return int(nonzero[0])
        if self.position == len(self.prefix):
            raise _Branch(p)
        index = self.prefix[self.position]
        self.position += 1
        return index

    def choose(self, role: str, context: str) -> int:
        count = len(self.game.decision(role, context).options)
        return self.chance(np.full(count, 1 / count))


def distribution(values: Any, size: int) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    if (
        p.shape != (size,)
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or not math.isclose(float(p.sum()), 1, abs_tol=1e-12, rel_tol=1e-12)
    ):
        raise ValueError(f"Expected a probability vector of length {size}")
    return p.copy()


class _LazyRoleArrays(MutableMapping):
    """Materialise a role's numerical array only when it is requested."""

    def __init__(self, factories):
        self.factories = dict(factories)
        self.cache = {}

    def __getitem__(self, role):
        if role not in self.cache:
            self.cache[role] = self.factories[role]()
        return self.cache[role]

    def __setitem__(self, role, value):
        self.factories[role] = lambda: value
        self.cache[role] = value

    def __delitem__(self, role):
        del self.factories[role]
        self.cache.pop(role, None)

    def __iter__(self):
        return iter(self.factories)

    def __len__(self):
        return len(self.factories)

    def defer(self, role, factory):
        self.factories[role] = factory
        self.cache.pop(role, None)


@dataclass
class FiniteGame:
    id: str
    roles: tuple[str, ...]
    decisions: tuple[FiniteDecision, ...]
    interaction: Callable[[DecisionProvider], ParticipantOutcome] = field(repr=False)
    parameters: dict[str, Any] = field(default_factory=dict)
    fidelity: dict[str, Any] = field(default_factory=dict)
    fixed_decisions: dict[str, list[float]] = field(default_factory=dict)
    max_role_strategies: int = 10_000
    max_leaves: int = 100_000
    max_mask_entries: int = 10_000_000

    def __post_init__(self) -> None:
        self.parameters = copy.deepcopy(self.parameters)
        self.fidelity = copy.deepcopy(self.fidelity)
        self.fixed_decisions = copy.deepcopy(self.fixed_decisions)
        if len(set(self.roles)) != len(self.roles) or not self.roles:
            raise ValueError("Roles must be distinct and nonempty")
        if any(
            not isinstance(v, int) or v <= 0
            for v in (self.max_role_strategies, self.max_leaves, self.max_mask_entries)
        ):
            raise ValueError("Finite-game resource budgets must be positive integers")
        self._rows = {(r.role, r.context): r for r in self.decisions}
        if len(self._rows) != len(self.decisions):
            raise ValueError("Duplicate finite decision")
        for row in self.decisions:
            if row.role not in self.roles or not row.options:
                raise ValueError("Each decision needs a declared role and options")
            if len({key(o) for o in row.options}) != len(row.options):
                raise ValueError("Duplicate decision option")
        for address, p in self.fixed_decisions.items():
            matches = [r for r in self.decisions if r.role + "/" + r.context == address]
            if len(matches) != 1:
                raise ValueError(f"Unknown frozen decision {address}")
            distribution(p, len(matches[0].options))
        self.rows = {
            role: tuple(
                r
                for r in self.decisions
                if r.role == role and role + "/" + r.context not in self.fixed_decisions
            )
            for role in self.roles
        }
        self.plan_counts = {
            role: math.prod(len(row.options) for row in rows)
            for role, rows in self.rows.items()
        }
        self.plans = _LazyRoleArrays(
            {role: lambda role=role: self._enumerate_plans(role) for role in self.roles}
        )
        self.row_indices = {
            (role, row.context): i
            for role, rows in self.rows.items()
            for i, row in enumerate(rows)
        }
        self.strategy_costs = _LazyRoleArrays(
            {
                role: lambda role=role: self._zero_strategy_costs(role)
                for role in self.roles
            }
        )
        self.payoff_matrices: tuple[np.ndarray, np.ndarray] | None = None
        self._compiled = False

    def _enumerate_plans(self, role):
        count = self.plan_counts[role]
        if count > self.max_role_strategies:
            raise ValueError(
                f"{role} needs {count} plans; max_role_strategies="
                f"{self.max_role_strategies}; exact finite capability unavailable"
            )
        rows = self.rows[role]
        values = list(itertools.product(*(range(len(r.options)) for r in rows)))
        return np.asarray(values, dtype=int).reshape(count, len(rows))

    def _zero_strategy_costs(self, role):
        # Exact costs share the same resource gate as complete plans.
        self.plans[role]
        return np.zeros(self.plan_counts[role])

    def decision(self, role: str, context: str) -> FiniteDecision:
        return self._rows[role, context]

    def declaration(self):
        """Decision structure and choices, available before exact compilation."""
        return {
            "id": self.id,
            "parameters": copy.deepcopy(self.parameters),
            "fidelity": copy.deepcopy(self.fidelity),
            "roles": list(self.roles),
            "decisions": [
                {
                    "role": d.role,
                    "context": d.context,
                    "options": list(d.options),
                    "module": d.module,
                }
                for d in self.decisions
            ],
            "fixed_decisions": copy.deepcopy(self.fixed_decisions),
            "strategy_counts": dict(self.plan_counts),
            "strategy_order": "lexicographic option indices in declared decision order",
        }

    def to_dict(self) -> dict[str, Any]:
        self.compile()
        signature = hashlib.sha256()
        for array in [
            self.chance_mass,
            self.success,
            self.returns,
            self.terms,
            self.exchanges,
            *self.strategy_costs.values(),
        ]:
            signature.update(str(array.shape).encode())
            signature.update(array.tobytes())
        if self.payoff_matrices is not None:
            for matrix in self.payoff_matrices:
                signature.update(matrix.tobytes())
        signature.update(
            json.dumps([self.draw_paths, self.credit_paths], sort_keys=True).encode()
        )
        return {
            "id": self.id,
            "parameters": copy.deepcopy(self.parameters),
            "fidelity": copy.deepcopy(self.fidelity),
            "roles": list(self.roles),
            "decisions": [
                {
                    "role": d.role,
                    "context": d.context,
                    "options": list(d.options),
                    "module": d.module,
                }
                for d in self.decisions
            ],
            "fixed_decisions": copy.deepcopy(self.fixed_decisions),
            "strategy_counts": {r: len(p) for r, p in self.plans.items()},
            "strategy_order": "lexicographic option indices in declared decision order",
            "finite_semantics_hash": signature.hexdigest(),
        }

    def encounter(self, provider: DecisionProvider) -> ParticipantOutcome:
        result = self.interaction(provider)
        if (
            len(result.returns) != len(self.roles)
            or not np.isfinite(result.returns).all()
        ):
            raise ValueError("Interaction must supply a finite return for every role")
        if result.success not in {True, False} or any(
            not isinstance(v, (int, np.integer)) or v < 0
            for v in (result.terms, result.exchanges)
        ):
            raise ValueError(
                "Interaction needs Boolean success and nonnegative integer term/exchange counts"
            )
        if provider.plan_indices is not None:
            if self.payoff_matrices is not None:
                a, b = (provider.plan_indices[r] for r in self.roles)
                returns = tuple(float(m[a, b]) for m in self.payoff_matrices)
            else:
                returns = tuple(
                    value - self.strategy_costs[role][provider.plan_indices[role]]
                    for role, value in zip(self.roles, result.returns)
                )
            result = ParticipantOutcome(
                result.success, returns, result.terms, result.exchanges
            )
        return result

    def compile(self) -> None:
        """Enumerate finite paths once; chance mass excludes decision probabilities."""
        if self._compiled:
            return
        for role in self.roles:
            self.plans[role]
        pending = [([], 1.0)]
        leaves = []
        visited = 0
        while pending:
            prefix, mass = pending.pop()
            visited += 1
            if visited > self.max_leaves * 20:
                raise ValueError("Finite path exploration exceeded node budget")
            provider = _Enumeration(self, prefix)
            try:
                result = self.encounter(provider)
            except _Branch as branch:
                # Decision factors are undone at each completed leaf below.
                pending.extend(
                    (prefix + [i], mass * float(p))
                    for i, p in enumerate(branch.probabilities)
                    if p > 0
                )
                continue
            if len(leaves) >= self.max_leaves:
                raise ValueError(f"Finite paths exceed max_leaves={self.max_leaves}")
            # Uniform branching in choose supplied 1/k; undo those factors.
            mass *= math.prod(
                len(self.decision(r, c).options) for r, c, _ in provider.draws
            )
            leaves.append((mass, result, provider.draws, provider.credits))
        if (
            len(leaves) * sum(len(p) for p in self.plans.values())
            > self.max_mask_entries
        ):
            raise ValueError("Finite compatibility masks exceed max_mask_entries")
        self.chance_mass = np.array([leaf[0] for leaf in leaves])
        self.success = np.array([leaf[1].success for leaf in leaves], dtype=float)
        self.returns = np.array([leaf[1].returns for leaf in leaves], dtype=float)
        self.terms = np.array([leaf[1].terms for leaf in leaves], dtype=float)
        self.exchanges = np.array([leaf[1].exchanges for leaf in leaves], dtype=float)
        self.draw_paths = [leaf[2] for leaf in leaves]
        self.credit_paths = [leaf[3] for leaf in leaves]
        self.masks = {}
        for role in self.roles:
            masks = np.ones((len(leaves), len(self.plans[role])), dtype=bool)
            for k, (_, _, draws, _) in enumerate(leaves):
                for r, c, option in draws:
                    if r == role:
                        masks[k] &= (
                            self.plans[role][:, self.row_indices[r, c]] == option
                        )
            self.masks[role] = masks
        self._compiled = True

    def expectations(self, role_frequencies: dict[str, np.ndarray]) -> dict[str, Any]:
        """Exact role-plan fitness, with independent individual matching."""
        if role_frequencies.keys() != self.plans.keys():
            raise ValueError("Expected a complete role-frequency mapping")
        role_frequencies = {
            r: distribution(role_frequencies[r], len(self.plans[r])) for r in self.roles
        }
        self.compile()
        z = {r: self.masks[r] @ role_frequencies[r] for r in self.roles}
        mass = self.chance_mass.copy()
        for r in self.roles:
            mass *= z[r]
        fitness = {}
        for i, role in enumerate(self.roles):
            other = self.chance_mass * self.returns[:, i]
            for r in self.roles:
                if r != role:
                    other = other * z[r]
            fitness[role] = self.masks[role].T @ other - self.strategy_costs[role]
        if self.payoff_matrices is not None:
            a, b = self.roles
            fitness[a] = self.payoff_matrices[0] @ role_frequencies[b]
            fitness[b] = self.payoff_matrices[1].T @ role_frequencies[a]
        return {
            "fitness": fitness,
            "success_probability": float(mass @ self.success),
            "expected_returns": {
                r: float(role_frequencies[r] @ fitness[r]) for r in self.roles
            },
            "expected_terms": float(mass @ self.terms),
            "expected_exchanges": float(mass @ self.exchanges),
            "completed_mass": float(mass.sum()),
        }

    def conditional_expectations(
        self, policies: dict[str, np.ndarray]
    ) -> dict[str, Any]:
        expected = {
            r + "/" + d.context: len(d.options)
            for r, rows in self.rows.items()
            for d in rows
        }
        if policies.keys() != expected.keys():
            raise ValueError("Expected all adaptive conditional decision policies")
        policies = {k: distribution(policies[k], size) for k, size in expected.items()}
        if self.payoff_matrices is not None or any(
            np.any(v) for v in self.strategy_costs.values()
        ):
            raise ValueError(
                "Strategy-level listening/table costs need complete-plan sampling; SKQ3"
            )
        self.compile()
        mass = self.chance_mass.copy()
        for k, draws in enumerate(self.draw_paths):
            for role, context, option in draws:
                mass[k] *= policies[role + "/" + context][option]
        return {
            "success_probability": float(mass @ self.success),
            "expected_returns": {
                r: float(mass @ self.returns[:, i]) for i, r in enumerate(self.roles)
            },
            "expected_terms": float(mass @ self.terms),
            "expected_exchanges": float(mass @ self.exchanges),
            "completed_mass": float(mass.sum()),
        }
