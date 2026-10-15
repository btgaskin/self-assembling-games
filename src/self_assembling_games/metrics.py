"""Bounded measurements, exact branching evaluation, and semantic fixtures."""

from __future__ import annotations

import itertools
import math
from collections import deque
from typing import TYPE_CHECKING, Any

from .core import BranchPoint, Outcome

if TYPE_CHECKING:
    from .simulation import Simulation


class Metrics:
    def __init__(self, window: int = 1000):
        self.window = window
        self.recent: deque[bool] = deque(maxlen=window)
        self.attempts = 0
        self.completed = 0
        self.successes = 0
        self.payoff = 0.0
        self.terms = 0
        self.exchanges = 0
        self.statuses: dict[str, int] = {}
        self.owner_returns: dict[str, float] = {}

    def add(self, outcome: Outcome) -> None:
        self.attempts += 1
        self.statuses[outcome.status] = self.statuses.get(outcome.status, 0) + 1
        if outcome.status != "complete":
            return
        self.completed += 1
        self.successes += int(outcome.success)
        self.payoff += outcome.payoff
        for owner, value in (outcome.owner_returns or {}).items():
            self.owner_returns[owner] = self.owner_returns.get(owner, 0.0) + value
        self.terms += outcome.terms
        self.exchanges += outcome.exchanges
        self.recent.append(outcome.success)

    def summary(self) -> dict[str, Any]:
        count = self.completed
        return {
            "attempts": self.attempts,
            "completed": count,
            "successes": self.successes,
            "cumulative_accuracy": self.successes / count if count else None,
            "window_accuracy": sum(self.recent) / len(self.recent)
            if self.recent
            else None,
            "mean_net_payoff": self.payoff / count if count else None,
            "mean_terms": self.terms / count if count else None,
            "mean_exchanges": self.exchanges / count if count else None,
            "statuses": dict(self.statuses),
            **({"mean_owner_returns": {k: v / count for k, v in self.owner_returns.items()}} if self.owner_returns and count else {}),
        }

    def state(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "recent": list(self.recent),
            "attempts": self.attempts,
            "completed": self.completed,
            "successes": self.successes,
            "payoff": self.payoff,
            "terms": self.terms,
            "exchanges": self.exchanges,
            "statuses": dict(self.statuses),
            **({"owner_returns": dict(self.owner_returns)} if self.owner_returns else {}),
        }

    @classmethod
    def restore(cls, state: dict[str, Any]) -> Metrics:
        metrics = cls(state["window"])
        for name, value in state.items():
            if name != "recent":
                setattr(metrics, name, value)
        metrics.recent = deque(state["recent"], maxlen=state["window"])
        return metrics


def evaluate(
    snapshot: dict[str, Any] | Simulation,
    protocol: str = "exact",
    *,
    samples: int = 1000,
    seed: int = 0,
    max_branches: int = 200_000,
) -> dict[str, Any]:
    from .simulation import Simulation

    base = snapshot.snapshot() if isinstance(snapshot, Simulation) else snapshot
    from .evolution import EvolutionSimulation
    from .finite_learning import ReinforcementSimulation
    from .multilocus import MultilocusSimulation

    if isinstance(
        snapshot, (EvolutionSimulation, ReinforcementSimulation, MultilocusSimulation)
    ):
        base = snapshot.snapshot()
    if isinstance(base, dict) and base.get("engine") == "adaptive":
        from .adaptive_runtime import restore_adaptive

        sim = restore_adaptive(base)
        if isinstance(sim, (EvolutionSimulation, MultilocusSimulation)):
            if protocol != "exact":
                raise ValueError(
                    "Evolutionary population evaluation uses exact finite expectations"
                )
            return sim.evaluate()
        return sim.evaluate(protocol=protocol, samples=samples, seed=seed)
    if protocol == "sampled":
        sim = Simulation.restore(base)
        import numpy as np

        sim.runtime.rng = np.random.Generator(np.random.PCG64(seed))
        sim.runtime.training = False
        evaluation = Metrics()
        for _ in range(samples):
            evaluation.add(sim.step())
        return {"protocol": "sampled", "seed": seed, **evaluation.summary()}
    if protocol != "exact":
        raise ValueError("Evaluation protocol is exact or sampled")
    if base["profile"]["model"] in {
        "INQ",
        "VIS",
        "STAG",
        "FFSR",
        "META-RULE",
        "INV",
        "ORD",
        "ADD",
    }:
        raise ValueError(
            "Exact one-task evaluation is unavailable for this stateful/dynamic family; use sampled"
        )
    pending: list[tuple[list[int], float]] = [([], 1.0)]
    leaves = 0
    explored = 0
    totals = {
        "success_probability": 0.0,
        "expected_net_payoff": 0.0,
        "expected_terms": 0.0,
        "expected_exchanges": 0.0,
        "completed_mass": 0.0,
        "truncated_mass": 0.0,
    }
    while pending:
        prefix, mass = pending.pop()
        explored += 1
        if explored > max_branches:
            raise ValueError(f"Exact evaluation exceeded max_branches={max_branches}")
        sim = Simulation.restore(base)
        sim.runtime.training = False
        sim.runtime.forced = prefix
        try:
            outcome = sim.step()
        except BranchPoint as branch:
            pending.extend(
                (prefix + [i], mass * p)
                for i, p in enumerate(branch.probabilities)
                if p > 0
            )
            continue
        leaves += 1
        if outcome.status != "complete":
            totals["truncated_mass"] += mass
            continue
        totals["completed_mass"] += mass
        totals["success_probability"] += mass * outcome.success
        totals["expected_net_payoff"] += mass * outcome.payoff
        totals["expected_terms"] += mass * outcome.terms
        totals["expected_exchanges"] += mass * outcome.exchanges
        for owner, value in (outcome.owner_returns or {}).items():
            returns = totals.setdefault("expected_owner_returns", {})
            returns[owner] = returns.get(owner, 0.0) + mass * value
    if base["profile"]["model"] == "GEN-APPROX":
        parameters = base["profile"]["parameters"]
        maximum = 2.0 if parameters["payoff_kind"] == "binary" else parameters["payoff_height"]
        totals["normalized_expected_payoff"] = totals["expected_net_payoff"] / maximum
    return {"protocol": "exact", "leaves": leaves, **totals}


def held_out_ordering(
    snapshot: dict[str, Any] | Simulation, *, samples: int = 1000, seed: int = 0
) -> dict[str, Any]:
    """Evaluate nonadjacent relations without using the training pair sampler."""
    import numpy as np

    from .games import ordering_call, relation
    from .simulation import Simulation

    base = snapshot.snapshot() if isinstance(snapshot, Simulation) else snapshot
    sim = Simulation.restore(base)
    rt = sim.runtime
    p = sim.profile.parameters
    if sim.profile.model not in {"TR-ORD", "TR-PART"}:
        raise ValueError("Held-out ordering requires an ordering-transfer snapshot")
    rt.training = False
    rt.rng = np.random.Generator(np.random.PCG64(seed))
    pairs = [(a, b) for a in range(p["n"]) for b in range(p["n"]) if abs(a - b) > 1]
    if not pairs:
        raise ValueError("No nonadjacent held-out pairs")
    successes = 0
    for _ in range(samples):
        rt.begin()
        a, b = pairs[rt.uniform("held_out_pair", len(pairs))]
        da = rt.decide("TA", a, a)
        db = rt.decide("TB", b, b)
        answer = (
            relation(da.option, db.option)
            if p["template_mode"] == "constructed"
            else ordering_call(rt, da.option, db.option)
        )
        successes += answer == relation(a, b)
    return {
        "protocol": "nonadjacent_frozen",
        "samples": samples,
        "successes": successes,
        "accuracy": successes / samples,
        "seed": seed,
    }


def nand_transfer_closure(
    *, bijective: bool = False, complement_output: bool = False
) -> set[tuple[int, ...]]:
    maps = [(0, 1), (1, 0)] if bijective else list(itertools.product([0, 1], repeat=2))
    results = set()
    for left, right in itertools.product(maps, repeat=2):
        table = tuple(
            1 - left[a] * right[b] for a, b in itertools.product([0, 1], repeat=2)
        )
        results.add(table)
        if complement_output:
            results.add(tuple(1 - x for x in table))
    return results


def wilson(successes: int, total: int) -> list[float]:
    if total <= 0:
        raise ValueError("A confidence interval needs observations")
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    middle = (p + z * z / (2 * total)) / denominator
    radius = (
        z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    )
    return [max(0.0, middle - radius), min(1.0, middle + radius)]


def conventional_kinds(
    snapshot: dict[str, Any] | Simulation, module: str, *, tolerance: float = 0.0
) -> list[list[Any]]:
    """Group contexts with matching option distributions under a declared tolerance."""
    from .simulation import Simulation

    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Similarity tolerance is finite and nonnegative")
    sim = Simulation.restore(
        snapshot.snapshot() if isinstance(snapshot, Simulation) else snapshot
    )
    groups: list[tuple[dict[str, float], list[Any]]] = []
    for table in sim.runtime.tables.values():
        if table.module != module:
            continue
        from .core import key

        policy = {
            key(option): float(p)
            for option, p in zip(table.options, table.probabilities())
        }
        for representative, contexts in groups:
            if policy.keys() == representative.keys() and all(
                abs(policy[k] - representative[k]) <= tolerance for k in policy
            ):
                contexts.append(table.context)
                break
        else:
            groups.append((policy, [table.context]))
    if not groups:
        raise ValueError(f"Unknown module {module}")
    return [contexts for _, contexts in groups]
