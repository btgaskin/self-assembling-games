"""Named method classes independent of source families and numerical solvers."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AdaptiveMethod:
    id: str
    method_class: str
    representation: str
    description: str
    nonnegative_returns: bool
    source: str

    def field(
        self, frequencies: np.ndarray, fitness: np.ndarray, mutation_rate: float = 0
    ) -> np.ndarray:
        frequencies = np.asarray(frequencies, dtype=float)
        fitness = np.asarray(fitness, dtype=float)
        if (
            frequencies.ndim != 1
            or not len(frequencies)
            or fitness.shape != frequencies.shape
            or not np.isfinite(fitness).all()
            or not np.isfinite(frequencies).all()
        ):
            raise ValueError(
                "The field needs equally sized finite fitness and frequency vectors"
            )
        if not np.isfinite(mutation_rate) or not 0 <= mutation_rate <= 1:
            raise ValueError("mutation_rate must be in [0,1]")
        if self.method_class != "population_evolution":
            raise ValueError("Reinforcement methods use encounter updates")
        mean = float(frequencies @ fitness)
        if self.id == "replicator.continuous":
            if mutation_rate != 0:
                raise ValueError("Pure replicator selection has no mutation parameter")
            return frequencies * (fitness - mean)
        if self.id == "replicator_mutator.birth_uniform":
            if np.any(fitness < 0):
                raise ValueError(
                    "Birth mutation requires nonnegative reproduction fitness"
                )
            return (
                (1 - mutation_rate) * frequencies * fitness
                + mutation_rate * mean / len(frequencies)
                - frequencies * mean
            )
        raise ValueError(f"No vector field for {self.id}")


_METHODS = {
    m.id: m
    for m in (
        AdaptiveMethod(
            "replicator.continuous",
            "population_evolution",
            "complete_inherited_plans",
            "Continuous-time relative selection with random matching",
            False,
            "https://doi.org/10.1098/rstb.2008.0258#section-3",
        ),
        AdaptiveMethod(
            "replicator_mutator.birth_uniform",
            "population_evolution",
            "complete_inherited_plans",
            "Uniform mutation at reproduction, including the parental type",
            True,
            "https://doi.org/10.1098/rstb.2008.0258#section-3",
        ),
        AdaptiveMethod(
            "reinforcement.conditional.R1",
            "reinforcement",
            "conditional_decisions",
            "Add addressed nonnegative returns to each participating decision occurrence",
            True,
            "Barrett B2; library instantiation for other source families",
        ),
        AdaptiveMethod(
            "reinforcement.strategy.R1",
            "reinforcement",
            "complete_role_plans",
            "Sample a complete plan and reinforce it once per individual encounter",
            True,
            "Explicit library counterpart; no historical Skyrms protocol asserted",
        ),
    )
}


def adaptive_methods(method_class: str | None = None) -> list[dict[str, Any]]:
    """Methods are listed by mathematical/learning class, separately from games."""
    if method_class is not None and method_class not in {
        "population_evolution",
        "reinforcement",
    }:
        raise ValueError("Unknown adaptive method class")
    return [
        asdict(m)
        for m in _METHODS.values()
        if method_class is None or m.method_class == method_class
    ]


def adaptive_method(identifier: str) -> AdaptiveMethod:
    try:
        return _METHODS[identifier]
    except KeyError:
        raise ValueError(
            f"Unknown adaptive method {identifier}; use sg methods"
        ) from None
