"""Ordered haplotypes, expressed plans, and concrete transmission operators."""

from __future__ import annotations

import copy
import itertools
import math
from dataclasses import dataclass

import numpy as np

from .core import key
from .finite import distribution


def positive_integer(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def probability(value, name, upper=1.0):
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not 0 <= value <= upper
    ):
        raise ValueError(f"{name} must be in [0,{upper}]")
    return float(value)


@dataclass(frozen=True)
class Locus:
    name: str
    alleles: tuple

    def __post_init__(self):
        object.__setattr__(self, "alleles", tuple(self.alleles))
        if not isinstance(self.name, str) or not self.name or not self.alleles:
            raise ValueError("A locus needs a name and an ordered allele domain")
        if any(not isinstance(a, (str, int, bool)) for a in self.alleles):
            raise ValueError("Allele labels must be strings, integers, or booleans")
        if len({key(a) for a in self.alleles}) != len(self.alleles):
            raise ValueError("Duplicate allele identity")


class GenomeSpec:
    """Haploid allele combinations in declared locus and allele order."""

    def __init__(self, loci, *, max_haplotypes=4096):
        self.loci = tuple(loci)
        if not self.loci or any(not isinstance(locus, Locus) for locus in self.loci):
            raise ValueError("A genome needs an ordered collection of loci")
        if len({locus.name for locus in self.loci}) != len(self.loci):
            raise ValueError("Duplicate locus name")
        self.shape = tuple(len(locus.alleles) for locus in self.loci)
        self.count = math.prod(self.shape)
        if self.count > positive_integer(max_haplotypes, "max_haplotypes"):
            raise ValueError("Genome exceeds max_haplotypes")
        self.coordinates = np.asarray(
            list(itertools.product(*(range(n) for n in self.shape)))
        )
        self.coordinates.setflags(write=False)

    @classmethod
    def from_dict(cls, value, **budgets):
        if not isinstance(value, dict) or set(value) != {"loci"}:
            raise ValueError("Genome specification requires loci")
        loci = []
        for row in value["loci"]:
            if not isinstance(row, dict) or set(row) != {"name", "alleles"}:
                raise ValueError("Loci require name and alleles")
            loci.append(Locus(row["name"], row["alleles"]))
        return cls(loci, **budgets)

    def to_dict(self):
        return {
            "loci": [
                {"name": locus.name, "alleles": list(locus.alleles)}
                for locus in self.loci
            ]
        }

    @property
    def haplotypes(self):
        return [
            [locus.alleles[i] for locus, i in zip(self.loci, row)]
            for row in self.coordinates
        ]

    def allele_frequencies(self, frequencies):
        x = distribution(frequencies, self.count)
        return {
            locus.name: np.bincount(
                self.coordinates[:, i], weights=x, minlength=len(locus.alleles)
            ).tolist()
            for i, locus in enumerate(self.loci)
        }

    def linkage_disequilibria(self, frequencies):
        x = distribution(frequencies, self.count)
        result = []
        for i, j in itertools.combinations(range(len(self.loci)), 2):
            if self.shape[i] == self.shape[j] == 2:
                a, b = self.coordinates[:, i] == 1, self.coordinates[:, j] == 1
                result.append(
                    {
                        "loci": [self.loci[i].name, self.loci[j].name],
                        "D": float(x @ (a & b) - (x @ a) * (x @ b)),
                    }
                )
        return result


class ExpressionMap:
    """One complete plan per encounter slot for each ordered haplotype.

    Several haplotypes may express the same plans. Their transmission states
    remain distinct. Plan indices refer to the finite game's declared coordinates.
    """

    def __init__(self, genome, game, value):
        if not isinstance(value, dict) or set(value) != {"plan_indices"}:
            raise ValueError("Expression requires plan_indices")
        if value["plan_indices"].keys() != game.plans.keys():
            raise ValueError("Expression must cover every encounter slot")
        self.plan_indices = {}
        for slot in game.roles:
            raw = np.asarray(value["plan_indices"][slot])
            if (
                raw.shape != (genome.count,)
                or raw.dtype.kind not in {"i", "u"}
                or np.any(raw < 0)
                or np.any(raw >= game.plan_counts[slot])
            ):
                raise ValueError(
                    f"Expression needs a valid plan index per haplotype for {slot}"
                )
            raw = raw.astype(np.int64, copy=True)
            raw.setflags(write=False)
            self.plan_indices[slot] = raw

    def to_dict(self):
        return {"plan_indices": {s: v.tolist() for s, v in self.plan_indices.items()}}


class TransmissionSpec:
    """Copying or parental-source exchange, then independent locus mutation.

    Parents are independent draws with replacement from selected haplotypes.
    The tensor contractions preserve associations within each parental subset.
    """

    def __init__(
        self,
        genome,
        value=None,
        *,
        max_recombination_patterns=65536,
        max_transmission_work=10_000_000,
    ):
        self.genome = genome
        value = copy.deepcopy(value or {})
        if set(value) - {"mutation", "recombination"}:
            raise ValueError("Transmission accepts mutation and recombination")
        mutation = value.get("mutation", {})
        if not isinstance(mutation, dict) or set(mutation) - {
            locus.name for locus in genome.loci
        }:
            raise ValueError("Mutation matrices must name declared loci")
        self.mutation = []
        for locus in genome.loci:
            size = len(locus.alleles)
            matrix = np.asarray(mutation.get(locus.name, np.eye(size)), dtype=float)
            if matrix.shape != (size, size):
                raise ValueError("Mutation matrices must cover the allele domain")
            matrix = np.stack([distribution(row, size) for row in matrix])
            matrix.setflags(write=False)
            self.mutation.append(matrix)
        recombination = value.get("recombination", {"kind": "clonal"})
        if not isinstance(recombination, dict):
            raise ValueError("Recombination requires a named mode")
        kind = recombination.get("kind")
        boundaries = len(genome.loci) - 1
        limit = positive_integer(
            max_recombination_patterns, "max_recombination_patterns"
        )
        work_limit = positive_integer(max_transmission_work, "max_transmission_work")
        if kind == "clonal" and set(recombination) == {"kind"}:
            self.recombination = {"kind": kind}
            count = 1
        elif kind == "independent_intervals" and set(recombination) == {
            "kind",
            "fractions",
        }:
            fractions = [
                probability(r, "recombination fraction", 0.5)
                for r in recombination["fractions"]
            ]
            if len(fractions) != boundaries:
                raise ValueError(
                    "Supply one recombination fraction per adjacent interval"
                )
            self.recombination = {"kind": kind, "fractions": fractions}
            count = 2 ** sum(r > 0 for r in fractions)
        elif kind == "single_boundary_exchange" and set(recombination) == {
            "kind",
            "exchange_probability",
        }:
            if not boundaries:
                raise ValueError(
                    "Single-boundary exchange requires an internal boundary"
                )
            c = probability(
                recombination["exchange_probability"], "exchange_probability"
            )
            self.recombination = {"kind": kind, "exchange_probability": c}
            count = int(c < 1) + boundaries * int(c > 0)
        else:
            raise ValueError("Unknown recombination mode or fields")
        work = genome.count * (count * len(genome.loci) + sum(genome.shape))
        if count > limit or work > work_limit:
            raise ValueError(
                "Transmission exceeds recombination-pattern or work budget"
            )
        self.estimated_work = work
        # Fix the first parental source to zero: parents are exchangeable iid draws.
        zero = (0,) * len(genome.loci)
        if kind == "clonal":
            self.patterns = [(zero, 1.0)]
        elif kind == "single_boundary_exchange":
            c = self.recombination["exchange_probability"]
            self.patterns = [(zero, 1 - c)] if c < 1 else []
            self.patterns += [
                ((0,) * b + (1,) * (len(genome.loci) - b), c / boundaries)
                for b in range(1, boundaries + 1)
                if c
            ]
        else:
            self.patterns = [((), 1.0)]
            for r in self.recombination["fractions"]:
                self.patterns = [
                    (mask + (switch,), weight * (r if switch else 1 - r))
                    for mask, weight in self.patterns
                    for switch in (0, 1)
                    if (r if switch else 1 - r) > 0
                ]
            self.patterns = [
                (tuple(np.cumsum((0,) + switches) % 2), weight)
                for switches, weight in self.patterns
            ]

    def to_dict(self):
        return {
            "mutation": {
                locus.name: m.tolist()
                for locus, m in zip(self.genome.loci, self.mutation)
            },
            "recombination": copy.deepcopy(self.recombination),
        }

    def recombine(self, frequencies):
        y = distribution(frequencies, self.genome.count).reshape(self.genome.shape)
        result = np.zeros_like(y)
        for mask, weight in self.patterns:
            first = tuple(i for i, source in enumerate(mask) if source == 0)
            second = tuple(i for i, source in enumerate(mask) if source == 1)
            if not second:
                result += weight * y
            else:
                result += (
                    weight
                    * y.sum(axis=second, keepdims=True)
                    * y.sum(axis=first, keepdims=True)
                )
        return self._normalise(result.reshape(-1))

    def mutate(self, frequencies):
        values = distribution(frequencies, self.genome.count).reshape(self.genome.shape)
        for axis, matrix in enumerate(self.mutation):
            values = np.moveaxis(np.tensordot(values, matrix, axes=(axis, 0)), -1, axis)
        return self._normalise(values.reshape(-1))

    @staticmethod
    def _normalise(values):
        if (
            not np.isfinite(values).all()
            or np.any(values < 0)
            or not math.isclose(float(values.sum()), 1, rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise ValueError("Transmission produced invalid offspring probabilities")
        return values / values.sum()
