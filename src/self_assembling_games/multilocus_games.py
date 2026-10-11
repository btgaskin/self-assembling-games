"""Constructed coordination, fixed-NAND transfer, and composition assays."""

from __future__ import annotations

import itertools

from .finite import FiniteGame, FiniteDecision, ParticipantOutcome, distribution


SOURCE = "constructed:material-realisations-demonstrations"
DESCRIPTIONS = {
    "ML-COORD": "Two intracellular binary code variants",
    "ML-TRANSFER": "Fixed NAND template with two inherited input adapters",
    "ML-COMPOSE": "Three preformed interface variants with parity compatibility",
}


def multilocus_catalogue():
    return [
        {
            "id": identifier,
            "family": "constructed",
            "name": description,
            "source": SOURCE,
            "support": "finite_game",
            "source_questions": [],
        }
        for identifier, description in DESCRIPTIONS.items()
    ]


def multilocus_game(identifier, parameters, *, fixed_decisions=None, **budgets):
    p = dict(parameters)
    fidelity = {
        "game": {
            "status": "explicit_constructed_assay",
            "source": SOURCE,
            "questions": [],
        }
    }
    if identifier == "ML-COORD":
        if set(p) - {"state_probabilities"}:
            raise ValueError("ML-COORD accepts state_probabilities")
        p.setdefault("state_probabilities", [0.5, 0.5])
        distribution(p["state_probabilities"], 2)
        slots = ("sender", "receiver")
        decisions = tuple(
            FiniteDecision(slot, f"input:{n}", (0, 1), slot)
            for slot in slots
            for n in range(2)
        )

        def encounter(provider):
            n = provider.chance(p["state_probabilities"])
            message = provider.decide("sender", f"input:{n}")
            act = provider.decide("receiver", f"input:{message}")
            success = n == act
            return ParticipantOutcome(success, (float(success),) * 2)
    elif identifier == "ML-TRANSFER":
        if set(p) - {"target"}:
            raise ValueError("ML-TRANSFER accepts target adapters")
        p.setdefault("target", [1, 1])
        if len(p["target"]) != 2 or any(
            type(v) is not int or v not in (0, 1) for v in p["target"]
        ):
            raise ValueError("Target requires two binary adapter values")
        slots = ("adapter1", "adapter2")
        decisions = tuple(
            FiniteDecision(slot, "adapter", (0, 1), slot) for slot in slots
        )
        fidelity["template"] = {
            "operation": "NAND",
            "state": "fixed_truth_table",
            "old_route": "retained by construction",
            "adaptive_components": "inherited input adapters",
            "origin": "supplied capacity",
        }

        def encounter(provider):
            pair = provider.chance([0.25] * 4)
            a, b = divmod(pair, 2)
            t1, t2 = (provider.decide(s, "adapter") for s in slots)
            act = 1 - ((a ^ t1) & (b ^ t2))
            target = 1 - ((a ^ p["target"][0]) & (b ^ p["target"][1]))
            success = act == target
            return ParticipantOutcome(success, (float(success),) * 2)
    elif identifier == "ML-COMPOSE":
        if p:
            raise ValueError("ML-COMPOSE has fixed binary compatibility")
        slots = ("source", "bridge", "receiver")
        decisions = tuple(FiniteDecision(s, "interface", (0, 1), s) for s in slots)
        fidelity["modules"] = {
            "origin": "preformed locally functional variants",
            "composite_criterion": "even parity",
        }

        def encounter(provider):
            variants = [provider.decide(s, "interface") for s in slots]
            success = sum(variants) % 2 == 0
            return ParticipantOutcome(success, (float(success),) * 3)
    else:
        raise ValueError(f"Unknown constructed game {identifier}")
    return FiniteGame(
        identifier,
        slots,
        decisions,
        encounter,
        p,
        fidelity,
        fixed_decisions or {},
        **budgets,
    )


def multilocus_profiles():
    from .finite_profiles import AdaptiveProfile

    profiles = []
    for model, names, expression in (
        ("ML-COORD", ("s", "r"), {"sender": [1, 1, 2, 2], "receiver": [1, 2, 1, 2]}),
        (
            "ML-TRANSFER",
            ("g1", "g2"),
            {"adapter1": [0, 0, 1, 1], "adapter2": [0, 1, 0, 1]},
        ),
        (
            "ML-COMPOSE",
            ("c1", "b", "c2"),
            {
                slot: [g[i] for g in itertools.product((0, 1), repeat=3)]
                for i, slot in enumerate(("source", "bridge", "receiver"))
            },
        ),
    ):
        for suffix, method in (
            ("deterministic", "haplotype.deterministic"),
            ("wf", "wright_fisher.haploid"),
        ):
            genetics = {
                "genome": {"loci": [{"name": n, "alleles": [0, 1]} for n in names]},
                "expression": {"plan_indices": expression},
                "transmission": {
                    "mutation": {n: [[0.99, 0.01], [0.01, 0.99]] for n in names},
                    "recombination": {"kind": "clonal"},
                },
                "fitness": {"background": 1, "scale": 1},
            }
            if suffix == "wf":
                genetics["population_size"] = 200
            profiles.append(
                AdaptiveProfile(
                    f"{model.lower()}.{suffix}.demo",
                    model,
                    method=method,
                    population={"individuals": {"cell": list(expression)}},
                    genetics=genetics,
                    assumptions=[
                        "Constructed functional assay with an explicit genotype-to-plan map.",
                        "Functional modules share one haploid individual's genotype.",
                    ],
                ).to_dict()
            )
    return profiles
