"""Present dispositions, adaptation, and association in complete plans.

Run with an installed package, or PYTHONPATH=src python examples/realised_dispositions.py.
"""

from self_assembling_games import (
    Address,
    Feedback,
    FiniteDecision,
    FiniteGame,
    ModelSpecification,
    ParticipantOutcome,
    Runtime,
    SourceDeclaration,
    build,
    register_finite_game,
)


def adaptation():
    runtime = Runtime()
    for name, mass in (("small", 1), ("large", 100)):
        address = runtime.add_table(name, "input", [0, 1], {"rule": "R1"}, initial=mass)
        table = runtime.tables[address]
        before = table.conditional_disposition().tolist()
        runtime.feedback(Feedback(Address(address, 0), True, 1))
        print(name, before, "after reward:", table.conditional_disposition().tolist())


def association():
    def factory(parameters, fixed_decisions, **budgets):
        def encounter(provider):
            left = provider.decide("individual", "left")
            right = provider.decide("individual", "right")
            return ParticipantOutcome(left == right, (float(left == right),))

        return FiniteGame(
            "ASSOCIATION-ASSAY",
            ("individual",),
            (
                FiniteDecision("individual", "left", (0, 1), "A"),
                FiniteDecision("individual", "right", (0, 1), "B"),
            ),
            encounter,
            parameters,
            {"game": {"status": "constructed_assay"}},
            fixed_decisions,
            **budgets,
        )

    register_finite_game(
        "ASSOCIATION-ASSAY",
        factory,
        family="example",
        description="Two internally associated acts",
        source="constructed:association-assay",
    )
    specification = ModelSpecification.resolve(
        game="ASSOCIATION-ASSAY",
        game_parameters={},
        realisation="finite.complete_plan",
        realisation_parameters={},
        source=SourceDeclaration(
            kind="deliberate_variant", identifiers=("constructed:association-assay",)
        ),
    )
    simulation = build(specification, seed=3)
    for name, probabilities in (
        ("correlated", [0.5, 0, 0, 0.5]),
        ("anticorrelated", [0, 0.5, 0.5, 0]),
        ("uniform", [0.25, 0.25, 0.25, 0.25]),
    ):
        simulation.set_policy("individual", probabilities)
        print(name, "success:", simulation.evaluate()["success_probability"])
    print("Encounter bindings:", simulation.model_instance.organisation.slots)


if __name__ == "__main__":
    adaptation()
    association()
