"""Adapters from the Barrett reinforcement runtime to generic finite games."""

from __future__ import annotations

import copy
from typing import Any

from .core import Decision, Runtime
from .finite import DecisionProvider, FiniteDecision, FiniteGame, ParticipantOutcome

_LEGACY_FINITE = {
    "SIG",
    "COD",
    "NAND",
    "HIER",
    "CUE",
    "MAN",
    "SCOD",
    "TR-LOG",
    "COMP-NAND",
    "TR-ORD",
    "TR-PART",
    "ADD-COMP",
    "DIAL",
    "DIAL-restricted",
    "ASK",
    "DIAL-costly",
    "ATT-R",
    "ATT-B",
    "MH",
}


def barrett_game(profile: Any, *, source_snapshot=None, **budgets: int) -> FiniteGame:
    """Read-only compatibility adapter for stationary finite Barrett profiles.

    The serial runtime supplies interaction order, chance, fixed templates and
    feedback. Adaptive decision policies are supplied by the finite backend.
    Stateful and structurally changing families need an additional model.
    """
    from .registry import load_profile
    from .simulation import Simulation
    from .games import play

    source_profile = load_profile(profile)
    if source_profile.model not in _LEGACY_FINITE:
        raise ValueError(f"{source_profile.model} has no stationary finite adapter")
    base = (
        Simulation(source_profile, 0)
        if source_snapshot is None
        else Simulation.restore(source_snapshot)
    )
    if base.profile.to_dict() != source_profile.to_dict():
        raise ValueError("Embedded Barrett source configuration differs")
    if source_profile.learner["rule"] in {
        "R3",
        "R4",
        "R6",
        "WSLS",
        "WSLR",
        "STRATEGY_WSLR",
        "DISCOUNT",
        "BOUNDED",
    }:
        raise ValueError(
            "Finite adapter needs a memoryless R1/R2/SIGNED source profile"
        )
    addresses = {
        identity: FiniteDecision(
            base.runtime.owners[table.module],
            identity,
            tuple(table.options),
            table.module,
        )
        for identity, table in base.runtime.tables.items()
        if not table.frozen
    }
    roles = tuple(dict.fromkeys(d.role for d in addresses.values()))
    if not roles:
        raise ValueError("Finite adapter needs at least one adaptive participant")

    class AdapterRuntime(Runtime):
        def __init__(self, provider: DecisionProvider):
            # Tables are read-only. Invocation/chance/occurrence state is local.
            self.__dict__ = base.runtime.__dict__.copy()
            self.invocations = {}
            self.events = []
            self.clocks = base.runtime.clocks.copy()
            self.provider = provider
            self.training = False
            self.next_occurrence = 0
            provider.credits = []

        def chance(self, label, probabilities):
            return self.provider.chance(probabilities)

        def decide(self, module, context, observation):
            identity = self.table_id(module, context)
            table = self.tables[identity]
            if table.frozen:
                option = table.options[self.provider.chance(table.probabilities())]
            else:
                row = addresses[identity]
                option = self.provider.decide(row.role, identity)
            result = Decision(
                self.next_occurrence, module, identity, observation, option
            )
            self.next_occurrence += 1
            return result

        def feedback(self, feedback):
            identity = feedback.address.table
            if identity in addresses:
                row = addresses[identity]
                if row.role + "/" + row.context in self.provider.game.fixed_decisions:
                    return
                self.provider.credits.append((
                    row.role, row.context,
                    self.tables[identity].index(feedback.address.option),
                    feedback.payoff,
                ))

    def interaction(provider):
        from .model import ModelInstance

        sim = copy.copy(base)
        sim.runtime = AdapterRuntime(provider)
        sim.world = copy.deepcopy(base.world)
        sim.interaction = copy.deepcopy(base.interaction)
        sim.model_instance = ModelInstance(
            base.model_instance.specification,
            base.model_instance.game,
            sim.runtime.organisation,
            sim.runtime,
        )
        result = play(sim)
        if result.status != "complete":
            raise ValueError(
                "Finite adapter encountered censoring; use an explicit finite terminating game"
            )
        returns = result.returns_for(base.runtime.owners.values())
        return ParticipantOutcome(
            result.success,
            tuple(returns[role] for role in roles),
            result.terms,
            result.exchanges,
        )

    game = FiniteGame(
        source_profile.model,
        roles,
        tuple(addresses.values()),
        interaction,
        {"barrett_profile": source_profile.to_dict()},
        {
            "game": {
                "status": "barrett_adapter",
                "source": source_profile.source,
                "assumptions": source_profile.assumptions,
            }
        },
        **budgets,
    )
    game.construction_state = {"barrett_source": base.snapshot()}
    return game
