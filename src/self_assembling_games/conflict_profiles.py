"""Source-game comparators and a declared SCOD utility-conflict research variant.

Martínez and Godfrey-Smith (2016), Table 1 and equations (1)-(2):
https://doi.org/10.1086/685743
Zollman, Bergstrom and Huttegger (2013), figures 1-3 and section 2b:
https://doi.org/10.1098/rspb.2012.1878

The named profiles retain their own initialisation and execution protocols.
They supply source-game comparisons rather than historical basin reproductions.
"""

from __future__ import annotations

import copy
import json
import math

import numpy as np


GM16 = "https://doi.org/10.1086/685743"
GSM13 = "https://doi.org/10.1371/journal.pcbi.1003282"
ZBH13 = "https://doi.org/10.1098/rspb.2012.1878"
SCOD_SOURCE = "https://doi.org/10.1093/9780197793572.003.0006"
GM16_SENDER = [[5, 6, 0], [2, 0, 6], [0, 1, 5]]
GM16_RECEIVER = [[0, 5, 6], [4, 0, 6], [5, 6, 3]]


def _profile(identifier, model, parameters, learner, sources, assumptions):
    return {
        "id": identifier,
        "model": model,
        # Persisted declarations have independent values rather than Python
        # aliases between participant utilities and local-feedback tables.
        "parameters": json.loads(json.dumps(parameters, allow_nan=False)),
        "learner": copy.deepcopy(learner),
        "kind": "deliberate_variant",
        "plays": 10_000,
        "source": list(sources),
        "assumptions": list(assumptions),
    }


def gm16_table1_profile():
    identity = [[state == act for act in range(3)] for state in range(3)]
    return _profile(
        "gm16.table1.r1",
        "SIG",
        {
            "states": 3,
            "signals": 3,
            "actions": 3,
            "state_probabilities": [1 / 3] * 3,
            "success_matrix": identity,
            "owner_payoffs": {"sender": GM16_SENDER, "receiver": GM16_RECEIVER},
            "feedback_payoffs": {"sender": GM16_SENDER, "receiver": GM16_RECEIVER},
        },
        {"rule": "R1"},
        [GM16, GSM13],
        [
            "Martínez and Godfrey-Smith (2016), Table 1; state-major payoff arrays.",
            "Conditional R1 is a library learning comparator; the source dynamics uses complete-plan populations.",
            "Scalar success and task reward measure state-action identity match; communication is measured by state-action mutual information.",
            "One initial propensity per option and 10000 plays are this comparator's protocol.",
        ],
    )


def zbh13_cost_profile(regime):
    if regime not in {"separating", "hybrid-equilibrium"}:
        raise ValueError("Cost regime is separating or hybrid-equilibrium")
    x, c1, c2 = (0.25, 0.1, 1.1 if regime == "separating" else 0.4)
    sender = [[0, 1], [0, 1]]
    receiver = [[0, 1], [1, 0]]
    return _profile(
        f"zbh13.{regime}.signed",
        "SIG",
        {
            "states": 2,
            "signals": 2,
            "actions": 2,
            "state_probabilities": [x, 1 - x],
            "success_matrix": receiver,
            "owner_payoffs": {"sender": sender, "receiver": receiver},
            "feedback_payoffs": {"sender": sender, "receiver": receiver},
            "sender_signal_costs": [[0, c1], [0, c2]],
        },
        {"rule": "SIGNED", "floor": 1.0},
        [ZBH13],
        [
            "Zollman, Bergstrom and Huttegger (2013), differential-cost game, figures 1-3 and section 2b.",
            f"Chosen comparator parameters: prior(T1)={x}, c1={c1}, c2={c2}; raw costs are retained.",
            "Signal 0 is silence, signal 1 is costly; action 0 is A2, action 1 is A1.",
            "Scalar success/task reward is the receiver's type-appropriate action; sender payoff is A1 benefit minus signal cost.",
            "SIGNED floor-one learning is a library comparator; the source evolutionary law is two-population replicator dynamics.",
        ],
    )


def scod_conflict_profile(mixture):
    """A/B prefer cyclic +1/-1 acts as the utility mixture moves toward one."""
    if (
        isinstance(mixture, bool)
        or not isinstance(mixture, (int, float))
        or not math.isfinite(mixture)
        or not 0 <= mixture <= 1
    ):
        raise ValueError("SCOD payoff mixture must be in [0,1]")
    receiver = [[float(state == act) for act in range(4)] for state in range(4)]
    sender_payoffs = {
        module: [
            [
                (1 - mixture) * float(act == state)
                + mixture * float(act == (state + offset) % 4)
                for act in range(4)
            ]
            for state in range(4)
        ]
        for module, offset in (("A", 1), ("B", 3))
    }
    feedback = {
        **sender_payoffs,
        **{module: receiver for module in ("R", "RA", "RB", "salience")},
    }
    return _profile(
        f"scod.conflict.lambda-{mixture:g}.r1",
        "SCOD",
        {
            "state_probabilities": [0.25] * 4,
            "success_matrix": receiver,
            "owner_payoffs": {**sender_payoffs, "receiver": receiver},
            "feedback_payoffs": feedback,
        },
        {"rule": "R1"},
        [SCOD_SOURCE, GM16, ZBH13],
        [
            "Constructed research variant of Barrett B6 SCOD; utility conflict is an explicit extension.",
            f"Payoff mixture lambda={mixture:g}; this is a utility-mixture parameter, not the published common-interest index.",
            "Receiver rewards identity acts; A mixes identity with cyclic +1, B with cyclic -1; lambda=.5 ties these preferred acts.",
            "Both senders observe the full state and participate; salience and the selected decoder receive receiver feedback.",
            "Four equiprobable states, binary sender options and all SCOD conditional tables are retained.",
        ],
    )


def conflict_profiles():
    """Named conditional presets; all returned declarations are independent."""
    return [
        gm16_table1_profile(),
        zbh13_cost_profile("separating"),
        zbh13_cost_profile("hybrid-equilibrium"),
        *(scod_conflict_profile(value) for value in (0, 0.25, 0.5, 0.75, 1)),
    ]


def conflict_adaptive_profiles():
    """The same source payoff games under the source's complete-plan law."""
    from .finite_profiles import AdaptiveProfile

    profiles = []
    for source in (
        gm16_table1_profile(),
        zbh13_cost_profile("separating"),
        zbh13_cost_profile("hybrid-equilibrium"),
    ):
        stem = source["id"].rsplit(".", 1)[0]
        profiles.append(
            AdaptiveProfile(
                f"{stem}.evolution",
                "SIG",
                {"profile": source},
                method="replicator.continuous",
                kind="paper_game",
                protocol={
                    "duration": 50.0,
                    "initialization": {"kind": "dirichlet", "concentration": 1.0},
                },
                assumptions=[
                    "Source-game comparator with separate complete-plan sender/receiver populations.",
                    "Participant returns supply selection fitness; local kernel feedback is inactive under evolution.",
                    "Seeded Dirichlet(1), duration 50 and the library solver are this study's protocol; historical basin reproduction is separate.",
                    *source["source"],
                ],
            ).to_dict()
        )
    return profiles


def signalling_information_from_policies(state_probabilities, sender, receiver):
    """Exact information in p(state, signal, act) for independent SIG participants.

    Sender rows condition on the state; receiver rows condition on the signal.
    All three information quantities use base-two logarithms and are in bits.
    """
    from .finite import distribution

    sender = np.asarray(sender, dtype=float)
    receiver = np.asarray(receiver, dtype=float)
    if (
        sender.ndim != 2
        or receiver.ndim != 2
        or not sender.shape[0]
        or not sender.shape[1]
        or not receiver.shape[1]
        or sender.shape[1] != receiver.shape[0]
    ):
        raise ValueError("Expected state×signal and signal×act policy matrices")
    nature = distribution(state_probabilities, sender.shape[0])
    sender = np.array([distribution(row, sender.shape[1]) for row in sender])
    receiver = np.array([distribution(row, receiver.shape[1]) for row in receiver])
    joint = np.einsum("s,sm,ma->sma", nature, sender, receiver)

    def mutual_information(pair):
        independent = pair.sum(axis=1)[:, None] * pair.sum(axis=0)[None, :]
        positive = pair > 0
        value = float(
            np.sum(pair[positive] * np.log2(pair[positive] / independent[positive]))
        )
        return max(0.0, value)

    return {
        "protocol": "exact_signalling_policy",
        "state_signal_mutual_information_bits": mutual_information(joint.sum(axis=2)),
        "signal_action_mutual_information_bits": mutual_information(joint.sum(axis=0)),
        "state_action_mutual_information_bits": mutual_information(joint.sum(axis=1)),
        "joint_state_signal_action": joint.tolist(),
    }


def signalling_information(simulation):
    """Inspect a built-in SIG System/simulation without advancing or adapting it.

    Example: signalling_information(model('gm16.table1.evolution').instantiate(3)).
    Population and complete-plan policies are marginalised within each distinct
    encounter participant. Same-participant joint plans require their own assay.
    """
    from .core import Table, key

    if hasattr(simulation, "_backend"):
        simulation = simulation._backend.simulation
    realisation = simulation.model_instance.realisation
    if realisation.id == "barrett.conditional":
        source = simulation.profile
        if source.model != "SIG":
            raise ValueError("Signalling information currently assays the SIG kernel")
        parameters = simulation.model_instance.parameters

        def policy(module, context):
            return realisation.tables[module + "/" + key(context)].probabilities()

    else:
        game = simulation.game
        source = game.parameters.get("barrett_profile", {})
        if source.get("model") != "SIG":
            raise ValueError("Signalling information currently assays the SIG kernel")
        parameters = source["parameters"]
        slots = {
            module: {row.role for row in game.decisions if row.module == module}
            for module in ("sender", "receiver")
        }
        if slots["sender"] & slots["receiver"]:
            raise ValueError(
                "Information assay requires distinct encounter participants"
            )
        if realisation.id == "population.complete_plan":
            frequencies = realisation.encounter_plan_distributions()
        elif realisation.id == "finite.complete_plan":
            frequencies = {
                role: weights / weights.sum()
                for role, weights in realisation.weights.items()
            }
        elif realisation.id == "finite.conditional":
            frequencies = None
        else:
            raise ValueError("Unsupported SIG policy representation")
        source_tables = game.construction_state["barrett_source"]["runtime"]["tables"]

        def policy(module, context):
            identity = module + "/" + key(context)
            row = next(
                (
                    row
                    for row in game.decisions
                    if row.context == identity and row.module == module
                ),
                None,
            )
            if row is None:
                return Table.from_dict(source_tables[identity]).probabilities()
            address = row.role + "/" + row.context
            if address in game.fixed_decisions:
                return np.array(game.fixed_decisions[address])
            if frequencies is None:
                weights = realisation.weights[address]
                return weights / weights.sum()
            column = game.row_indices[row.role, row.context]
            return np.bincount(
                game.plans[row.role][:, column],
                weights=frequencies[row.role],
                minlength=len(row.options),
            )

    sender = np.array(
        [policy("sender", state) for state in range(parameters["states"])]
    )
    receiver = np.array(
        [policy("receiver", signal) for signal in range(parameters["signals"])]
    )
    return signalling_information_from_policies(
        parameters["state_probabilities"], sender, receiver
    )
