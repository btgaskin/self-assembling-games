"""Source-specific Skyrms (2009) games built on the generic finite core."""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

from .finite import (
    DecisionProvider,
    FiniteDecision,
    FiniteGame,
    ParticipantOutcome,
    distribution,
)

SKYRMS_SOURCE = "https://doi.org/10.1098/rstb.2008.0258"


def _positive_integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def skyrms_game(
    model: str,
    parameters: dict[str, Any] | None = None,
    *,
    fixed_decisions: dict[str, list[float]] | None = None,
    **budgets: int,
) -> FiniteGame:
    """Build a paper-defined game or an explicitly named unresolved variant."""
    p = copy.deepcopy(parameters or {})
    supported = {"SK-SIG", "SK-COMP", "SK-VOTE", "SK-TEAM", "SK-CHAIN", "SK-DIALOGUE"}
    if model not in supported:
        raise ValueError(f"Unknown Skyrms game {model}")
    allowed = {
        "SK-SIG": {
            "states",
            "signals",
            "actions",
            "state_probabilities",
            "payoffs",
            "success_matrix",
            "background",
            "signal_costs",
            "listening_cost",
            "cost_case",
            "cost_interpretation",
            "c",
        },
        "SK-COMP": {"state_probabilities"},
        "SK-VOTE": {"error", "state_probabilities"},
        "SK-TEAM": {"state_probabilities"},
        "SK-CHAIN": {"state_probabilities"},
        "SK-DIALOGUE": {
            "state_probabilities",
            "problem_probabilities",
            "reply_binding",
        },
    }[model]
    if p.keys() - allowed:
        raise ValueError(
            "Unknown game parameters: " + ", ".join(sorted(p.keys() - allowed))
        )
    roles = (
        ("sender", "receiver")
        if model == "SK-SIG"
        else (
            ("A", "B", "receiver")
            if model == "SK-COMP"
            else ("A", "B", "C", "receiver")
            if model == "SK-VOTE"
            else ("sender", "R1", "R2")
            if model == "SK-TEAM"
            else ("sender", "translator", "receiver")
            if model == "SK-CHAIN"
            else ("observer", "decider")
        )
    )
    rows: list[FiniteDecision] = []

    def add(role: str, name: str, count: int, options: int, module: str | None = None):
        rows.extend(
            FiniteDecision(role, f"{name}:{i}", tuple(range(options)), module or role)
            for i in range(count)
        )

    n = 4 if model in {"SK-COMP", "SK-TEAM", "SK-DIALOGUE"} else 2
    if model == "SK-SIG":
        n = _positive_integer(p.setdefault("states", 2), "states")
        signals = _positive_integer(p.setdefault("signals", n), "signals")
        actions = _positive_integer(p.setdefault("actions", n), "actions")
        case = p.get("cost_case")
        if case is not None:
            if (n, signals, actions) != (2, 2, 2) or case not in {
                "sending",
                "listening",
                "state_specific",
            }:
                raise ValueError(
                    "Paper cost cases require a binary game and a named case"
                )
            interpretation = p.get("cost_interpretation")
            if interpretation not in {"prose_derived", "table_literal"}:
                raise ValueError(
                    "Cost cases require explicit cost_interpretation; SKQ2"
                )
            conflicting = {
                "state_probabilities",
                "background",
                "signal_costs",
                "listening_cost",
                "payoffs",
                "success_matrix",
            } & p.keys()
            if conflicting:
                raise ValueError(
                    "Paper cost cases specify their own distributions and costs"
                )
            p["state_probabilities"] = (
                [0.5, 0.5] if case == "sending" else [1 / 3, 2 / 3]
            )
            p["background"] = 1.0
            c = float(p.get("c", 0.1))
            if not math.isfinite(c) or c < 0:
                raise ValueError("c must be finite and nonnegative")
            p["signal_costs"] = (
                [[2 * c, 0], [2 * c, 0]]
                if case == "sending"
                else [[0.3, 0], [0.3, 0]]
                if case == "listening"
                else [[0, 0.3], [0.3, 0]]
            )
            p["listening_cost"] = 0 if case == "sending" else 0.1
        elif "cost_interpretation" in p or "c" in p:
            raise ValueError("cost_interpretation/c require cost_case")
        add("sender", "state", n, signals)
        add("receiver", "signal", signals, actions)
        reward = np.asarray(p.get("payoffs", np.eye(n, actions).tolist()), dtype=float)
        success = np.asarray(p.get("success_matrix", np.eye(n, actions).tolist()))
        if reward.shape != (n, actions) or not np.isfinite(reward).all():
            raise ValueError("payoffs must be finite states × actions")
        if "payoffs" in p and "success_matrix" not in p:
            raise ValueError("Custom payoffs need a separate success_matrix")
        if success.shape != reward.shape or not np.isin(success, [False, True]).all():
            raise ValueError("success_matrix must be Boolean states × actions")
        if "payoffs" not in p and n != actions:
            raise ValueError("Unequal states/actions need payoffs and success_matrix")
        costs = np.asarray(
            p.get("signal_costs", np.zeros((n, signals)).tolist()), dtype=float
        )
        if (
            costs.shape != (n, signals)
            or not np.isfinite(costs).all()
            or np.any(costs < 0)
        ):
            raise ValueError("signal_costs must be nonnegative states × signals")
        background = float(p.get("background", 0))
        listening = float(p.get("listening_cost", 0))
        if (
            not math.isfinite(background)
            or not math.isfinite(listening)
            or listening < 0
        ):
            raise ValueError(
                "Background/listening cost must be finite; listening cost >= 0"
            )
    elif model == "SK-COMP":
        for r in ["A", "B"]:
            add(r, "cell", 2, 2)
        add("receiver", "pair", 4, 4)
    elif model == "SK-VOTE":
        error = float(p.setdefault("error", 0.1))
        if not math.isfinite(error) or not 0 <= error <= 1:
            raise ValueError("Observation error must be in [0,1]")
        for r in ["A", "B", "C"]:
            add(r, "apparent", 2, 2)
        add("receiver", "count", 4, 2)
    elif model == "SK-TEAM":
        add("sender", "state", 4, 4)
        for r in ["R1", "R2"]:
            add(r, "signal", 2, 2)
    elif model == "SK-CHAIN":
        add("sender", "state", 2, 2)
        add("translator", "signal", 2, 2)
        add("receiver", "signal", 2, 2)
    else:
        if p.get("reply_binding") not in {"partition_cell", "query_cell"}:
            raise ValueError("Dialogue needs explicit reply_binding; SKQ4")
        if "state_probabilities" not in p:
            raise ValueError("Dialogue needs explicit four-state distribution; SKQ4")
        problem_probabilities = distribution(
            p.setdefault("problem_probabilities", [0.5, 0.5]), 2
        )
        add("observer", "select", 2, 2, "partition_selector")
        add("observer", "reply", 4, 2, "reply")
        add("decider", "query", 2, 2, "query")
        add("decider", "decode", 4, 2, "decoder")
    nature = distribution(p.setdefault("state_probabilities", [1 / n] * n), n)
    if np.any(nature <= 0):
        raise ValueError(
            "Skyrms source games require every state to have positive probability"
        )

    def interaction(provider: DecisionProvider) -> ParticipantOutcome:
        if model == "SK-DIALOGUE":
            problem = provider.chance(problem_probabilities)
            query = provider.decide("decider", f"query:{problem}")
            partition = provider.decide("observer", f"select:{query}")
            x = provider.chance(nature)
            cell = x // 2 if partition == 0 else x % 2
            binding = partition if p["reply_binding"] == "partition_cell" else query
            reply = provider.decide("observer", f"reply:{2 * binding + cell}")
            action = provider.decide("decider", f"decode:{2 * problem + reply}")
            correct = action == (x // 2 if problem == 0 else x % 2)
            return ParticipantOutcome(correct, (float(correct),) * 2, 2, 1)
        x = provider.chance(nature)
        if model == "SK-SIG":
            signal = provider.decide("sender", f"state:{x}")
            act = provider.decide("receiver", f"signal:{signal}")
            value = background + reward[x, act]
            return ParticipantOutcome(
                bool(success[x, act]),
                (float(value - costs[x, signal]), float(value)),
                1,
            )
        if model == "SK-COMP":
            a = provider.decide("A", f"cell:{x // 2}")
            b = provider.decide("B", f"cell:{x % 2}")
            correct = provider.decide("receiver", f"pair:{2 * a + b}") == x
            terms = 2
        elif model == "SK-VOTE":
            signals = []
            for r in ["A", "B", "C"]:
                apparent = x ^ provider.chance([1 - error, error])
                signals.append(provider.decide(r, f"apparent:{apparent}"))
            correct = provider.decide("receiver", f"count:{sum(signals)}") == x
            terms = 3
        elif model == "SK-TEAM":
            pair = provider.decide("sender", f"state:{x}")
            a = provider.decide("R1", f"signal:{pair // 2}")
            b = provider.decide("R2", f"signal:{pair % 2}")
            correct = 2 * a + b == x
            terms = 2
        else:
            signal = provider.decide("sender", f"state:{x}")
            translated = provider.decide("translator", f"signal:{signal}")
            correct = provider.decide("receiver", f"signal:{translated}") == x
            terms = 2
        return ParticipantOutcome(correct, (float(correct),) * len(roles), terms)

    fidelity = {
        "game": {
            "status": "explicit_variant"
            if model == "SK-DIALOGUE" or p.get("cost_case")
            else "paper_defined",
            "source": SKYRMS_SOURCE,
            "return_semantics": (
                "printed_normal_form_payoff_per_encounter"
                if p.get("cost_interpretation") == "table_literal"
                else "state_contingent_participant_return"
            ),
            "questions": (
                ["SKQ4"]
                if model == "SK-DIALOGUE"
                else ["SKQ2"]
                if p.get("cost_case") in {"listening", "state_specific"}
                else []
            ),
        },
    }
    if (
        (
            model in {"SK-VOTE", "SK-TEAM", "SK-CHAIN"}
            and not np.allclose(nature, 1 / n, rtol=0, atol=1e-15)
        )
        or (model == "SK-VOTE" and error != 0.1)
        or (
            model == "SK-SIG"
            and (
                (n, signals, actions) != (n, n, n)
                or (
                    not p.get("cost_case")
                    and bool(
                        {
                            "payoffs",
                            "success_matrix",
                            "background",
                            "signal_costs",
                            "listening_cost",
                        }
                        & (parameters or {}).keys()
                    )
                )
            )
        )
    ):
        fidelity["game"]["status"] = "explicit_parameter_variant"
    fidelity["game"]["nature_distribution"] = nature.tolist()
    if model == "SK-COMP":
        fidelity["game"]["parameter_note"] = (
            "The paper permits any positive state distribution; this profile records its choice."
        )
    game = FiniteGame(
        model,
        roles,
        tuple(rows),
        interaction,
        p,
        fidelity,
        fixed_decisions or {},
        **budgets,
    )
    if model == "SK-SIG":
        # Contingency costs are exact role-plan arrays. Keep their construction
        # deferred so declaration and sampled execution use only decision rows.
        def receiver_costs():
            full = []
            for row in (r for r in rows if r.role == "receiver"):
                frozen = game.fixed_decisions.get("receiver/" + row.context)
                if frozen is not None:
                    full.append(
                        np.full(game.plan_counts["receiver"], int(np.argmax(frozen)))
                    )
                else:
                    full.append(
                        game.plans["receiver"][
                            :, game.row_indices["receiver", row.context]
                        ]
                    )
            contingent = np.any(np.array(full) != full[0], axis=0)
            return contingent.astype(float) * listening

        if listening:
            for row in (r for r in rows if r.role == "receiver"):
                frozen = game.fixed_decisions.get("receiver/" + row.context)
                if frozen is not None and np.count_nonzero(frozen) != 1:
                    raise ValueError(
                        "Listening with stochastic frozen rows needs a complete plan; SKQ3"
                    )
            game.strategy_costs.defer("receiver", receiver_costs)
        if p.get("cost_interpretation") == "table_literal":
            if game.fixed_decisions:
                raise ValueError(
                    "Table-literal profiles require unreduced complete strategies"
                )
            # Paper ordering: identity, inverse, constant first, constant second.
            base = np.array(
                [
                    [2, 1, 1.5, 1.5],
                    [1, 2, 1.5, 1.5],
                    [1.5, 1.5, 1.5, 1.5],
                    [1.5, 1.5, 1.5, 1.5],
                ]
            )
            if case != "sending":
                # Preserve the published decimal table, including its rounding.
                base[:, 2] = 1.33
                base[:, 3] = 1.67
            sending = (
                [c, c, 2 * c, 0]
                if case == "sending"
                else [0.1, 0.2, 0.3, 0]
                if case == "listening"
                else [0, 0.3, 0.2, 0.1]
            )
            order = [2, 0, 1, 3]
            sender_matrix = base - np.array(sending)[:, None]
            receiver_matrix = base - np.array([listening, listening, 0, 0])[None, :]
            game.payoff_matrices = (
                sender_matrix[np.ix_(order, order)],
                receiver_matrix[np.ix_(order, order)],
            )
            game.strategy_costs.defer(
                "receiver", lambda: game._zero_strategy_costs("receiver")
            )
    return game
