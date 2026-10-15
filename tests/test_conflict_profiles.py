"""Source payoff transcriptions, explicit local credit, and bounded comparators."""

import copy

import numpy as np
import pytest

from self_assembling_games import (
    AdaptiveProfile,
    adaptive_profiles,
    build,
    build_adaptive,
    evaluate,
    model,
    profiles,
    restore_adaptive,
)
from self_assembling_games.conflict_profiles import (
    GM16,
    GSM13,
    ZBH13,
    conflict_profiles,
    gm16_table1_profile,
    scod_conflict_profile,
    signalling_information,
    signalling_information_from_policies,
    zbh13_cost_profile,
)
from self_assembling_games.simulation import Simulation


@pytest.mark.parametrize("profile", conflict_profiles(), ids=lambda p: p["id"])
def test_named_conditional_comparators_restore_and_report_owner_returns(profile):
    simulation = build(profile, seed=4)
    simulation.run(plays=5)
    before = simulation.snapshot()
    result = evaluate(simulation)
    assert result["completed_mass"] == pytest.approx(1)
    assert set(result["expected_owner_returns"]) == set(
        simulation.runtime.owners.values()
    )
    assert simulation.snapshot() == before
    restored = Simulation.restore(before)
    simulation.run(plays=5)
    restored.run(plays=5)
    assert restored.snapshot() == simulation.snapshot()


def test_conflict_presets_are_discoverable_and_declarations_are_independent():
    assert {p["id"] for p in conflict_profiles()} <= {p["id"] for p in profiles()}
    assert {
        "gm16.table1.evolution",
        "zbh13.separating.evolution",
        "zbh13.hybrid-equilibrium.evolution",
    } <= {p["id"] for p in adaptive_profiles()}
    declaration = gm16_table1_profile()
    declaration["parameters"]["owner_payoffs"]["sender"][0][0] = -100
    assert declaration["parameters"]["feedback_payoffs"]["sender"][0][0] == 5
    assert gm16_table1_profile()["parameters"]["owner_payoffs"]["sender"][0][0] == 5


def test_gm_table1_credits_actual_recipient_utilities_and_keeps_task_score_separate():
    simulation = build("gm16.table1.r1")
    simulation.runtime.forced = [0, 0, 1]  # S1, message1, A2
    simulation.runtime.record = True
    result = simulation.step()
    assert not result.success and result.payoff == 0
    assert result.owner_returns == {"sender": 6, "receiver": 5}
    assert simulation.runtime.tables["sender/0"].weights.tolist() == [7, 1, 1]
    assert simulation.runtime.tables["receiver/0"].weights.tolist() == [1, 6, 1]
    credit = [e for e in simulation.runtime.events if e["event"] == "feedback"]
    assert [(e["table"], e["payoff"]) for e in credit] == [
        ("sender/0", 6),
        ("receiver/0", 5),
    ]


def test_gm_table1_complete_plan_fitness_matches_direct_independent_enumeration():
    simulation = build("gm16.table1.evolution", seed=2)
    assert simulation.population.counts == {"receiver": 27, "sender": 27}
    sender_utility = np.array([[5, 6, 0], [2, 0, 6], [0, 1, 5]])
    receiver_utility = np.array([[0, 5, 6], [4, 0, 6], [5, 6, 3]])
    plans = simulation.game.plans
    state = np.arange(3)
    payoffs = [np.empty((27, 27)), np.empty((27, 27))]
    for i, sender in enumerate(plans["sender"]):
        for j, receiver in enumerate(plans["receiver"]):
            act = receiver[sender]
            payoffs[0][i, j] = np.mean(sender_utility[state, act])
            payoffs[1][i, j] = np.mean(receiver_utility[state, act])
    expected = simulation._expectations(simulation.frequencies)
    np.testing.assert_allclose(
        expected["fitness"]["sender"], payoffs[0] @ simulation.frequencies["receiver"]
    )
    np.testing.assert_allclose(
        expected["fitness"]["receiver"], payoffs[1].T @ simulation.frequencies["sender"]
    )
    assert GM16 in simulation.game.fidelity["game"]["source"]
    assert GSM13 in simulation.game.fidelity["game"]["source"]
    assert simulation.profile.kind == "paper_game"
    simulation.run(duration=0.05)
    assert restore_adaptive(simulation.snapshot()).snapshot() == simulation.snapshot()


@pytest.mark.parametrize(
    "regime,sender_probs,receiver_probs,returns",
    [
        ("separating", [1, 0], [0, 1], {"sender": 0.225, "receiver": 1}),
        (
            "hybrid-equilibrium",
            [1, 1 / 3],
            [0, 0.4],
            {"sender": 0.075, "receiver": 0.75},
        ),
    ],
)
def test_cost_game_source_equilibria_and_raw_costs(
    regime, sender_probs, receiver_probs, returns
):
    simulation = build(f"zbh13.{regime}.evolution")
    for role, probabilities in (("sender", sender_probs), ("receiver", receiver_probs)):
        p = np.array(probabilities)
        plan_frequencies = np.prod(
            np.where(simulation.game.plans[role] == 1, p, 1 - p), axis=1
        )
        simulation.set_population(role, plan_frequencies)
    result = simulation.evaluate()
    assert result["expected_returns"] == pytest.approx(returns)
    assert max(result["stationarity_residual"].values()) < 1e-15
    assert result["completed_mass"] == pytest.approx(1)
    assert not simulation.game.has_strategy_costs
    assert ZBH13 in simulation.game.fidelity["game"]["source"]
    assert min(simulation.game.returns[:, 0]) < 0
    raw = zbh13_cost_profile(regime)
    raw["learner"] = {"rule": "R1"}
    with pytest.raises(ValueError, match="recipient-local feedback"):
        build(raw)


def test_scod_zero_mixture_preserves_existing_common_interest_trajectory():
    original = build("scod.demo", seed=11)
    variant = build("scod.conflict.lambda-0.r1", seed=11)
    for _ in range(50):
        assert original.step().success == variant.step().success
    assert (
        original.runtime.rng.bit_generator.state
        == variant.runtime.rng.bit_generator.state
    )
    for address, table in original.runtime.tables.items():
        np.testing.assert_array_equal(
            table.weights, variant.runtime.tables[address].weights
        )
    assert len(variant.runtime.tables) == 17
    assert sum(len(table.weights) for table in variant.runtime.tables.values()) == 51


@pytest.mark.parametrize("mixture", [0, 0.25, 0.5, 0.75, 1])
def test_scod_conflict_formula_and_source_participation(mixture):
    simulation = build(scod_conflict_profile(mixture))
    parameters = simulation.profile.parameters
    for state in range(4):
        for act in range(4):
            assert parameters["owner_payoffs"]["A"][state][act] == (
                (1 - mixture) * (act == state) + mixture * (act == (state + 1) % 4)
            )
            assert parameters["owner_payoffs"]["B"][state][act] == (
                (1 - mixture) * (act == state) + mixture * (act == (state + 3) % 4)
            )
    before = {k: table.weights.copy() for k, table in simulation.runtime.tables.items()}
    simulation.runtime.record = True
    # State0, A0, B0, attention A, decoder act1: only A receives positive feedback.
    simulation.runtime.forced = [0, 0, 0, 0, 1]
    outcome = simulation.step()
    assert outcome.owner_returns == {"A": mixture, "B": 0, "receiver": 0}
    credit = [e for e in simulation.runtime.events if e["event"] == "feedback"]
    assert [(e["table"], e["payoff"]) for e in credit] == [
        ("A/0", mixture),
        ("B/0", 0),
        ('salience/"start"', 0),
        ("RA/0", 0),
    ]
    for address, table in simulation.runtime.tables.items():
        expected = before[address].copy()
        if address == "A/0":
            expected[0] += mixture
        np.testing.assert_array_equal(table.weights, expected)


def test_scod_receiver_return_is_counted_once_and_frozen_sender_keeps_state():
    simulation = build("scod.conflict.lambda-0.75.r1")
    simulation.runtime.freeze("A")
    before = copy.deepcopy(simulation.runtime.tables["A/0"].to_dict())
    simulation.runtime.forced = [0, 0, 0, 2, 0]  # identity act; salience AB
    outcome = simulation.step()
    assert outcome.owner_returns == {"A": 0.25, "B": 0.25, "receiver": 1}
    assert simulation.metrics.summary()["mean_owner_returns"]["receiver"] == 1
    assert simulation.runtime.tables["A/0"].to_dict() == before
    assert len(simulation.runtime.tables) == 17


def test_local_feedback_does_not_follow_shared_owner_identity_or_owner_payoff():
    declaration = gm16_table1_profile()
    p = declaration["parameters"]
    p["owners"] = {"sender": "animal", "receiver": "animal"}
    p["owner_payoffs"] = {"animal": [[-2] * 3 for _ in range(3)]}
    p["payoffs"] = [[-3] * 3 for _ in range(3)]
    p["feedback_payoffs"] = {
        "sender": [[0.25] * 3 for _ in range(3)],
        "receiver": [[0.75] * 3 for _ in range(3)],
    }
    simulation = build(declaration)
    simulation.runtime.forced = [0, 0, 0]
    outcome = simulation.step()
    assert outcome.payoff == -3
    assert outcome.owner_returns == {"animal": -2}
    assert simulation.runtime.tables["sender/0"].weights[0] == 1.25
    assert simulation.runtime.tables["receiver/0"].weights[0] == 1.75


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda p: p["parameters"].pop("feedback_payoffs"), "require owner_payoffs"),
        (lambda p: p["parameters"]["owner_payoffs"].pop("receiver"), "recipients"),
        (
            lambda p: p["parameters"]["feedback_payoffs"].update(unknown=[[0] * 3] * 3),
            "recipients",
        ),
        (
            lambda p: p["parameters"]["owner_payoffs"].update(sender=[[0]]),
            "states × actions",
        ),
        (lambda p: p["learner"].update(rule="R2"), "R1 or SIGNED"),
    ],
)
def test_separate_return_configuration_rejects_ambiguous_or_incomplete_contract(
    change, match
):
    declaration = gm16_table1_profile()
    change(declaration)
    with pytest.raises(ValueError, match=match):
        build(declaration)


@pytest.mark.parametrize("value", [-0.1, 1.1, True, float("nan")])
def test_scod_mixture_validation(value):
    with pytest.raises(ValueError, match="mixture"):
        scod_conflict_profile(value)


def test_information_separates_informative_signals_from_information_use():
    identity = np.eye(3)
    perfect = signalling_information_from_policies([1 / 3] * 3, identity, identity)
    assert perfect["state_action_mutual_information_bits"] == pytest.approx(np.log2(3))
    assert perfect["state_signal_mutual_information_bits"] == pytest.approx(np.log2(3))
    constant = np.tile([1, 0, 0], (3, 1))
    ignored = signalling_information_from_policies([1 / 3] * 3, identity, constant)
    assert ignored["state_signal_mutual_information_bits"] == pytest.approx(np.log2(3))
    assert ignored["signal_action_mutual_information_bits"] == 0
    assert ignored["state_action_mutual_information_bits"] == 0


def test_information_helper_matches_conditional_and_complete_plan_policies():
    conditional = build("gm16.table1.r1")
    evolved = build("gm16.table1.evolution")
    for module in ("sender", "receiver"):
        for context in range(3):
            conditional.set_policy(module, context, context)
        identity = np.flatnonzero(
            np.all(evolved.game.plans[module] == [0, 1, 2], axis=1)
        )[0]
        frequencies = np.zeros(27)
        frequencies[identity] = 1
        evolved.set_population(module, frequencies)
    before = [conditional.snapshot(), evolved.snapshot()]
    learned_info = signalling_information(conditional)
    assert learned_info == signalling_information(evolved)
    assert learned_info["state_action_mutual_information_bits"] == pytest.approx(
        np.log2(3)
    )
    assert conditional.snapshot() == before[0]
    assert evolved.snapshot() == before[1]
    system = model("gm16.table1.evolution").instantiate(seed=3)
    state = system.snapshot()
    assert signalling_information(system)["protocol"] == "exact_signalling_policy"
    assert system.snapshot() == state


def test_information_checks_policy_probabilities_and_joint_matching_assumption():
    with pytest.raises(ValueError, match="probability vector"):
        signalling_information_from_policies([0.5, 0.5], [[1, 1], [0, 1]], np.eye(2))
    declaration = gm16_table1_profile()
    parameters = declaration["parameters"]
    parameters["owners"] = {"sender": "animal", "receiver": "animal"}
    parameters["owner_payoffs"] = {"animal": [[1] * 3 for _ in range(3)]}
    simulation = build_adaptive(
        AdaptiveProfile("joint-owner", "SIG", {"profile": declaration})
    )
    with pytest.raises(ValueError, match="distinct encounter participants"):
        signalling_information(simulation)
